from __future__ import annotations

import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig
from recursive_training_engine.kernels import optimized as K
from recursive_training_engine.kernels.svd_sparse_ffn_triton import triton_svd_sparse_ffn_forward
from recursive_training_engine.recipes import RecipeBank, RecipeSpec, spec_from_factors


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return K.k_fused_rmsnorm(x, self.weight, eps=self.eps)[0]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = q.shape[-2]
        cos = self.cos[:s].to(q.device, q.dtype)
        sin = self.sin[:s].to(q.device, q.dtype)
        return K.k_rope_apply(q, k, cos, sin)


class DenseCausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.wqkv = nn.Linear(d, 3 * d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rope = RotaryEmbedding(self.head_dim) if config.use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.wqkv(x).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k)
        y = K.k_flash_causal_dense(q, k, v)
        y = y.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(y)


class DenseSwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.wug = nn.Linear(d, 2 * config.d_ff, bias=False)
        self.wd = nn.Linear(config.d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.wug(x).chunk(2, dim=-1)
        return self.wd(up * F.silu(gate))


class SVDFactorSparseFFN(nn.Module):
    """Token-level sparse SwiGLU FFN with SVD factor candidate retrieval.

    This is the executable version of the SVD factor-union oracle: candidate
    neurons are retrieved from low-rank approximations to W_up/W_gate, then only
    those candidate rows are evaluated exactly.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        rank: int,
        top_k: int,
        up_m: int,
        gate_m: int | None = None,
        product_m: int = 0,
        candidate_mode: str = "mask",
        refresh_on_init: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.rank = min(max(1, int(rank)), self.d_model, self.d_ff)
        self.top_k = min(max(1, int(top_k)), self.d_ff)
        self.up_m = min(max(0, int(up_m)), self.d_ff)
        self.gate_m = self.up_m if gate_m is None else min(max(0, int(gate_m)), self.d_ff)
        self.product_m = min(max(0, int(product_m)), self.d_ff)
        if candidate_mode not in {"mask", "slots", "triton"}:
            raise ValueError("candidate_mode must be 'mask', 'slots', or 'triton'")
        self.candidate_mode = candidate_mode
        self.refresh_on_init = bool(refresh_on_init)
        self.w_up = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w_gate = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.w_down = nn.Parameter(torch.empty(self.d_ff, self.d_model))
        self.register_buffer("up_a", torch.empty(self.d_model, self.rank), persistent=False)
        self.register_buffer("up_b", torch.empty(self.rank, self.d_ff), persistent=False)
        self.register_buffer("gate_a", torch.empty(self.d_model, self.rank), persistent=False)
        self.register_buffer("gate_b", torch.empty(self.rank, self.d_ff), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        scale = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.w_up, std=scale)
        nn.init.normal_(self.w_gate, std=scale)
        nn.init.normal_(self.w_down, std=scale)
        if self.refresh_on_init:
            self.refresh_svd()
        else:
            nn.init.normal_(self.up_a, std=scale)
            nn.init.normal_(self.up_b, std=scale)
            nn.init.normal_(self.gate_a, std=scale)
            nn.init.normal_(self.gate_b, std=scale)

    @classmethod
    def from_dense(
        cls,
        mlp: DenseSwiGLU,
        *,
        rank: int,
        top_k: int,
        up_m: int,
        gate_m: int | None = None,
        product_m: int = 0,
        candidate_mode: str = "mask",
    ) -> "SVDFactorSparseFFN":
        module = cls(
            mlp.wug.in_features,
            mlp.wd.in_features,
            rank=rank,
            top_k=top_k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            candidate_mode=candidate_mode,
        ).to(device=mlp.wug.weight.device, dtype=mlp.wug.weight.dtype)
        with torch.no_grad():
            up, gate = mlp.wug.weight.detach().chunk(2, dim=0)
            module.w_up.copy_(up)
            module.w_gate.copy_(gate)
            module.w_down.copy_(mlp.wd.weight.detach().t())
            module.refresh_svd()
        return module

    @torch.no_grad()
    def refresh_svd(self) -> None:
        for prefix, weight in (("up", self.w_up), ("gate", self.w_gate)):
            mat = weight.detach().float().t().cpu().contiguous()
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            rank = self.rank
            getattr(self, f"{prefix}_a").copy_((u[:, :rank] * s[:rank]).to(weight.device, weight.dtype))
            getattr(self, f"{prefix}_b").copy_(vh[:rank, :].to(weight.device, weight.dtype))

    def _add_top_candidates(
        self,
        candidate_mask: torch.Tensor,
        scores: torch.Tensor,
        count: int,
    ) -> None:
        if count <= 0:
            return
        ids = torch.topk(scores, k=min(count, self.d_ff), dim=-1).indices
        candidate_mask.scatter_(-1, ids, True)

    def _top_candidate_ids(self, scores: torch.Tensor, count: int) -> torch.Tensor | None:
        if count <= 0:
            return None
        return torch.topk(scores, k=min(count, self.d_ff), dim=-1).indices

    def _first_occurrence_mask(self, ids: torch.Tensor) -> torch.Tensor:
        slots = ids.shape[-1]
        if slots <= 1:
            return torch.ones_like(ids, dtype=torch.bool)
        same = ids.unsqueeze(2).eq(ids.unsqueeze(1))
        earlier = torch.triu(
            torch.ones(slots, slots, device=ids.device, dtype=torch.bool),
            diagonal=1,
        )
        return ~(same & earlier).any(dim=1)

    def _candidate_ids_from_mask(
        self,
        up_scores: torch.Tensor,
        gate_scores: torch.Tensor,
        product_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_mask = torch.zeros(
            up_scores.shape[0],
            self.d_ff,
            device=up_scores.device,
            dtype=torch.bool,
        )
        self._add_top_candidates(candidate_mask, up_scores, self.up_m)
        self._add_top_candidates(candidate_mask, gate_scores, self.gate_m)
        self._add_top_candidates(candidate_mask, product_scores, self.product_m)
        too_small = candidate_mask.sum(dim=-1) < self.top_k
        if bool(too_small.any()):
            fill_ids = torch.topk(product_scores[too_small], k=self.top_k, dim=-1).indices
            candidate_mask[too_small].scatter_(-1, fill_ids, True)
        max_candidates = min(
            self.d_ff,
            max(self.top_k, self.up_m + self.gate_m + self.product_m),
        )
        candidate_ids = torch.topk(
            candidate_mask.to(torch.int64),
            k=max_candidates,
            dim=-1,
        ).indices
        candidate_valid = candidate_mask.gather(dim=-1, index=candidate_ids)
        return candidate_ids, candidate_valid, candidate_mask.sum(dim=-1).float()

    def _candidate_ids_from_slots(
        self,
        up_scores: torch.Tensor,
        gate_scores: torch.Tensor,
        product_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pieces = [
            ids
            for ids in (
                self._top_candidate_ids(up_scores, self.up_m),
                self._top_candidate_ids(gate_scores, self.gate_m),
                self._top_candidate_ids(product_scores, self.product_m),
            )
            if ids is not None
        ]
        if not pieces:
            pieces = [torch.topk(product_scores, k=self.top_k, dim=-1).indices]
        candidate_ids = torch.cat(pieces, dim=-1)
        candidate_valid = self._first_occurrence_mask(candidate_ids)
        if bool((candidate_valid.sum(dim=-1) < self.top_k).any()):
            fallback_ids = torch.topk(product_scores, k=self.top_k, dim=-1).indices
            candidate_ids = torch.cat([candidate_ids, fallback_ids], dim=-1)
            candidate_valid = self._first_occurrence_mask(candidate_ids)
        return candidate_ids, candidate_valid, candidate_valid.sum(dim=-1).float()

    def _sync_for_profile(self, device: torch.device) -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_aux: bool = False,
        profile: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        timings: dict[str, float] = {}

        def mark(name: str, previous: float | None = None) -> float:
            if not profile:
                return 0.0
            self._sync_for_profile(x.device)
            now = time.perf_counter()
            if previous is not None:
                timings[name] = now - previous
            return now

        t0 = mark("start")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.d_model)
        wd_norm = self.w_down.detach().float().norm(dim=-1).to(flat.device, flat.dtype)
        if self.candidate_mode == "triton":
            t_start = mark("start")
            out = triton_svd_sparse_ffn_forward(
                flat.contiguous(),
                self.w_up.contiguous(),
                self.w_gate.contiguous(),
                self.w_down.contiguous(),
                self.up_a.contiguous(),
                self.up_b.contiguous(),
                self.gate_a.contiguous(),
                self.gate_b.contiguous(),
                wd_norm.contiguous(),
                top_k=self.top_k,
                up_m=self.up_m,
                gate_m=self.gate_m,
                product_m=self.product_m,
            )
            mark("triton_sparse_ffn_time", t_start)
            out = out.view(*original_shape, self.d_model)
            if not return_aux:
                return out
            aux: dict[str, torch.Tensor | float] = {
                "avg_candidate_size": float(self.up_m + self.gate_m + self.product_m),
                "candidate_slots": float(self.up_m + self.gate_m + self.product_m),
                "candidate_mode": self.candidate_mode,
            }
            aux.update(timings)
            return out, aux

        q_up = flat @ self.up_a
        q_gate = flat @ self.gate_a
        up_hat = q_up @ self.up_b
        gate_hat = q_gate @ self.gate_b
        gate_hat_act = F.silu(gate_hat)
        product_scores = (up_hat * gate_hat_act).detach().abs() * wd_norm
        up_scores = up_hat.detach().abs() * wd_norm
        gate_scores = gate_hat_act.detach().abs() * wd_norm
        t1 = mark("selector_score_time", t0)
        if self.candidate_mode == "mask":
            candidate_ids, candidate_valid, candidate_sizes = self._candidate_ids_from_mask(
                up_scores,
                gate_scores,
                product_scores,
            )
        else:
            candidate_ids, candidate_valid, candidate_sizes = self._candidate_ids_from_slots(
                up_scores,
                gate_scores,
                product_scores,
            )
        t2 = mark("candidate_union_dedup_time", t1)
        up_rows = self.w_up[candidate_ids]
        gate_rows = self.w_gate[candidate_ids]
        exact_up = torch.einsum("nd,ncd->nc", flat, up_rows)
        exact_gate = torch.einsum("nd,ncd->nc", flat, gate_rows)
        z = exact_up * F.silu(exact_gate)
        t3 = mark("exact_candidate_activation_time", t2)
        exact_scores = (z.detach().abs() * wd_norm[candidate_ids]).masked_fill(
            ~candidate_valid,
            -torch.inf,
        )
        selected_local = torch.topk(exact_scores, k=self.top_k, dim=-1).indices
        selected_ids = candidate_ids.gather(dim=-1, index=selected_local)
        selected_z = z.gather(dim=-1, index=selected_local)
        t4 = mark("rerank_topk_time", t3)
        out = (selected_z.unsqueeze(-1) * self.w_down[selected_ids]).sum(dim=1)
        mark("down_sum_time", t4)
        out = out.view(*original_shape, self.d_model)
        if not return_aux:
            return out
        aux: dict[str, torch.Tensor | float] = {
            "candidate_ids": candidate_ids,
            "candidate_valid": candidate_valid,
            "selected_ids": selected_ids,
            "avg_candidate_size": float(candidate_sizes.mean().detach().cpu()),
            "candidate_slots": float(candidate_ids.shape[-1]),
            "candidate_mode": self.candidate_mode,
        }
        aux.update(timings)
        return out, aux


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = DenseCausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = DenseSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x + self.attn(self.norm1(x))
        return u + self.mlp(self.norm2(u))


class BankedAttention(nn.Module):
    def __init__(self, config: ModelConfig, recipe_bank: RecipeBank):
        super().__init__()
        self.config = config
        d = config.d_model
        scale = 1.0 / math.sqrt(d)
        self.wqkv = nn.Parameter(torch.randn(config.attn_banks, d, 3 * d) * scale)
        self.wo = nn.Parameter(torch.randn(config.attn_banks, d, d) * scale)
        self.rope = RotaryEmbedding(d // config.n_heads) if config.use_rope else None
        cols, qkv_cols, lengths = self._build_recipe_cols(recipe_bank)
        self.register_buffer("recipe_cols", cols, persistent=False)
        self.register_buffer("recipe_qkv_cols", qkv_cols, persistent=False)
        self.register_buffer("recipe_col_lengths", lengths, persistent=False)

    def _cols_for_groups(self, spec: RecipeSpec) -> list[int]:
        heads_per_group = self.config.n_heads // self.config.head_groups
        head_dim = self.config.d_model // self.config.n_heads
        cols: list[int] = []
        for group in spec.head_groups:
            start_head = group * heads_per_group
            for head in range(start_head, start_head + heads_per_group):
                start = head * head_dim
                cols.extend(range(start, start + head_dim))
        return cols

    def _build_recipe_cols(self, recipe_bank: RecipeBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cols = torch.zeros(self.config.recipe_count, self.config.d_model, dtype=torch.long)
        qkv_cols = torch.zeros(self.config.recipe_count, 3 * self.config.d_model, dtype=torch.long)
        lengths = torch.zeros(self.config.recipe_count, dtype=torch.long)
        for spec in recipe_bank.recipes:
            values = self._cols_for_groups(spec)
            lengths[spec.recipe_id] = len(values)
            value_tensor = torch.tensor(values, dtype=torch.long)
            cols[spec.recipe_id, : len(values)] = value_tensor
            qkv_cols[spec.recipe_id, : 3 * len(values)] = torch.cat(
                [
                    value_tensor,
                    value_tensor + self.config.d_model,
                    value_tensor + 2 * self.config.d_model,
                ]
            )
        return cols, qkv_cols, lengths

    def forward(self, x: torch.Tensor, spec: RecipeSpec) -> torch.Tensor:
        b, s, d = x.shape
        heads_per_group = self.config.n_heads // self.config.head_groups
        head_dim = d // self.config.n_heads
        active_dim = len(spec.head_groups) * heads_per_group * head_dim
        if spec.recipe_id >= 0:
            cols = self.recipe_cols[spec.recipe_id, :active_dim]
            qkv_cols = self.recipe_qkv_cols[spec.recipe_id, : 3 * active_dim]
        else:
            cols = torch.tensor(self._cols_for_groups(spec), dtype=torch.long, device=x.device)
            qkv_cols = torch.cat(
                [cols, cols + self.config.d_model, cols + 2 * self.config.d_model]
            )
        active_heads = active_dim // head_dim
        out = x.new_zeros(b, s, d)
        for bank in spec.attention_banks:
            qkv = x @ self.wqkv[bank, :, qkv_cols]
            q, k, v = qkv.split(active_dim, dim=-1)
            q = q.view(b, s, active_heads, head_dim).transpose(1, 2)
            k = k.view(b, s, active_heads, head_dim).transpose(1, 2)
            v = v.view(b, s, active_heads, head_dim).transpose(1, 2)
            if self.rope is not None:
                q, k = self.rope(q, k)
            y = K.k_flash_causal_grouped(q, k, v)
            y = y.transpose(1, 2).contiguous().view(b, s, active_dim)
            out = out + K.k_out_proj_grouped(y, self.wo[bank], cols)
        return out / max(len(spec.attention_banks), 1)


class BankedSwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, recipe_bank: RecipeBank):
        super().__init__()
        self.config = config
        d = config.d_model
        scale = 1.0 / math.sqrt(d)
        self.wug = nn.Parameter(torch.randn(config.ffn_banks, d, 2 * config.d_ff) * scale)
        self.wd = nn.Parameter(torch.randn(config.ffn_banks, config.d_ff, d) * scale)
        slabs, gate_cols, lengths = self._build_recipe_slabs(recipe_bank)
        self.register_buffer("recipe_slabs", slabs, persistent=False)
        self.register_buffer("recipe_gate_cols", gate_cols, persistent=False)
        self.register_buffer("recipe_slab_lengths", lengths, persistent=False)

    def _slabs_for_groups(self, spec: RecipeSpec) -> list[int]:
        slab = self.config.d_ff // self.config.ffn_groups
        cols: list[int] = []
        for group in spec.ffn_groups:
            cols.extend(range(group * slab, (group + 1) * slab))
        return cols

    def _build_recipe_slabs(self, recipe_bank: RecipeBank) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slabs = torch.zeros(self.config.recipe_count, self.config.d_ff, dtype=torch.long)
        gate_cols = torch.zeros(self.config.recipe_count, 2 * self.config.d_ff, dtype=torch.long)
        lengths = torch.zeros(self.config.recipe_count, dtype=torch.long)
        for spec in recipe_bank.recipes:
            values = self._slabs_for_groups(spec)
            lengths[spec.recipe_id] = len(values)
            value_tensor = torch.tensor(values, dtype=torch.long)
            slabs[spec.recipe_id, : len(values)] = value_tensor
            gate_cols[spec.recipe_id, : 2 * len(values)] = torch.cat(
                [value_tensor, value_tensor + self.config.d_ff]
            )
        return slabs, gate_cols, lengths

    def forward(self, x: torch.Tensor, spec: RecipeSpec) -> torch.Tensor:
        slab = self.config.d_ff // self.config.ffn_groups
        active_dim = len(spec.ffn_groups) * slab
        if spec.recipe_id >= 0:
            slabs = self.recipe_slabs[spec.recipe_id, :active_dim]
            gate_cols = self.recipe_gate_cols[spec.recipe_id, : 2 * active_dim]
        else:
            slabs = torch.tensor(self._slabs_for_groups(spec), dtype=torch.long, device=x.device)
            gate_cols = torch.cat([slabs, slabs + self.config.d_ff])
        out = x.new_zeros(*x.shape)
        for bank in spec.ffn_banks:
            fused = x @ self.wug[bank, :, gate_cols]
            up, gate = fused.split(slabs.numel(), dim=-1)
            out = out + (up * F.silu(gate)) @ self.wd[bank, slabs, :]
        return out / max(len(spec.ffn_banks), 1)


class GlobalLowRankCorrector(nn.Module):
    def __init__(self, d_model: int, t_max: int, rank: int):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_h0 = RMSNorm(d_model)
        self.pass_emb = nn.Embedding(t_max, rank)
        self.v_h = nn.Linear(d_model, rank, bias=False)
        self.v_h0 = nn.Linear(d_model, rank, bias=False)
        self.u = nn.Linear(rank, d_model, bias=False)
        self.pass_gate = nn.Parameter(torch.ones(t_max, rank))
        self.pass_scale = nn.Parameter(torch.zeros(t_max))

    def forward(self, h: torch.Tensor, h0: torch.Tensor, pass_idx: int) -> torch.Tensor:
        pass_idx = int(pass_idx)
        z = (
            self.v_h(self.norm_h(h))
            + self.v_h0(self.norm_h0(h0))
            + self.pass_emb.weight[pass_idx]
        )
        z = F.silu(z) * self.pass_gate[pass_idx]
        scale = 0.1 * torch.tanh(self.pass_scale[pass_idx]) + 0.1
        return scale * self.u(z)


class BankedRecursiveCore(nn.Module):
    def __init__(self, config: ModelConfig, recipe_bank: RecipeBank):
        super().__init__()
        self.config = config
        self.recipe_bank = recipe_bank
        self.norm1 = RMSNorm(config.d_model)
        self.attn = BankedAttention(config, recipe_bank)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = BankedSwiGLU(config, recipe_bank)
        self.pass_gamma1 = nn.Parameter(torch.zeros(config.t_max, config.d_model))
        self.pass_beta1 = nn.Parameter(torch.zeros(config.t_max, config.d_model))
        self.pass_gamma2 = nn.Parameter(torch.zeros(config.t_max, config.d_model))
        self.pass_beta2 = nn.Parameter(torch.zeros(config.t_max, config.d_model))
        self.pass_attn_scale = nn.Parameter(torch.full((config.t_max,), 1.0))
        self.pass_mlp_scale = nn.Parameter(torch.full((config.t_max,), 1.0))
        self.global_corrector = (
            GlobalLowRankCorrector(config.d_model, config.t_max, config.global_corrector_rank)
            if config.use_global_lowrank_corrector
            else None
        )
        if config.use_recursive_input_skip:
            self.alpha_inj = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("alpha_inj", None)

    def step_group(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        spec: RecipeSpec,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        pass_idx = int(pass_idx)
        h_tilde = h if self.alpha_inj is None else h + self.alpha_inj * h0

        n1 = self.norm1(h_tilde)
        n1 = n1 * (1.0 + self.pass_gamma1[pass_idx]) + self.pass_beta1[pass_idx]
        attn_out = self.attn(n1, spec)
        u = h_tilde + self.pass_attn_scale[pass_idx] * attn_out

        n2 = self.norm2(u)
        n2 = n2 * (1.0 + self.pass_gamma2[pass_idx]) + self.pass_beta2[pass_idx]
        mlp_out = self.mlp(n2, spec)
        sparse_out = u + self.pass_mlp_scale[pass_idx] * mlp_out
        if self.global_corrector is None:
            return sparse_out
        return sparse_out + self.global_corrector(sparse_out, h0, pass_idx)

    def forward_step(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        out = h.clone()
        if active_mask is None:
            active_mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        for rid in torch.unique(recipe_ids[active_mask]).detach().cpu().tolist():
            mask = active_mask & (recipe_ids == int(rid))
            spec = self.recipe_bank.get_recipe(int(rid))
            assert isinstance(spec, RecipeSpec)
            out[mask] = self.step_group(h[mask], h0[mask], spec, pass_idx=pass_idx)
        return out

    def deferred_grouped_block(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_schedule: list[int] | tuple[int, ...],
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = h.clone()
        if active_mask is None:
            active_mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        if not bool(active_mask.any().item()) or len(recipe_schedule) == 0:
            return out
        specs = []
        for recipe_id in recipe_schedule:
            spec = self.recipe_bank.get_recipe(int(recipe_id))
            assert isinstance(spec, RecipeSpec)
            specs.append(spec)

        h_active = h[active_mask]
        h0_active = h0[active_mask]
        h_tilde = h_active if self.alpha_inj is None else h_active + self.alpha_inj * h0_active

        n1_base = self.norm1(h_tilde)
        attn_accum = torch.zeros_like(h_active)
        for pass_idx, spec in enumerate(specs):
            idx = min(pass_idx, self.config.t_max - 1)
            n1 = n1_base * (1.0 + self.pass_gamma1[idx]) + self.pass_beta1[idx]
            attn_accum = attn_accum + self.pass_attn_scale[idx] * self.attn(n1, spec)
        u = h_tilde + attn_accum

        n2_base = self.norm2(u)
        mlp_accum = torch.zeros_like(h_active)
        for pass_idx, spec in enumerate(specs):
            idx = min(pass_idx, self.config.t_max - 1)
            n2 = n2_base * (1.0 + self.pass_gamma2[idx]) + self.pass_beta2[idx]
            mlp_accum = mlp_accum + self.pass_mlp_scale[idx] * self.mlp(n2, spec)
        grouped_out = u + mlp_accum
        if self.global_corrector is not None:
            grouped_out = grouped_out + self.global_corrector(grouped_out, h0_active, 0)
        out[active_mask] = grouped_out
        return out

    def forward_step_factorized(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        attn_banks: torch.Tensor,
        ffn_banks: torch.Tensor,
        head_slots: torch.Tensor,
        ffn_slots: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        out = h.clone()
        if active_mask is None:
            active_mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        factors = torch.stack(
            [
                attn_banks.to(h.device).long(),
                ffn_banks.to(h.device).long(),
                head_slots.to(h.device).long(),
                ffn_slots.to(h.device).long(),
            ],
            dim=-1,
        )
        active_factors = factors[active_mask]
        if active_factors.numel() == 0:
            return out
        for attn_bank, ffn_bank, head_slot, ffn_slot in torch.unique(active_factors, dim=0).detach().cpu().tolist():
            factor = factors.new_tensor([attn_bank, ffn_bank, head_slot, ffn_slot])
            mask = active_mask & (factors == factor).all(dim=-1)
            spec = spec_from_factors(
                self.config,
                attn_bank=int(attn_bank),
                ffn_bank=int(ffn_bank),
                head_slot=int(head_slot),
                ffn_slot=int(ffn_slot),
            )
            out[mask] = self.step_group(h[mask], h0[mask], spec, pass_idx=pass_idx)
        return out
