from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig
from recursive_training_engine.kernels import optimized as K
from recursive_training_engine.recipes import RecipeBank, RecipeSpec


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
        cols = self.recipe_cols[spec.recipe_id, :active_dim]
        qkv_cols = self.recipe_qkv_cols[spec.recipe_id, : 3 * active_dim]
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
        slabs = self.recipe_slabs[spec.recipe_id, :active_dim]
        gate_cols = self.recipe_gate_cols[spec.recipe_id, : 2 * active_dim]
        out = x.new_zeros(*x.shape)
        for bank in spec.ffn_banks:
            fused = x @ self.wug[bank, :, gate_cols]
            up, gate = fused.split(slabs.numel(), dim=-1)
            out = out + (up * F.silu(gate)) @ self.wd[bank, slabs, :]
        return out / max(len(spec.ffn_banks), 1)


class BankedRecursiveCore(nn.Module):
    def __init__(self, config: ModelConfig, recipe_bank: RecipeBank):
        super().__init__()
        self.config = config
        self.recipe_bank = recipe_bank
        self.norm1 = RMSNorm(config.d_model)
        self.attn = BankedAttention(config, recipe_bank)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = BankedSwiGLU(config, recipe_bank)
        if config.use_recursive_input_skip:
            self.alpha_inj = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("alpha_inj", None)

    def step_group(self, h: torch.Tensor, h0: torch.Tensor, spec: RecipeSpec) -> torch.Tensor:
        h_tilde = h if self.alpha_inj is None else h + self.alpha_inj * h0
        u = h_tilde + self.attn(self.norm1(h_tilde), spec)
        return u + self.mlp(self.norm2(u), spec)

    def forward_step(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = h.clone()
        if active_mask is None:
            active_mask = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        for rid in torch.unique(recipe_ids[active_mask]).detach().cpu().tolist():
            mask = active_mask & (recipe_ids == int(rid))
            spec = self.recipe_bank.get_recipe(int(rid))
            assert isinstance(spec, RecipeSpec)
            out[mask] = self.step_group(h[mask], h0[mask], spec)
        return out
