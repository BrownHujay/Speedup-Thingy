from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig


@dataclass(slots=True)
class MacroTrace:
    states: dict[int, torch.Tensor]
    stride_sequence: list[list[int]]
    physical_passes: torch.Tensor
    decomposition_error: torch.Tensor | None = None


@dataclass(slots=True)
class MacroCompilerOutput:
    predicted_hidden: torch.Tensor
    predicted_logit_delta: torch.Tensor | None
    predicted_trace: torch.Tensor
    confidence: torch.Tensor


class MacroOperators(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.strides = list(config.depth_choices)
        self.stride_to_idx = {stride: i for i, stride in enumerate(self.strides)}
        r = config.recipe_count
        k = len(self.strides)
        d = config.d_model
        m = config.macro_rank
        input_dim = 3 * d
        if config.macro_include_delta_to_h0:
            input_dim += d
        if config.macro_use_depth_embedding:
            input_dim += d
            self.depth_embed = nn.Parameter(torch.randn(k, d) * (1.0 / max(d, 1) ** 0.5))
        else:
            self.register_parameter("depth_embed", None)
        hidden = max(m, m * config.macro_hidden_mult)
        self.d_gain = nn.Parameter(torch.zeros(r, k, d))
        self.v = nn.Parameter(torch.randn(r, k, input_dim, hidden) * (1.0 / input_dim**0.5))
        if hidden != m:
            self.v2 = nn.Parameter(torch.randn(r, k, hidden, m) * (1.0 / hidden**0.5))
        else:
            self.register_parameter("v2", None)
        self.u = nn.Parameter(torch.randn(r, k, m, d) * 1e-3)
        update_scale = torch.full((r, k, d), float(config.macro_update_scale_init))
        legacy_shape = (
            config.macro_hidden_mult == 1
            and not config.macro_use_gated_update
            and not config.macro_include_delta_to_h0
            and not config.macro_use_depth_embedding
            and config.macro_update_scale_init == 1.0
        )
        if legacy_shape:
            self.register_buffer("update_scale", update_scale, persistent=True)
        else:
            self.update_scale = nn.Parameter(update_scale)
        self.bias = nn.Parameter(torch.zeros(r, k, d))
        if config.macro_use_gated_update:
            self.gate_v = nn.Parameter(torch.randn(r, k, input_dim, m) * (1.0 / input_dim**0.5))
            self.gate_u = nn.Parameter(torch.randn(r, k, m, d) * 1e-3)
            self.gate_bias = nn.Parameter(torch.zeros(r, k, d))
        else:
            self.register_parameter("gate_v", None)
            self.register_parameter("gate_u", None)
            self.register_parameter("gate_bias", None)
        self.register_buffer(
            "depth_values", torch.tensor(self.strides, dtype=torch.long), persistent=False
        )
        pairs = [(i, self.stride_to_idx[stride * 2]) for i, stride in enumerate(self.strides) if stride * 2 in self.stride_to_idx]
        if pairs:
            self.register_buffer(
                "cons_src_idx", torch.tensor([p[0] for p in pairs], dtype=torch.long), persistent=False
            )
            self.register_buffer(
                "cons_dst_idx", torch.tensor([p[1] for p in pairs], dtype=torch.long), persistent=False
            )
        else:
            self.register_buffer("cons_src_idx", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("cons_dst_idx", torch.empty(0, dtype=torch.long), persistent=False)

    def apply_group(self, h: torch.Tensor, h0: torch.Tensor, recipe_id: int, stride: int) -> torch.Tensor:
        idx = self.stride_to_idx[int(stride)]
        recipe_ids = torch.full((h.shape[0],), recipe_id, dtype=torch.long, device=h.device)
        stride_ids = torch.full((h.shape[0],), idx, dtype=torch.long, device=h.device)
        return self.apply_vectorized(h, h0, recipe_ids, stride_ids)

    def stride_ids_for_depths(self, depths: torch.Tensor) -> torch.Tensor:
        matches = depths.unsqueeze(-1) == self.depth_values.to(depths.device).unsqueeze(0)
        return matches.float().argmax(dim=-1)

    def _decompose_one(self, depth: int) -> list[int]:
        if self.config.macro_decomposition == "direct":
            return [depth]
        if self.config.macro_decomposition in {"binary", "consistency_tree"} and depth > 1:
            half = depth // 2
            if half in self.stride_to_idx and depth % 2 == 0:
                return [half, half]
        remaining = depth
        out: list[int] = []
        greedy_strides = [stride for stride in sorted(self.strides, reverse=True) if stride < depth]
        if not greedy_strides:
            return [depth]
        for stride in greedy_strides:
            while remaining >= stride:
                out.append(stride)
                remaining -= stride
        if remaining:
            out.append(depth)
        return out or [depth]

    def apply_vectorized(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        stride_ids: torch.Tensor,
    ) -> torch.Tensor:
        gain = self.d_gain[recipe_ids, stride_ids]
        v = self.v[recipe_ids, stride_ids]
        u = self.u[recipe_ids, stride_ids]
        bias = self.bias[recipe_ids, stride_ids]
        scale = self.update_scale[recipe_ids, stride_ids]
        normed = h * torch.rsqrt(h.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(h.dtype)
        normed_h0 = h0 * torch.rsqrt(h0.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5).to(h0.dtype)
        pooled = normed.mean(dim=1, keepdim=True).expand_as(normed)
        parts = [normed, pooled, h0]
        if self.config.macro_include_delta_to_h0:
            parts.append(normed - normed_h0)
        if self.depth_embed is not None:
            depth = self.depth_embed[stride_ids].unsqueeze(1).expand_as(normed)
            parts.append(depth)
        z = torch.cat(parts, dim=-1)
        low = F.silu(torch.bmm(z, v))
        if self.v2 is not None:
            v2 = self.v2[recipe_ids, stride_ids]
            low = F.silu(torch.bmm(low, v2))
        low = torch.bmm(low, u)
        update = gain.unsqueeze(1) * h + low + bias.unsqueeze(1)
        if self.gate_v is not None and self.gate_u is not None and self.gate_bias is not None:
            gate_v = self.gate_v[recipe_ids, stride_ids]
            gate_u = self.gate_u[recipe_ids, stride_ids]
            gate_bias = self.gate_bias[recipe_ids, stride_ids]
            gate = torch.sigmoid(torch.bmm(F.silu(torch.bmm(z, gate_v)), gate_u) + gate_bias.unsqueeze(1))
        else:
            gate = 1.0
        return h + self.config.macro_update_scale * scale.unsqueeze(1) * gate * torch.tanh(update)

    def _forward_decomposed(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, MacroTrace]:
        sequences = [self._decompose_one(int(d)) for d in depths.detach().cpu().tolist()]
        max_passes = max(len(seq) for seq in sequences)
        out = h
        states: dict[int, torch.Tensor] = {}
        cumulative = torch.zeros(depths.shape[0], dtype=torch.long, device=h.device)
        physical = torch.tensor([len(seq) for seq in sequences], dtype=torch.float32, device=h.device)
        stride_table = torch.full((depths.shape[0], max_passes), -1, dtype=torch.long, device=h.device)
        for row, seq in enumerate(sequences):
            stride_table[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=h.device)
        for pass_idx in range(max_passes):
            stride = stride_table[:, pass_idx]
            active = stride > 0
            safe_stride = torch.where(active, stride, depths)
            stride_ids = self.stride_ids_for_depths(safe_stride)
            updated = self.apply_vectorized(out, h0, recipe_ids, stride_ids)
            out = torch.where(active.view(-1, 1, 1), updated, out)
            cumulative = cumulative + torch.where(active, stride, torch.zeros_like(stride))
            for boundary in self.strides:
                hit = bool((cumulative == boundary).any().item())
                if hit and boundary not in states:
                    states[boundary] = out.clone()
        direct_ids = self.stride_ids_for_depths(depths)
        direct = self.apply_vectorized(h, h0, recipe_ids, direct_ids).detach()
        decomposition_error = (out.detach() - direct).square().mean().sqrt()
        trace = MacroTrace(
            states=states,
            stride_sequence=sequences,
            physical_passes=physical,
            decomposition_error=decomposition_error,
        )
        return out, trace

    def forward(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        depths: torch.Tensor,
        *,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, MacroTrace]:
        del return_states
        if self.config.macro_decomposition != "direct":
            return self._forward_decomposed(h, h0, recipe_ids, depths)
        stride_ids = self.stride_ids_for_depths(depths)
        out = self.apply_vectorized(h, h0, recipe_ids, stride_ids)
        trace = MacroTrace(
            states={},
            stride_sequence=[[int(d)] for d in depths.detach().cpu().tolist()],
            physical_passes=torch.ones(depths.shape[0], dtype=torch.float32, device=h.device),
            decomposition_error=h.new_zeros(()),
        )
        return out, trace

    def consistency_loss(self, h: torch.Tensor, h0: torch.Tensor, recipe_ids: torch.Tensor) -> torch.Tensor:
        if self.cons_src_idx.numel() == 0:
            return h.new_zeros(())
        pair_count = self.cons_src_idx.numel()
        batch = h.shape[0]
        h_rep = h.unsqueeze(0).expand(pair_count, *h.shape).reshape(pair_count * batch, *h.shape[1:])
        h0_rep = h0.unsqueeze(0).expand(pair_count, *h0.shape).reshape(pair_count * batch, *h0.shape[1:])
        recipe_rep = recipe_ids.unsqueeze(0).expand(pair_count, batch).reshape(pair_count * batch)
        src = self.cons_src_idx.to(h.device).unsqueeze(1).expand(pair_count, batch).reshape(-1)
        dst = self.cons_dst_idx.to(h.device).unsqueeze(1).expand(pair_count, batch).reshape(-1)
        direct = self.apply_vectorized(h_rep, h0_rep, recipe_rep, dst)
        twice = self.apply_vectorized(
            self.apply_vectorized(h_rep, h0_rep, recipe_rep, src),
            h0_rep,
            recipe_rep,
            src,
        )
        return F.mse_loss(direct, twice)

    def compile_phi(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        depths: torch.Tensor,
        *,
        vocab_weight: torch.Tensor | None = None,
    ) -> MacroCompilerOutput:
        hidden, trace = self.forward(h, h0, recipe_ids, depths, return_states=True)
        logit_delta = None
        if vocab_weight is not None:
            logit_delta = (hidden - h) @ vocab_weight.t()
        uncertainty = torch.zeros(hidden.shape[0], device=hidden.device, dtype=hidden.dtype)
        if trace.decomposition_error is not None:
            uncertainty = uncertainty + trace.decomposition_error.to(hidden.dtype)
        confidence = torch.exp(-uncertainty).clamp(0.0, 1.0)
        sketch = hidden.mean(dim=1)
        return MacroCompilerOutput(
            predicted_hidden=hidden,
            predicted_logit_delta=logit_delta,
            predicted_trace=sketch,
            confidence=confidence,
        )

    def compile_trace(self, h: torch.Tensor, h0: torch.Tensor, recipe_ids: torch.Tensor) -> MacroCompilerOutput:
        depths = self.depth_values.to(h.device)[-1].expand(h.shape[0])
        return self.compile_phi(h, h0, recipe_ids, depths)

    def compile_adjoint(self, h: torch.Tensor, h0: torch.Tensor, recipe_ids: torch.Tensor) -> MacroCompilerOutput:
        return self.compile_trace(h, h0, recipe_ids)


def macro_distill_loss(
    hot_hidden: torch.Tensor,
    exact_hidden: torch.Tensor,
    hot_logits: torch.Tensor | None,
    exact_logits: torch.Tensor | None,
    *,
    lambda_hid: float,
    lambda_cos: float,
    lambda_kl: float,
    lambda_norm: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    exact_hidden = exact_hidden.detach()
    if exact_logits is not None:
        exact_logits = exact_logits.detach()
    hid = F.mse_loss(hot_hidden, exact_hidden)
    cos = 1.0 - F.cosine_similarity(hot_hidden.flatten(1), exact_hidden.flatten(1), dim=-1).mean()
    if hot_logits is not None and exact_logits is not None and hot_logits.shape == exact_logits.shape:
        temp = max(float(temperature), 1e-6)
        exact_logp = F.log_softmax(exact_logits / temp, dim=-1)
        hot_logp = F.log_softmax(hot_logits / temp, dim=-1)
        kl = F.kl_div(hot_logp, exact_logp.exp(), reduction="batchmean") * (temp**2)
    else:
        kl = hot_hidden.new_zeros(())
    hot_rms = hot_hidden.float().pow(2).mean(dim=-1).sqrt()
    exact_rms = exact_hidden.float().pow(2).mean(dim=-1).sqrt()
    norm = (hot_rms - exact_rms).abs().mean().to(hot_hidden.dtype)
    return {
        "hid": lambda_hid * hid,
        "cos": lambda_cos * cos,
        "kl": lambda_kl * kl,
        "norm": lambda_norm * norm,
    }
