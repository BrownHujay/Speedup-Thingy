from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig


def _rms(x: torch.Tensor, *, dim: int | tuple[int, ...] = -1, keepdim: bool = False) -> torch.Tensor:
    return x.float().pow(2).mean(dim=dim, keepdim=keepdim).sqrt().to(x.dtype)


def _rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)
    return x * scale


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(1e-8)
    return value + torch.log(-torch.expm1(-value))


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
        if config.macro_type == "v2_delta_radius":
            input_dim = 3 * d
            if config.macro_use_delta_to_h0 or config.macro_include_delta_to_h0:
                input_dim += d
            if config.macro_use_depth_embedding:
                input_dim += d
                self.depth_embed = nn.Parameter(torch.randn(k, d) * (1.0 / max(d, 1) ** 0.5))
            else:
                self.register_parameter("depth_embed", None)
            if config.macro_use_recipe_embedding:
                input_dim += d
                self.recipe_embed = nn.Parameter(torch.randn(r, d) * (1.0 / max(d, 1) ** 0.5))
            else:
                self.register_parameter("recipe_embed", None)
            hidden = max(m, m * config.macro_hidden_mult)
            self.v = nn.Parameter(torch.randn(r, k, input_dim, hidden) * (1.0 / input_dim**0.5))
            self.v2 = nn.Parameter(torch.randn(r, k, hidden, m) * (1.0 / hidden**0.5))
            self.u = nn.Parameter(torch.randn(r, k, m, d) * 1e-3)
            self.bias = nn.Parameter(torch.zeros(r, k, d))
            radius = max(float(config.macro_update_scale_init), float(config.macro_update_scale), 1e-3)
            rho = _inverse_softplus(torch.full((r, k), radius))
            self.rho_base_raw = nn.Parameter(rho)
            self.rho_head = nn.Parameter(torch.zeros(r, k, d, 1))
            self.register_buffer("teacher_delta_rms", torch.zeros(r, k), persistent=True)
            self.register_parameter("d_gain", None)
            self.register_parameter("update_scale", None)
            self.register_parameter("gate_v", None)
            self.register_parameter("gate_u", None)
            self.register_parameter("gate_bias", None)
        else:
            input_dim = 3 * d
            if config.macro_include_delta_to_h0:
                input_dim += d
            if config.macro_use_depth_embedding:
                input_dim += d
                self.depth_embed = nn.Parameter(torch.randn(k, d) * (1.0 / max(d, 1) ** 0.5))
            else:
                self.register_parameter("depth_embed", None)
            self.register_parameter("recipe_embed", None)
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
            self.register_parameter("rho_base_raw", None)
            self.register_parameter("rho_head", None)
            self.register_buffer("teacher_delta_rms", torch.zeros(r, k), persistent=True)
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

    def initialize_radius_from_teacher_delta(
        self,
        recipe_id: int | torch.Tensor,
        depth: int | torch.Tensor,
        delta_rms: float | torch.Tensor,
    ) -> None:
        if self.config.macro_type != "v2_delta_radius":
            return
        recipe = int(recipe_id.detach().cpu().item()) if isinstance(recipe_id, torch.Tensor) else int(recipe_id)
        depth_value = int(depth.detach().cpu().item()) if isinstance(depth, torch.Tensor) else int(depth)
        idx = self.stride_to_idx[depth_value]
        value = torch.as_tensor(delta_rms, dtype=self.rho_base_raw.dtype, device=self.rho_base_raw.device)
        value = value.detach().clamp_min(1e-6)
        with torch.no_grad():
            self.rho_base_raw[recipe, idx].copy_(_inverse_softplus(value))
            self.teacher_delta_rms[recipe, idx].copy_(value)

    def current_radius(self, recipe_ids: torch.Tensor, stride_ids: torch.Tensor) -> torch.Tensor:
        if self.config.macro_type != "v2_delta_radius":
            return recipe_ids.new_zeros(recipe_ids.shape, dtype=torch.float32).to(recipe_ids.device)
        return F.softplus(self.rho_base_raw[recipe_ids, stride_ids])

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
        if self.config.macro_type == "v2_delta_radius":
            return self._apply_vectorized_v2(h, h0, recipe_ids, stride_ids)
        return self._apply_vectorized_bounded(h, h0, recipe_ids, stride_ids)

    def _apply_vectorized_bounded(
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
        normed = _rmsnorm(h)
        normed_h0 = _rmsnorm(h0)
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

    def _apply_vectorized_v2(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        stride_ids: torch.Tensor,
    ) -> torch.Tensor:
        del h
        source = h0
        normed = _rmsnorm(source)
        normed_h0 = _rmsnorm(h0)
        pooled = normed.mean(dim=1)
        pool_broadcast = pooled.unsqueeze(1).expand_as(normed)
        parts = [normed, pool_broadcast, h0]
        if self.config.macro_use_delta_to_h0 or self.config.macro_include_delta_to_h0:
            parts.append(normed - normed_h0)
        if self.depth_embed is not None:
            parts.append(self.depth_embed[stride_ids].unsqueeze(1).expand_as(normed))
        if self.recipe_embed is not None:
            parts.append(self.recipe_embed[recipe_ids].unsqueeze(1).expand_as(normed))
        z = torch.cat(parts, dim=-1)
        v = self.v[recipe_ids, stride_ids]
        v2 = self.v2[recipe_ids, stride_ids]
        u = self.u[recipe_ids, stride_ids]
        raw_dir = torch.bmm(F.silu(torch.bmm(F.silu(torch.bmm(z, v)), v2)), u)
        raw_dir = raw_dir + self.bias[recipe_ids, stride_ids].unsqueeze(1)
        dir_rms = _rms(raw_dir, dim=(1, 2), keepdim=True).clamp_min(1e-6)
        direction = raw_dir / dir_rms
        rho_head = torch.bmm(
            pooled.unsqueeze(1),
            self.rho_head[recipe_ids, stride_ids],
        ).squeeze(-1).squeeze(-1)
        rho = F.softplus(self.rho_base_raw[recipe_ids, stride_ids] + rho_head)
        teacher_radius = self.teacher_delta_rms[recipe_ids, stride_ids].to(rho.device)
        has_teacher_radius = teacher_radius > 0
        if self.config.macro_radius_init_from_teacher and bool(has_teacher_radius.any().item()):
            lo = teacher_radius * self.config.macro_radius_clamp_mult_min
            hi = teacher_radius * self.config.macro_radius_clamp_mult_max
            clamped = torch.clamp(rho, min=lo, max=hi)
            rho = torch.where(has_teacher_radius, clamped, rho)
        delta = rho.view(-1, 1, 1).to(direction.dtype) * direction
        return h0 + delta

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
    h0: torch.Tensor | None = None,
    lambda_hid: float = 0.0,
    lambda_cos: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_norm: float = 0.0,
    lambda_delta_dir: float = 0.0,
    lambda_delta_rms: float = 0.0,
    lambda_endpoint_normed: float = 0.0,
    lambda_endpoint_raw: float = 0.0,
    lambda_macro_rms_trust: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    exact_hidden = exact_hidden.detach()
    if exact_logits is not None:
        exact_logits = exact_logits.detach()
    zero = hot_hidden.new_zeros(())
    use_delta_objective = h0 is not None
    if use_delta_objective:
        h0_detached = h0.detach()
        delta_pred = hot_hidden - h0_detached
        delta_exact = exact_hidden - h0_detached
        delta_dir = 1.0 - F.cosine_similarity(
            delta_pred.flatten(1),
            delta_exact.flatten(1),
            dim=-1,
            eps=1e-8,
        ).mean()
        pred_rms = _rms(delta_pred, dim=(1, 2)).clamp_min(1e-8)
        exact_rms = _rms(delta_exact, dim=(1, 2)).clamp_min(1e-8)
        delta_rms = (torch.log(pred_rms) - torch.log(exact_rms)).square().mean()
        endpoint_normed = F.smooth_l1_loss(_rmsnorm(hot_hidden), _rmsnorm(exact_hidden))
        endpoint_raw = F.smooth_l1_loss(hot_hidden, exact_hidden)
        rms_trust = macro_rms_trust_penalty(hot_hidden, exact_hidden)
        hid = zero
        cos = zero
        norm = zero
    else:
        hid = F.mse_loss(hot_hidden, exact_hidden)
        cos = 1.0 - F.cosine_similarity(
            hot_hidden.flatten(1),
            exact_hidden.flatten(1),
            dim=-1,
            eps=1e-8,
        ).mean()
        hot_rms = hot_hidden.float().pow(2).mean(dim=-1).sqrt()
        exact_rms = exact_hidden.float().pow(2).mean(dim=-1).sqrt()
        norm = (hot_rms - exact_rms).abs().mean().to(hot_hidden.dtype)
        delta_dir = zero
        delta_rms = zero
        endpoint_normed = zero
        endpoint_raw = zero
        rms_trust = zero
    if hot_logits is not None and exact_logits is not None and hot_logits.shape == exact_logits.shape:
        temp = max(float(temperature), 1e-6)
        exact_logp = F.log_softmax(exact_logits / temp, dim=-1)
        hot_logp = F.log_softmax(hot_logits / temp, dim=-1)
        kl = F.kl_div(hot_logp, exact_logp.exp(), reduction="batchmean") * (temp**2)
    else:
        kl = zero
    return {
        "hid": lambda_hid * hid,
        "cos": lambda_cos * cos,
        "kl": lambda_kl * kl,
        "norm": lambda_norm * norm,
        "delta_dir": lambda_delta_dir * delta_dir,
        "delta_rms": lambda_delta_rms * delta_rms,
        "endpoint_normed": lambda_endpoint_normed * endpoint_normed,
        "endpoint_raw": lambda_endpoint_raw * endpoint_raw,
        "rms_trust": lambda_macro_rms_trust * rms_trust,
    }


def macro_rms_trust_penalty(hot_hidden: torch.Tensor, exact_hidden: torch.Tensor) -> torch.Tensor:
    ratio = _rms(hot_hidden, dim=(1, 2)).clamp_min(1e-8) / _rms(
        exact_hidden.detach(),
        dim=(1, 2),
    ).clamp_min(1e-8)
    return torch.log(ratio).square().mean().to(hot_hidden.dtype)


def apply_macro_rms_clamp(
    hot_hidden: torch.Tensor,
    exact_hidden: torch.Tensor,
    *,
    enabled: bool,
    min_scale: float,
    max_scale: float,
) -> torch.Tensor:
    if not enabled:
        return hot_hidden
    scale = _rms(exact_hidden.detach(), dim=(1, 2), keepdim=True).clamp_min(1e-8) / _rms(
        hot_hidden,
        dim=(1, 2),
        keepdim=True,
    ).clamp_min(1e-8)
    scale = torch.clamp(scale, min=float(min_scale), max=float(max_scale)).to(hot_hidden.dtype)
    return hot_hidden * scale


def macro_alignment_metrics(
    hot_hidden: torch.Tensor,
    exact_hidden: torch.Tensor,
    *,
    h0: torch.Tensor | None,
    hot_logits: torch.Tensor | None = None,
    exact_logits: torch.Tensor | None = None,
    hot_nll: torch.Tensor | None = None,
    exact_nll: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    exact_hidden = exact_hidden.detach()
    metrics: dict[str, torch.Tensor] = {}
    metrics["hidden_cosine_exact_macro"] = F.cosine_similarity(
        hot_hidden.flatten(1),
        exact_hidden.flatten(1),
        dim=-1,
        eps=1e-8,
    ).mean()
    metrics["hidden_mse_exact_macro"] = (hot_hidden - exact_hidden).square().mean()
    macro_norm = _rms(hot_hidden, dim=(1, 2)).mean()
    exact_norm = _rms(exact_hidden, dim=(1, 2)).mean()
    metrics["macro_norm"] = macro_norm
    metrics["exact_norm"] = exact_norm
    metrics["macro_exact_norm_ratio"] = macro_norm / exact_norm.clamp_min(1e-8)
    if h0 is not None:
        h0_detached = h0.detach()
        delta_pred = hot_hidden - h0_detached
        delta_exact = exact_hidden - h0_detached
        metrics["delta_cosine_exact_macro"] = F.cosine_similarity(
            delta_pred.flatten(1),
            delta_exact.flatten(1),
            dim=-1,
            eps=1e-8,
        ).mean()
        pred_rms = _rms(delta_pred, dim=(1, 2)).mean()
        exact_rms = _rms(delta_exact, dim=(1, 2)).mean()
        metrics["delta_rms_exact"] = exact_rms
        metrics["delta_rms_pred"] = pred_rms
        metrics["delta_rms_ratio"] = pred_rms / exact_rms.clamp_min(1e-8)
    if hot_logits is not None and exact_logits is not None and hot_logits.shape == exact_logits.shape:
        temp_exact = F.log_softmax(exact_logits.detach(), dim=-1)
        temp_hot = F.log_softmax(hot_logits, dim=-1)
        metrics["logit_kl_exact_macro"] = (
            temp_exact.exp() * (temp_exact - temp_hot)
        ).sum(dim=-1).mean()
    if hot_nll is not None and exact_nll is not None:
        metrics["hot_exact_nll_gap"] = hot_nll - exact_nll
    return metrics
