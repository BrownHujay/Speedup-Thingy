from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def k_fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    y = x if residual is None else x + residual
    normed = y * torch.rsqrt(y.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(y.dtype)
    return normed * weight, y if residual is not None else None


def k_rope_apply(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


def k_qkv_dense(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x @ wq, x @ wk, x @ wv


def k_flash_causal_dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def k_qkv_grouped(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    column_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return x @ wq[:, column_indices], x @ wk[:, column_indices], x @ wv[:, column_indices]


def k_flash_causal_grouped(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def k_out_proj_grouped(x: torch.Tensor, wo: torch.Tensor, row_indices: torch.Tensor) -> torch.Tensor:
    return x @ wo[row_indices, :]


def k_swiglu_dense(
    x: torch.Tensor,
    wu: torch.Tensor,
    wg: torch.Tensor,
    wd: torch.Tensor,
) -> torch.Tensor:
    return (x @ wu) * F.silu(x @ wg) @ wd


def k_swiglu_grouped(
    x: torch.Tensor,
    wu: torch.Tensor,
    wg: torch.Tensor,
    wd: torch.Tensor,
    slab_indices: torch.Tensor,
) -> torch.Tensor:
    return (x @ wu[:, slab_indices]) * F.silu(x @ wg[:, slab_indices]) @ wd[slab_indices, :]


@dataclass(slots=True)
class PackedByRecipe:
    packed: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    keys: torch.Tensor
    offsets: torch.Tensor
    unique_keys: torch.Tensor


def k_pack_by_recipe(
    hidden: torch.Tensor,
    recipe_ids: torch.Tensor,
    depth_ids: torch.Tensor,
    audit_flags: torch.Tensor | None = None,
) -> PackedByRecipe:
    audit = torch.zeros_like(recipe_ids) if audit_flags is None else audit_flags.long()
    keys = recipe_ids.long() * 1_000_000 + depth_ids.long() * 10 + audit
    sorted_keys, perm = torch.sort(keys, stable=True)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    unique, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return PackedByRecipe(hidden[perm], perm, inverse, sorted_keys, offsets, unique)


def k_unpack_from_recipe(packed: torch.Tensor, inverse_permutation: torch.Tensor) -> torch.Tensor:
    return packed[inverse_permutation]


def k_macro_phi(
    h: torch.Tensor,
    h0: torch.Tensor,
    d_gain: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    normed = h * torch.rsqrt(h.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(h.dtype)
    pooled = normed.mean(dim=1, keepdim=True).expand_as(normed)
    z = torch.cat([normed, pooled, h0], dim=-1)
    low = F.silu(z @ v) @ u
    return h + d_gain * h + low + bias


def k_recurrent_exact_loop(step_fn, h: torch.Tensor, h0: torch.Tensor, depth: int, boundaries: set[int]) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    states: dict[int, torch.Tensor] = {}
    for t in range(depth):
        h = step_fn(h, h0)
        boundary = t + 1
        if boundary in boundaries:
            states[boundary] = h
    return h, states


def k_logits_full(hidden: torch.Tensor, vocab_weight: torch.Tensor) -> torch.Tensor:
    return hidden @ vocab_weight.t()


def k_logits_shortlist(hidden: torch.Tensor, vocab_weight: torch.Tensor, shortlist: torch.Tensor) -> torch.Tensor:
    rows = vocab_weight[shortlist]
    return (hidden.unsqueeze(-2) * rows).sum(dim=-1)


def k_cross_entropy_unreduced(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none").view_as(targets)


def k_sample_audit_mask(p: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(seed)
    return torch.bernoulli(p, generator=gen).bool()


def k_metrics_reduce(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().float().mean().cpu()) for k, v in values.items()}
