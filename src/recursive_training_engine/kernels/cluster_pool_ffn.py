from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:  # pragma: no cover - availability depends on CUDA machine.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _require_available() -> None:
    if not available():
        raise RuntimeError("Cluster-pool FFN kernels require NVIDIA CUDA + Triton")


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _assign_slots_kernel(
        assignments,
        slot_counters,
        flat_positions,
        overflow,
        N_TOKENS: tl.constexpr,
        MAX_COUNT: tl.constexpr,
    ):
        token = tl.program_id(0)
        cluster = tl.load(assignments + token)
        slot = tl.atomic_add(slot_counters + cluster, 1, sem="relaxed")
        valid = slot < MAX_COUNT
        tl.store(flat_positions + token, tl.where(valid, cluster * MAX_COUNT + slot, -1))
        tl.store(overflow, 1, mask=~valid)

    @triton.jit
    def _pack_x_kernel(
        x,
        flat_positions,
        x_pad,
        N_TOKENS: tl.constexpr,
        D_MODEL: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_block = tl.program_id(1)
        offsets_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offsets_d < D_MODEL
        flat_pos = tl.load(flat_positions + token)
        values = tl.load(x + token * D_MODEL + offsets_d, mask=valid_d, other=0.0)
        tl.store(
            x_pad + flat_pos * D_MODEL + offsets_d,
            values,
            mask=(flat_pos >= 0) & valid_d,
        )

    @triton.jit
    def _gather_y_kernel(
        y_pad,
        flat_positions,
        out,
        N_TOKENS: tl.constexpr,
        D_MODEL: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_block = tl.program_id(1)
        offsets_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offsets_d < D_MODEL
        flat_pos = tl.load(flat_positions + token)
        values = tl.load(
            y_pad + flat_pos * D_MODEL + offsets_d,
            mask=(flat_pos >= 0) & valid_d,
            other=0.0,
        )
        tl.store(out + token * D_MODEL + offsets_d, values, mask=valid_d)


def prepare_cluster_pool_weights(
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather cluster candidate weights once for GEMM-shaped execution.

    Returned shapes are:
    - `wup_pool`: `[clusters, d_model, candidate_m]`
    - `wgate_pool`: `[clusters, d_model, candidate_m]`
    - `wdown_pool`: `[clusters, candidate_m, d_model]`
    """

    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [clusters, candidate_m]")
    candidate_ids = candidate_ids.to(device=w_up.device, dtype=torch.long)
    wup_pool = w_up.index_select(0, candidate_ids.reshape(-1)).view(
        candidate_ids.shape[0],
        candidate_ids.shape[1],
        w_up.shape[1],
    )
    wgate_pool = w_gate.index_select(0, candidate_ids.reshape(-1)).view(
        candidate_ids.shape[0],
        candidate_ids.shape[1],
        w_gate.shape[1],
    )
    wdown_pool = w_down.index_select(0, candidate_ids.reshape(-1)).view(
        candidate_ids.shape[0],
        candidate_ids.shape[1],
        w_down.shape[1],
    )
    return (
        wup_pool.transpose(1, 2).contiguous(),
        wgate_pool.transpose(1, 2).contiguous(),
        wdown_pool.contiguous(),
    )


def route_to_static_centers(
    x: torch.Tensor,
    up_a: torch.Tensor,
    gate_a: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    """Assign tokens to fixed cluster centers from SVD selector features."""

    q_up = x @ up_a
    q_gate = x @ gate_a
    features = F.normalize(torch.cat([q_up, q_gate], dim=-1).float(), dim=-1)
    return (features @ centers.float().t()).argmax(dim=-1)


def cluster_pool_ffn_reference(
    x: torch.Tensor,
    assignments: torch.Tensor,
    wup_pool: torch.Tensor,
    wgate_pool: torch.Tensor,
    wdown_pool: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation with a Python loop for CPU tests only."""

    out = x.new_zeros(x.shape[0], x.shape[1])
    for cluster_idx in range(wup_pool.shape[0]):
        token_idx = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten()
        if token_idx.numel() == 0:
            continue
        xc = x.index_select(0, token_idx)
        up = xc @ wup_pool[cluster_idx]
        gate = xc @ wgate_pool[cluster_idx]
        z = up * F.silu(gate)
        out.index_copy_(0, token_idx, z @ wdown_pool[cluster_idx])
    return out


def build_static_pack_gather_indices(
    assignments: torch.Tensor,
    *,
    cluster_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute padded pack/gather indices for a fixed assignment pattern."""

    if assignments.ndim != 1:
        raise ValueError("assignments must have shape [tokens]")
    assignments = assignments.to(dtype=torch.long)
    counts = torch.bincount(assignments, minlength=cluster_count)
    max_count = int(counts.max().item()) if counts.numel() else 0
    pack_index = torch.zeros(cluster_count, max_count, device=assignments.device, dtype=torch.long)
    flat_gather = torch.empty(assignments.shape[0], device=assignments.device, dtype=torch.long)
    for cluster_idx in range(cluster_count):
        token_idx = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten()
        n = token_idx.numel()
        if n:
            pack_index[cluster_idx, :n] = token_idx
            flat_gather[token_idx] = cluster_idx * max_count + torch.arange(
                n,
                device=assignments.device,
                dtype=torch.long,
            )
    return pack_index.contiguous(), flat_gather.contiguous(), counts, max_count


def cluster_pool_ffn_forward_preindexed(
    x: torch.Tensor,
    pack_index: torch.Tensor,
    flat_gather: torch.Tensor,
    wup_pool: torch.Tensor,
    wgate_pool: torch.Tensor,
    wdown_pool: torch.Tensor,
) -> torch.Tensor:
    """Differentiable packed cluster-pool FFN for training benchmarks.

    This uses only PyTorch tensor ops (`index_select` + `bmm` + `index_select`)
    and supports autograd. It is the forward/backward benchmark shape for static
    cluster pools; custom CUDA can later replace the pack/gather pieces.
    """

    if x.ndim != 2:
        raise ValueError("x must have shape [tokens, d_model]")
    cluster_count, max_count = pack_index.shape
    x_pad = x.index_select(0, pack_index.reshape(-1)).view(cluster_count, max_count, x.shape[-1])
    up = torch.bmm(x_pad, wup_pool)
    gate = torch.bmm(x_pad, wgate_pool)
    z = up * F.silu(gate)
    y_pad = torch.bmm(z, wdown_pool)
    return y_pad.reshape(cluster_count * max_count, x.shape[-1]).index_select(0, flat_gather)


def scatter_cluster_pool_grads(
    candidate_ids: torch.Tensor,
    wup_pool_grad: torch.Tensor,
    wgate_pool_grad: torch.Tensor,
    wdown_pool_grad: torch.Tensor,
    *,
    d_ff: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter prepacked pool gradients back into dense FFN row gradients."""

    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [clusters, candidate_m]")
    ids = candidate_ids.reshape(-1).to(dtype=torch.long, device=wup_pool_grad.device)
    d_model = wdown_pool_grad.shape[-1]
    grad_up = wup_pool_grad.transpose(1, 2).reshape(-1, d_model)
    grad_gate = wgate_pool_grad.transpose(1, 2).reshape(-1, d_model)
    grad_down = wdown_pool_grad.reshape(-1, d_model)
    dense_up = torch.zeros(d_ff, d_model, device=wup_pool_grad.device, dtype=wup_pool_grad.dtype)
    dense_gate = torch.zeros_like(dense_up)
    dense_down = torch.zeros_like(dense_up)
    dense_up.index_add_(0, ids, grad_up)
    dense_gate.index_add_(0, ids, grad_gate)
    dense_down.index_add_(0, ids, grad_down)
    return dense_up, dense_gate, dense_down


def cluster_pool_ffn_forward_from_assignments(
    x: torch.Tensor,
    assignments: torch.Tensor,
    wup_pool: torch.Tensor,
    wgate_pool: torch.Tensor,
    wdown_pool: torch.Tensor,
    *,
    max_tokens_per_cluster: int | None = None,
    block_d: int = 64,
    return_overflow: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute a cluster-shared sparse FFN with no Python per-cluster loop.

    CUDA path:
    1. Triton assigns each token a slot inside its cluster.
    2. Triton packs tokens into `[clusters, max_tokens_per_cluster, d_model]`.
    3. cuBLAS batched GEMMs run the pooled SwiGLU FFN.
    4. Triton gathers padded cluster outputs back to token order.

    CPU/non-CUDA path intentionally falls back to the loop reference for tests.
    """

    if x.ndim != 2:
        raise ValueError("x must have shape [tokens, d_model]")
    if assignments.ndim != 1 or assignments.shape[0] != x.shape[0]:
        raise ValueError("assignments must have shape [tokens]")
    if wup_pool.ndim != 3 or wgate_pool.ndim != 3 or wdown_pool.ndim != 3:
        raise ValueError("cluster pools must be rank-3 tensors")
    if wup_pool.shape[0] != wgate_pool.shape[0] or wup_pool.shape[0] != wdown_pool.shape[0]:
        raise ValueError("cluster pools must have the same cluster count")
    if x.device.type != "cuda":
        out = cluster_pool_ffn_reference(x, assignments, wup_pool, wgate_pool, wdown_pool)
        overflow = torch.zeros((), device=x.device, dtype=torch.int32)
        return (out, overflow) if return_overflow else out
    _require_available()

    tensors = (x, assignments, wup_pool, wgate_pool, wdown_pool)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("cluster-pool FFN CUDA path requires all tensors on CUDA")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("cluster-pool FFN CUDA path is inference/eval only; backward is not implemented")
    if any(not tensor.is_contiguous() for tensor in tensors):
        x = x.contiguous()
        assignments = assignments.contiguous()
        wup_pool = wup_pool.contiguous()
        wgate_pool = wgate_pool.contiguous()
        wdown_pool = wdown_pool.contiguous()

    n_tokens, d_model = x.shape
    cluster_count = wup_pool.shape[0]
    if max_tokens_per_cluster is None:
        # This is a conservative no-overflow default. Benchmarks should pass a
        # tighter capacity such as ceil(tokens / clusters * 1.25).
        max_tokens_per_cluster = n_tokens
    max_tokens_per_cluster = max(1, int(max_tokens_per_cluster))
    block_d = _next_power_of_2(block_d)

    slot_counters = torch.zeros(cluster_count, device=x.device, dtype=torch.int32)
    flat_positions = torch.empty(n_tokens, device=x.device, dtype=torch.int32)
    overflow = torch.zeros((), device=x.device, dtype=torch.int32)
    _assign_slots_kernel[(n_tokens,)](
        assignments,
        slot_counters,
        flat_positions,
        overflow,
        n_tokens,
        max_tokens_per_cluster,
        num_warps=4,
    )
    x_pad = torch.zeros(
        (cluster_count, max_tokens_per_cluster, d_model),
        device=x.device,
        dtype=x.dtype,
    )
    _pack_x_kernel[(n_tokens, triton.cdiv(d_model, block_d))](
        x,
        flat_positions,
        x_pad,
        n_tokens,
        d_model,
        block_d,
        num_warps=4,
    )
    up = torch.bmm(x_pad, wup_pool)
    gate = torch.bmm(x_pad, wgate_pool)
    z = up * F.silu(gate)
    y_pad = torch.bmm(z, wdown_pool)
    out = torch.empty_like(x)
    _gather_y_kernel[(n_tokens, triton.cdiv(d_model, block_d))](
        y_pad,
        flat_positions,
        out,
        n_tokens,
        d_model,
        block_d,
        num_warps=4,
    )
    return (out, overflow) if return_overflow else out


def cluster_pool_ffn_forward_static(
    x: torch.Tensor,
    up_a: torch.Tensor,
    gate_a: torch.Tensor,
    centers: torch.Tensor,
    wup_pool: torch.Tensor,
    wgate_pool: torch.Tensor,
    wdown_pool: torch.Tensor,
    *,
    max_tokens_per_cluster: int | None = None,
    block_d: int = 64,
    return_overflow: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to fixed centers, then execute the packed cluster FFN."""

    assignments = route_to_static_centers(x, up_a, gate_a, centers)
    return cluster_pool_ffn_forward_from_assignments(
        x,
        assignments,
        wup_pool,
        wgate_pool,
        wdown_pool,
        max_tokens_per_cluster=max_tokens_per_cluster,
        block_d=block_d,
        return_overflow=return_overflow,
    )


def balanced_synthetic_assignments(tokens: int, clusters: int, device: torch.device) -> torch.Tensor:
    """Deterministic balanced assignments for kernel benchmarks."""

    return (torch.arange(tokens, device=device, dtype=torch.long) % max(1, clusters)).contiguous()


def synthetic_cluster_centers(
    x: torch.Tensor,
    up_a: torch.Tensor,
    gate_a: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
) -> torch.Tensor:
    """Build normalized centers for a fixed assignment pattern."""

    with torch.no_grad():
        q_up = x @ up_a
        q_gate = x @ gate_a
        features = F.normalize(torch.cat([q_up, q_gate], dim=-1).float(), dim=-1)
        centers = features.new_zeros(clusters, features.shape[-1])
        counts = torch.bincount(assignments, minlength=clusters).float().clamp_min(1.0)
        centers.index_add_(0, assignments, features)
        centers = centers / counts.unsqueeze(-1)
        return F.normalize(centers, dim=-1).to(dtype=x.dtype)


def suggested_cluster_capacity(tokens: int, clusters: int, capacity_factor: float = 1.25) -> int:
    return max(1, int(math.ceil(tokens / max(1, clusters) * float(capacity_factor))))
