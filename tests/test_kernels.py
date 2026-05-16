from __future__ import annotations

import pytest
import torch

from recursive_training_engine.kernels import optimized, reference
from recursive_training_engine.kernels.cluster_pool_ffn import (
    build_static_pack_gather_indices,
    cluster_pool_ffn_forward_from_assignments,
    cluster_pool_ffn_forward_preindexed,
    prepare_cluster_pool_weights,
    scatter_cluster_pool_grads,
)
from recursive_training_engine.layers import SVDFactorSparseFFN


def test_rmsnorm_reference_matches_manual() -> None:
    x = torch.randn(2, 3, 5)
    weight = torch.randn(5)
    out, _ = reference.k_fused_rmsnorm(x, weight)
    manual = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * weight
    assert torch.allclose(out, manual, atol=1e-6)


def test_optimized_rmsnorm_matches_reference() -> None:
    x = torch.randn(2, 3, 8)
    weight = torch.randn(8)
    ref, _ = reference.k_fused_rmsnorm(x, weight)
    opt, _ = optimized.k_fused_rmsnorm(x, weight)
    assert torch.allclose(ref, opt, atol=1e-6)


def test_pack_unpack_roundtrip() -> None:
    hidden = torch.randn(4, 2, 3)
    recipes = torch.tensor([2, 1, 2, 1])
    depths = torch.tensor([0, 0, 1, 1])
    packed = reference.k_pack_by_recipe(hidden, recipes, depths)
    restored = reference.k_unpack_from_recipe(packed.packed, packed.inverse_permutation)
    assert torch.allclose(hidden, restored)


def test_shortlist_logits_shape() -> None:
    hidden = torch.randn(2, 3, 4)
    weight = torch.randn(10, 4)
    shortlist = torch.randint(0, 10, (2, 3, 5))
    logits = reference.k_logits_shortlist(hidden, weight, shortlist)
    assert logits.shape == (2, 3, 5)


def test_triton_optimized_path_marked_cuda_required() -> None:
    if optimized.triton_available():
        pytest.skip("CUDA Triton is available in this environment")
    with pytest.raises(RuntimeError):
        optimized.cuda_required()


def test_strict_optimized_kernel_does_not_silently_fallback_on_cpu() -> None:
    optimized.set_strict_cuda(True)
    try:
        with pytest.raises(RuntimeError, match="strict_cuda"):
            optimized.k_fused_rmsnorm(torch.randn(2, 3, 4), torch.ones(4))
    finally:
        optimized.set_strict_cuda(False)


def test_triton_svd_sparse_ffn_mode_requires_cuda() -> None:
    if optimized.triton_available():
        pytest.skip("CUDA Triton is available in this environment")
    sparse = SVDFactorSparseFFN(
        8,
        16,
        rank=4,
        top_k=4,
        up_m=4,
        product_m=4,
        candidate_mode="triton",
    )
    with pytest.raises(RuntimeError, match="CUDA|Triton"):
        sparse(torch.randn(2, 8))


def test_cluster_pool_ffn_recovers_dense_when_pool_is_full_on_cpu() -> None:
    tokens = 6
    d_model = 8
    d_ff = 12
    x = torch.randn(tokens, d_model)
    w_up = torch.randn(d_ff, d_model)
    w_gate = torch.randn(d_ff, d_model)
    w_down = torch.randn(d_ff, d_model)
    candidate_ids = torch.arange(d_ff).repeat(2, 1)
    assignments = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    wup_pool, wgate_pool, wdown_pool = prepare_cluster_pool_weights(
        w_up,
        w_gate,
        w_down,
        candidate_ids,
    )
    dense = ((x @ w_up.t()) * torch.nn.functional.silu(x @ w_gate.t())) @ w_down
    clustered = cluster_pool_ffn_forward_from_assignments(
        x,
        assignments,
        wup_pool,
        wgate_pool,
        wdown_pool,
        max_tokens_per_cluster=3,
    )
    assert torch.allclose(clustered, dense, atol=1e-5)


def test_preindexed_cluster_pool_backward_matches_dense_full_pool() -> None:
    tokens = 5
    d_model = 4
    d_ff = 7
    x_dense = torch.randn(tokens, d_model, requires_grad=True)
    w_up = torch.randn(d_ff, d_model, requires_grad=True)
    w_gate = torch.randn(d_ff, d_model, requires_grad=True)
    w_down = torch.randn(d_ff, d_model, requires_grad=True)
    x_sparse = x_dense.detach().clone().requires_grad_()
    candidate_ids = torch.arange(d_ff).view(1, d_ff)
    assignments = torch.zeros(tokens, dtype=torch.long)
    pack_index, flat_gather, _, _ = build_static_pack_gather_indices(assignments, cluster_count=1)
    wup_pool = w_up.detach().t().unsqueeze(0).contiguous().requires_grad_()
    wgate_pool = w_gate.detach().t().unsqueeze(0).contiguous().requires_grad_()
    wdown_pool = w_down.detach().unsqueeze(0).contiguous().requires_grad_()
    dense = ((x_dense @ w_up.t()) * torch.nn.functional.silu(x_dense @ w_gate.t())) @ w_down
    sparse = cluster_pool_ffn_forward_preindexed(
        x_sparse,
        pack_index,
        flat_gather,
        wup_pool,
        wgate_pool,
        wdown_pool,
    )
    assert torch.allclose(sparse, dense, atol=1e-6)
    dense.square().mean().backward()
    sparse.square().mean().backward()
    assert torch.allclose(x_sparse.grad, x_dense.grad, atol=1e-6)
    assert wup_pool.grad is not None and wgate_pool.grad is not None and wdown_pool.grad is not None
    grad_up, grad_gate, grad_down = scatter_cluster_pool_grads(
        candidate_ids,
        wup_pool.grad,
        wgate_pool.grad,
        wdown_pool.grad,
        d_ff=d_ff,
    )
    assert torch.allclose(grad_up, w_up.grad, atol=1e-6)
    assert torch.allclose(grad_gate, w_gate.grad, atol=1e-6)
    assert torch.allclose(grad_down, w_down.grad, atol=1e-5)
