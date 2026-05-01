from __future__ import annotations

import pytest
import torch

from recursive_training_engine.kernels import optimized, reference


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
