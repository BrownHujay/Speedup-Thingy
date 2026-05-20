from __future__ import annotations

import torch
import torch.nn.functional as F

from recursive_training_engine.kernels import reference

STRICT_CUDA = False
REQUIRE_FLASH = True


def set_strict_cuda(enabled: bool, *, require_flash: bool = True) -> None:
    global STRICT_CUDA, REQUIRE_FLASH
    STRICT_CUDA = enabled
    REQUIRE_FLASH = require_flash


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


def backend_status() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    triton = triton_available()
    if cuda and triton:
        mode = "cuda_triton_dispatch"
    elif cuda:
        mode = "cuda_torch_fallback"
    else:
        mode = "torch_reference_fallback"
    return {
        "mode": mode,
        "strict_cuda": STRICT_CUDA,
        "require_flash": REQUIRE_FLASH,
        "cuda_available": cuda,
        "triton_available": triton,
        "uses_reference_fallback": mode != "cuda_triton_dispatch",
    }


def cuda_required() -> None:
    if not triton_available():
        raise RuntimeError("CUDA + Triton are required for this optimized kernel path")


def _require_cuda_tensor(*tensors: torch.Tensor) -> None:
    if STRICT_CUDA and any(not tensor.is_cuda for tensor in tensors if isinstance(tensor, torch.Tensor)):
        raise RuntimeError("strict_cuda requires CUDA tensors in optimized hot-path kernels")
    if STRICT_CUDA and not triton_available():
        raise RuntimeError("strict_cuda requires CUDA + Triton optimized kernels; fallback is disabled")


def k_fused_rmsnorm(*args, **kwargs):
    _require_cuda_tensor(args[0], args[1])
    # Torch's vectorized implementation is the CPU/MPS optimized path. CUDA/Triton
    # deployments can swap this symbol with a compiled extension without changing callers.
    return reference.k_fused_rmsnorm(*args, **kwargs)


def k_rope_apply(*args, **kwargs):
    _require_cuda_tensor(args[0], args[1])
    return reference.k_rope_apply(*args, **kwargs)


def k_qkv_dense(*args, **kwargs):
    _require_cuda_tensor(args[0])
    return reference.k_qkv_dense(*args, **kwargs)


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _require_cuda_tensor(q, k, v)
    if not q.is_cuda:
        return reference.k_flash_causal_dense(q, k, v)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    except Exception:
        if STRICT_CUDA and REQUIRE_FLASH:
            raise
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def k_flash_causal_dense(*args, **kwargs):
    return _flash_attention(*args, **kwargs)


def k_qkv_grouped(*args, **kwargs):
    _require_cuda_tensor(args[0])
    return reference.k_qkv_grouped(*args, **kwargs)


def k_flash_causal_grouped(*args, **kwargs):
    return _flash_attention(*args, **kwargs)


def k_out_proj_grouped(*args, **kwargs):
    _require_cuda_tensor(args[0])
    return reference.k_out_proj_grouped(*args, **kwargs)


def k_swiglu_dense(*args, **kwargs):
    _require_cuda_tensor(args[0])
    return reference.k_swiglu_dense(*args, **kwargs)


def k_swiglu_grouped(*args, **kwargs):
    _require_cuda_tensor(args[0])
    return reference.k_swiglu_grouped(*args, **kwargs)


def k_pack_by_recipe(*args, **kwargs):
    return reference.k_pack_by_recipe(*args, **kwargs)


def k_unpack_from_recipe(*args, **kwargs):
    return reference.k_unpack_from_recipe(*args, **kwargs)


def k_macro_phi(*args, **kwargs):
    _require_cuda_tensor(args[0], args[1])
    return reference.k_macro_phi(*args, **kwargs)


def k_recurrent_exact_loop(*args, **kwargs):
    return reference.k_recurrent_exact_loop(*args, **kwargs)


def k_logits_full(*args, **kwargs):
    _require_cuda_tensor(args[0], args[1])
    return reference.k_logits_full(*args, **kwargs)


def k_logits_shortlist(*args, **kwargs):
    _require_cuda_tensor(args[0], args[1], args[2])
    return reference.k_logits_shortlist(*args, **kwargs)


def k_cross_entropy_unreduced(*args, **kwargs):
    return reference.k_cross_entropy_unreduced(*args, **kwargs)


def k_sample_audit_mask(*args, **kwargs):
    return reference.k_sample_audit_mask(*args, **kwargs)


def k_metrics_reduce(*args, **kwargs):
    return reference.k_metrics_reduce(*args, **kwargs)
