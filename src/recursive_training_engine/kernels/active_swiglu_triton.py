from __future__ import annotations

import torch

try:  # pragma: no cover - availability depends on CUDA environment.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _swiglu_forward_kernel(
        ug,
        z,
        TOTAL: tl.constexpr,
        M: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        row = offsets // M
        col = offsets - row * M
        base = row * (2 * M) + col
        up = tl.load(ug + base, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(ug + base + M, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        out = up * gate * sig
        tl.store(z + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        ug,
        dz,
        dug,
        TOTAL: tl.constexpr,
        M: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        row = offsets // M
        col = offsets - row * M
        base = row * (2 * M) + col
        up = tl.load(ug + base, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(ug + base + M, mask=mask, other=0.0).to(tl.float32)
        grad_z = tl.load(dz + offsets, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sig
        d_silu = sig * (1.0 + gate * (1.0 - sig))
        tl.store(dug + base, grad_z * silu, mask=mask)
        tl.store(dug + base + M, grad_z * up * d_silu, mask=mask)


class _TritonPackedSwiGLUFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, wug: torch.Tensor, wdown_rows: torch.Tensor) -> torch.Tensor:
        if not available():
            raise RuntimeError("Triton packed SwiGLU FFN requires CUDA + Triton")
        if x.ndim != 2 or wug.ndim != 2 or wdown_rows.ndim != 2:
            raise ValueError("expected x [N,D], wug [2M,D], wdown_rows [M,D]")
        if wug.shape[0] % 2 != 0:
            raise ValueError("wug first dimension must be 2M")
        m = wug.shape[0] // 2
        if wdown_rows.shape[0] != m:
            raise ValueError("wdown_rows first dimension must match M")
        x = x.contiguous()
        wug = wug.contiguous()
        wdown_rows = wdown_rows.contiguous()
        ug = x @ wug.t()
        z = torch.empty((x.shape[0], m), device=x.device, dtype=x.dtype)
        total = z.numel()
        block = 256
        grid = (triton.cdiv(total, block),)
        _swiglu_forward_kernel[grid](ug, z, total, m, BLOCK=block)
        y = z @ wdown_rows
        ctx.save_for_backward(x, wug, wdown_rows, ug, z)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, wug, wdown_rows, ug, z = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_for_wdown = grad_output.to(dtype=wdown_rows.dtype)
        dz = grad_for_wdown @ wdown_rows.t()
        d_wdown = z.to(dtype=wdown_rows.dtype).t() @ grad_for_wdown
        dug = torch.empty_like(ug)
        total = dz.numel()
        block = 256
        grid = (triton.cdiv(total, block),)
        _swiglu_backward_kernel[grid](ug, dz.to(dtype=ug.dtype), dug, total, wdown_rows.shape[0], BLOCK=block)
        dug_for_wug = dug.to(dtype=wug.dtype)
        d_x = dug_for_wug @ wug
        d_wug = dug_for_wug.t() @ x.to(dtype=wug.dtype)
        return d_x, d_wug, d_wdown


def triton_packed_swiglu_ffn(
    x: torch.Tensor,
    wug: torch.Tensor,
    wdown_rows: torch.Tensor,
) -> torch.Tensor:
    """Packed SwiGLU FFN with Triton-fused activation forward/backward.

    Shapes:
      x: [tokens, d_model]
      wug: [2 * active_m, d_model]
      wdown_rows: [active_m, d_model]
    """

    return _TritonPackedSwiGLUFFN.apply(x, wug, wdown_rows)
