from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch


def require_mlx():
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - exercised only without optional dep.
        raise RuntimeError(
            "MLX is required for this backend. Run with `uv run --with mlx ...` "
            "or install the optional MLX package on Apple Silicon."
        ) from exc
    return mx


def _silu(mx, x):
    return x * mx.sigmoid(x)


def _top_indices(mx, scores, k: int):
    return mx.argpartition(-scores, kth=k, axis=-1)[:, :k]


def _first_occurrence_mask(mx, ids):
    slots = ids.shape[-1]
    if slots <= 1:
        return mx.ones_like(ids, dtype=mx.bool_)
    same = ids[:, :, None] == ids[:, None, :]
    earlier = mx.triu(mx.ones((slots, slots), dtype=mx.bool_), k=1)
    return ~mx.any(same & earlier[None, :, :], axis=1)


def build_mlx_svd_sparse_ffn(
    *,
    top_k: int,
    up_m: int,
    gate_m: int | None = None,
    product_m: int = 0,
    compile_fn: bool = True,
) -> Callable[..., Any]:
    mx = require_mlx()
    gate_m = up_m if gate_m is None else gate_m

    def forward(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm):
        q_up = x @ up_a
        q_gate = x @ gate_a
        up_hat = q_up @ up_b
        gate_hat = q_gate @ gate_b
        gate_hat_act = _silu(mx, gate_hat)
        pieces = [
            _top_indices(mx, mx.abs(up_hat) * wd_norm[None, :], up_m),
            _top_indices(mx, mx.abs(gate_hat_act) * wd_norm[None, :], gate_m),
        ]
        if product_m > 0:
            pieces.append(
                _top_indices(mx, mx.abs(up_hat * gate_hat_act) * wd_norm[None, :], product_m)
            )
        candidate_ids = mx.concatenate(pieces, axis=-1)
        candidate_valid = _first_occurrence_mask(mx, candidate_ids)

        up_rows = mx.take(w_up, candidate_ids, axis=0)
        gate_rows = mx.take(w_gate, candidate_ids, axis=0)
        exact_up = mx.sum(x[:, None, :] * up_rows, axis=-1)
        exact_gate = mx.sum(x[:, None, :] * gate_rows, axis=-1)
        z = exact_up * _silu(mx, exact_gate)
        scores = mx.where(
            candidate_valid,
            mx.abs(z) * mx.take(wd_norm, candidate_ids, axis=0),
            -mx.inf,
        )
        selected_local = _top_indices(mx, scores, top_k)
        selected_ids = mx.take_along_axis(candidate_ids, selected_local, axis=-1)
        selected_z = mx.take_along_axis(z, selected_local, axis=-1)
        down_rows = mx.take(w_down, selected_ids, axis=0)
        return mx.sum(selected_z[:, :, None] * down_rows, axis=1)

    return mx.compile(forward) if compile_fn else forward


def build_mlx_svd_candidate_slots(
    *,
    up_m: int,
    gate_m: int | None = None,
    product_m: int = 0,
    compile_fn: bool = True,
) -> Callable[..., Any]:
    mx = require_mlx()
    gate_m = up_m if gate_m is None else gate_m

    def forward(x, up_a, up_b, gate_a, gate_b, wd_norm):
        q_up = x @ up_a
        q_gate = x @ gate_a
        up_hat = q_up @ up_b
        gate_hat = q_gate @ gate_b
        gate_hat_act = _silu(mx, gate_hat)
        pieces = [
            _top_indices(mx, mx.abs(up_hat) * wd_norm[None, :], up_m),
            _top_indices(mx, mx.abs(gate_hat_act) * wd_norm[None, :], gate_m),
        ]
        if product_m > 0:
            pieces.append(
                _top_indices(mx, mx.abs(up_hat * gate_hat_act) * wd_norm[None, :], product_m)
            )
        return mx.concatenate(pieces, axis=-1)

    return mx.compile(forward) if compile_fn else forward


def build_mlx_dense_swiglu(*, compile_fn: bool = True) -> Callable[..., Any]:
    mx = require_mlx()

    def forward(x, w_up, w_gate, w_down):
        return ((x @ w_up.T) * _silu(mx, x @ w_gate.T)) @ w_down

    return mx.compile(forward) if compile_fn else forward


def build_metal_candidate_swiglu_downsum(
    *,
    d_model: int,
    candidate_slots: int,
    top_k: int,
    threads: int = 256,
) -> Callable[..., Any]:
    mx = require_mlx()
    source = r"""
        uint token = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;

        threadgroup float partial_up[THREADS];
        threadgroup float partial_gate[THREADS];
        threadgroup float up_vals[CANDIDATE_SLOTS];
        threadgroup float gate_vals[CANDIDATE_SLOTS];
        threadgroup float z_vals[CANDIDATE_SLOTS];
        threadgroup float score_vals[CANDIDATE_SLOTS];
        threadgroup int valid_vals[CANDIDATE_SLOTS];
        threadgroup float selected_z[TOP_K];
        threadgroup int selected_ids[TOP_K];

        for (int c = 0; c < CANDIDATE_SLOTS; ++c) {
            int id = int(candidate_ids[token * CANDIDATE_SLOTS + c]);
            bool valid = true;
            for (int p = 0; p < c; ++p) {
                valid = valid && (int(candidate_ids[token * CANDIDATE_SLOTS + p]) != id);
            }

            float acc_up = 0.0f;
            float acc_gate = 0.0f;
            for (int d = int(lid); d < D_MODEL; d += THREADS) {
                float xv = x[token * D_MODEL + d];
                acc_up += xv * w_up[id * D_MODEL + d];
                acc_gate += xv * w_gate[id * D_MODEL + d];
            }
            partial_up[lid] = acc_up;
            partial_gate[lid] = acc_gate;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    partial_up[lid] += partial_up[lid + stride];
                    partial_gate[lid] += partial_gate[lid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                float gate_act = partial_gate[0] / (1.0f + metal::exp(-partial_gate[0]));
                float z = partial_up[0] * gate_act;
                up_vals[c] = partial_up[0];
                gate_vals[c] = partial_gate[0];
                z_vals[c] = z;
                valid_vals[c] = valid ? 1 : 0;
                score_vals[c] = valid ? metal::abs(z) * wd_norm[id] : -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            for (int s = 0; s < TOP_K; ++s) {
                float best_score = -INFINITY;
                int best_c = 0;
                for (int c = 0; c < CANDIDATE_SLOTS; ++c) {
                    if (score_vals[c] > best_score) {
                        best_score = score_vals[c];
                        best_c = c;
                    }
                }
                int id = int(candidate_ids[token * CANDIDATE_SLOTS + best_c]);
                selected_ids[s] = id;
                selected_z[s] = z_vals[best_c];
                score_vals[best_c] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int d = int(lid); d < D_MODEL; d += THREADS) {
            float acc = 0.0f;
            for (int s = 0; s < TOP_K; ++s) {
                int id = selected_ids[s];
                acc += selected_z[s] * w_down[id * D_MODEL + d];
            }
            out[token * D_MODEL + d] = acc;
        }
    """
    kernel = mx.fast.metal_kernel(
        name="parallel_candidate_swiglu_downsum",
        input_names=["x", "candidate_ids", "w_up", "w_gate", "w_down", "wd_norm"],
        output_names=["out"],
        source=source,
    )

    def forward(x, candidate_ids, w_up, w_gate, w_down, wd_norm):
        return kernel(
            inputs=[x, candidate_ids, w_up, w_gate, w_down, wd_norm],
            template=[
                ("D_MODEL", d_model),
                ("CANDIDATE_SLOTS", candidate_slots),
                ("TOP_K", top_k),
                ("THREADS", threads),
            ],
            grid=(x.shape[0] * threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(x.shape[0], d_model)],
            output_dtypes=[x.dtype],
        )[0]

    return forward


def build_metal_fused_svd_sparse_ffn(
    *,
    d_model: int,
    d_ff: int,
    rank: int,
    top_k: int,
    up_m: int,
    gate_m: int | None = None,
    product_m: int = 0,
) -> Callable[..., Any]:
    mx = require_mlx()
    gate_m = up_m if gate_m is None else gate_m
    product_storage = max(product_m, 1)
    source = r"""
        uint n = thread_position_in_grid.x;

        float q_up[RANK];
        float q_gate[RANK];
        for (int r = 0; r < RANK; ++r) {
            float acc_up = 0.0f;
            float acc_gate = 0.0f;
            for (int d = 0; d < D_MODEL; ++d) {
                float xv = x[n * D_MODEL + d];
                acc_up += xv * up_a[d * RANK + r];
                acc_gate += xv * gate_a[d * RANK + r];
            }
            q_up[r] = acc_up;
            q_gate[r] = acc_gate;
        }

        float up_scores[UP_M];
        int up_ids[UP_M];
        float gate_scores[GATE_M];
        int gate_ids[GATE_M];
        float product_scores[PRODUCT_STORAGE];
        int product_ids[PRODUCT_STORAGE];
        for (int i = 0; i < UP_M; ++i) {
            up_scores[i] = -INFINITY;
            up_ids[i] = 0;
        }
        for (int i = 0; i < GATE_M; ++i) {
            gate_scores[i] = -INFINITY;
            gate_ids[i] = 0;
        }
        for (int i = 0; i < PRODUCT_STORAGE; ++i) {
            product_scores[i] = -INFINITY;
            product_ids[i] = 0;
        }

        for (int j = 0; j < D_FF; ++j) {
            float up_hat = 0.0f;
            float gate_hat = 0.0f;
            for (int r = 0; r < RANK; ++r) {
                up_hat += q_up[r] * up_b[r * D_FF + j];
                gate_hat += q_gate[r] * gate_b[r * D_FF + j];
            }
            float gate_act = gate_hat / (1.0f + metal::exp(-gate_hat));
            float norm = wd_norm[j];
            float score = metal::abs(up_hat) * norm;
            if (score > up_scores[UP_M - 1]) {
                up_scores[UP_M - 1] = score;
                up_ids[UP_M - 1] = j;
                for (int p = UP_M - 1; p > 0; --p) {
                    if (up_scores[p] > up_scores[p - 1]) {
                        float ts = up_scores[p - 1];
                        int ti = up_ids[p - 1];
                        up_scores[p - 1] = up_scores[p];
                        up_ids[p - 1] = up_ids[p];
                        up_scores[p] = ts;
                        up_ids[p] = ti;
                    }
                }
            }
            score = metal::abs(gate_act) * norm;
            if (score > gate_scores[GATE_M - 1]) {
                gate_scores[GATE_M - 1] = score;
                gate_ids[GATE_M - 1] = j;
                for (int p = GATE_M - 1; p > 0; --p) {
                    if (gate_scores[p] > gate_scores[p - 1]) {
                        float ts = gate_scores[p - 1];
                        int ti = gate_ids[p - 1];
                        gate_scores[p - 1] = gate_scores[p];
                        gate_ids[p - 1] = gate_ids[p];
                        gate_scores[p] = ts;
                        gate_ids[p] = ti;
                    }
                }
            }
            if (PRODUCT_M > 0) {
                score = metal::abs(up_hat * gate_act) * norm;
                if (score > product_scores[PRODUCT_M - 1]) {
                    product_scores[PRODUCT_M - 1] = score;
                    product_ids[PRODUCT_M - 1] = j;
                    for (int p = PRODUCT_M - 1; p > 0; --p) {
                        if (product_scores[p] > product_scores[p - 1]) {
                            float ts = product_scores[p - 1];
                            int ti = product_ids[p - 1];
                            product_scores[p - 1] = product_scores[p];
                            product_ids[p - 1] = product_ids[p];
                            product_scores[p] = ts;
                            product_ids[p] = ti;
                        }
                    }
                }
            }
        }

        float selected_scores[TOP_K];
        float selected_z[TOP_K];
        int selected_ids[TOP_K];
        for (int i = 0; i < TOP_K; ++i) {
            selected_scores[i] = -INFINITY;
            selected_z[i] = 0.0f;
            selected_ids[i] = -1;
        }

        for (int slot = 0; slot < TOTAL_CANDIDATES; ++slot) {
            int id = 0;
            if (slot < UP_M) {
                id = up_ids[slot];
            } else if (slot < UP_M + GATE_M) {
                id = gate_ids[slot - UP_M];
            } else {
                id = product_ids[slot - UP_M - GATE_M];
            }
            bool duplicate = false;
            for (int s = 0; s < TOP_K; ++s) {
                duplicate = duplicate || (selected_ids[s] == id);
            }
            if (duplicate) {
                continue;
            }

            float exact_up = 0.0f;
            float exact_gate = 0.0f;
            for (int d = 0; d < D_MODEL; ++d) {
                float xv = x[n * D_MODEL + d];
                exact_up += xv * w_up[id * D_MODEL + d];
                exact_gate += xv * w_gate[id * D_MODEL + d];
            }
            float z = exact_up * (exact_gate / (1.0f + metal::exp(-exact_gate)));
            float score = metal::abs(z) * wd_norm[id];
            if (score > selected_scores[TOP_K - 1]) {
                selected_scores[TOP_K - 1] = score;
                selected_ids[TOP_K - 1] = id;
                selected_z[TOP_K - 1] = z;
                for (int p = TOP_K - 1; p > 0; --p) {
                    if (selected_scores[p] > selected_scores[p - 1]) {
                        float ts = selected_scores[p - 1];
                        int ti = selected_ids[p - 1];
                        float tz = selected_z[p - 1];
                        selected_scores[p - 1] = selected_scores[p];
                        selected_ids[p - 1] = selected_ids[p];
                        selected_z[p - 1] = selected_z[p];
                        selected_scores[p] = ts;
                        selected_ids[p] = ti;
                        selected_z[p] = tz;
                    }
                }
            }
        }

        for (int d = 0; d < D_MODEL; ++d) {
            float acc = 0.0f;
            for (int s = 0; s < TOP_K; ++s) {
                int id = selected_ids[s];
                if (id >= 0) {
                    acc += selected_z[s] * w_down[id * D_MODEL + d];
                }
            }
            out[n * D_MODEL + d] = acc;
        }
    """
    kernel = mx.fast.metal_kernel(
        name="fused_svd_sparse_ffn_serial_token",
        input_names=["x", "w_up", "w_gate", "w_down", "up_a", "up_b", "gate_a", "gate_b", "wd_norm"],
        output_names=["out"],
        source=source,
    )

    def forward(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm):
        return kernel(
            inputs=[x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm],
            template=[
                ("D_MODEL", d_model),
                ("D_FF", d_ff),
                ("RANK", rank),
                ("TOP_K", top_k),
                ("UP_M", up_m),
                ("GATE_M", gate_m),
                ("PRODUCT_M", product_m),
                ("PRODUCT_STORAGE", product_storage),
                ("TOTAL_CANDIDATES", up_m + gate_m + product_m),
            ],
            grid=(x.shape[0], 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(x.shape[0], d_model)],
            output_dtypes=[x.dtype],
        )[0]

    return forward


def torch_svd_ffn_to_mlx(sparse_ffn) -> dict[str, Any]:
    mx = require_mlx()

    def convert(tensor: torch.Tensor):
        return mx.array(tensor.detach().cpu().float().numpy())

    return {
        "w_up": convert(sparse_ffn.w_up),
        "w_gate": convert(sparse_ffn.w_gate),
        "w_down": convert(sparse_ffn.w_down),
        "up_a": convert(sparse_ffn.up_a),
        "up_b": convert(sparse_ffn.up_b),
        "gate_a": convert(sparse_ffn.gate_a),
        "gate_b": convert(sparse_ffn.gate_b),
        "wd_norm": convert(sparse_ffn.w_down.detach().float().norm(dim=-1)),
    }


@dataclass(slots=True)
class MLXSparseFFNBenchmarkRow:
    backend: str
    d_model: int
    d_ff: int
    tokens: int
    rank: int
    factor_m: int
    product_factor_m: int
    k: int
    dense_ms: float
    sparse_ms: float
    measured_speedup: float
    estimated_speedup: float
    max_abs_diff_compiled_vs_eager: float


def benchmark_mlx_svd_sparse_ffn(
    *,
    sizes: list[tuple[int, int, int]],
    rank: int,
    factor_m: int,
    product_factor_m: int,
    k: int,
    iters: int,
    warmup: int,
    seed: int = 0,
    backend: str = "graph",
) -> list[MLXSparseFFNBenchmarkRow]:
    mx = require_mlx()
    if backend == "parallel":
        backend = "hybrid"
    if backend not in {"graph", "metal", "hybrid"}:
        raise ValueError("backend must be 'graph', 'metal', 'parallel', or 'hybrid'")
    rng = np.random.default_rng(seed)
    rows: list[MLXSparseFFNBenchmarkRow] = []
    for d_model, d_ff, tokens in sizes:
        rank_i = min(rank, d_model, d_ff)
        k_i = min(k, d_ff)
        factor_m_i = min(factor_m, d_ff)
        product_m_i = min(product_factor_m, d_ff)
        x = mx.array(rng.standard_normal((tokens, d_model)).astype(np.float32))
        w_up = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32) / np.sqrt(d_model))
        w_gate = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32) / np.sqrt(d_model))
        w_down = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32) / np.sqrt(d_model))
        up_a = mx.array(rng.standard_normal((d_model, rank_i)).astype(np.float32) / np.sqrt(d_model))
        up_b = mx.array(rng.standard_normal((rank_i, d_ff)).astype(np.float32) / np.sqrt(rank_i))
        gate_a = mx.array(rng.standard_normal((d_model, rank_i)).astype(np.float32) / np.sqrt(d_model))
        gate_b = mx.array(rng.standard_normal((rank_i, d_ff)).astype(np.float32) / np.sqrt(rank_i))
        wd_norm = mx.sqrt(mx.sum(w_down * w_down, axis=-1))

        dense = build_mlx_dense_swiglu(compile_fn=True)
        sparse_eager = build_mlx_svd_sparse_ffn(
            top_k=k_i,
            up_m=factor_m_i,
            product_m=product_m_i,
            compile_fn=False,
        )
        if backend == "metal":
            sparse = build_metal_fused_svd_sparse_ffn(
                d_model=d_model,
                d_ff=d_ff,
                rank=rank_i,
                top_k=k_i,
                up_m=factor_m_i,
                product_m=product_m_i,
            )
        elif backend == "hybrid":
            selector = build_mlx_svd_candidate_slots(
                up_m=factor_m_i,
                product_m=product_m_i,
                compile_fn=True,
            )
            candidate_slots = (2 * factor_m_i) + product_m_i
            threads = 256 if d_model >= 256 else 64
            candidate_kernel = build_metal_candidate_swiglu_downsum(
                d_model=d_model,
                candidate_slots=candidate_slots,
                top_k=k_i,
                threads=threads,
            )

            def sparse(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm):
                candidate_ids = selector(x, up_a, up_b, gate_a, gate_b, wd_norm)
                return candidate_kernel(x, candidate_ids, w_up, w_gate, w_down, wd_norm)

        else:
            sparse = build_mlx_svd_sparse_ffn(
                top_k=k_i,
                up_m=factor_m_i,
                product_m=product_m_i,
                compile_fn=True,
            )

        for _ in range(warmup):
            mx.eval(dense(x, w_up, w_gate, w_down))
            mx.eval(sparse(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm))

        eager_out = sparse_eager(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
        compiled_out = sparse(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
        mx.eval(eager_out, compiled_out)
        max_abs_diff = float(mx.max(mx.abs(eager_out - compiled_out)).item())

        def measure(fn):
            start = time.perf_counter()
            for _ in range(max(iters, 1)):
                mx.eval(fn())
            return (time.perf_counter() - start) / max(iters, 1)

        dense_seconds = measure(lambda: dense(x, w_up, w_gate, w_down))
        sparse_seconds = measure(
            lambda: sparse(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
        )
        dense_ops = 3.0 * d_model * d_ff
        sparse_ops = (
            2.0 * d_model * rank_i
            + 2.0 * rank_i * d_ff
            + 2.0 * d_model * min(2 * factor_m_i + product_m_i, d_ff)
            + d_model * k_i
        )
        rows.append(
            MLXSparseFFNBenchmarkRow(
                backend=backend,
                d_model=d_model,
                d_ff=d_ff,
                tokens=tokens,
                rank=rank_i,
                factor_m=factor_m_i,
                product_factor_m=product_m_i,
                k=k_i,
                dense_ms=dense_seconds * 1000.0,
                sparse_ms=sparse_seconds * 1000.0,
                measured_speedup=dense_seconds / max(sparse_seconds, 1e-12),
                estimated_speedup=dense_ops / max(sparse_ops, 1.0),
                max_abs_diff_compiled_vs_eager=max_abs_diff,
            )
        )
    return rows
