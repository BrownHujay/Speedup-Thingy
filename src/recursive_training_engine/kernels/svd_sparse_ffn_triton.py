from __future__ import annotations

import math

import torch

try:  # pragma: no cover - import availability depends on CUDA machine.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


def _require_available() -> None:
    if not available():
        raise RuntimeError("SVDFactorSparseFFN Triton mode requires NVIDIA CUDA + Triton")


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _selector_full_kernel(
        q_up,
        q_gate,
        up_b,
        gate_b,
        wd_norm,
        candidate_ids,
        D_FF: tl.constexpr,
        RANK: tl.constexpr,
        VIEWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        TOP_M: tl.constexpr,
        RANK_BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets_m = tl.arange(0, TOP_M)

        best_up_scores = tl.full((TOP_M,), -float("inf"), tl.float32)
        best_gate_scores = tl.full((TOP_M,), -float("inf"), tl.float32)
        best_product_scores = tl.full((TOP_M,), -float("inf"), tl.float32)
        best_up_ids = tl.zeros((TOP_M,), tl.int32)
        best_gate_ids = tl.zeros((TOP_M,), tl.int32)
        best_product_ids = tl.zeros((TOP_M,), tl.int32)

        for h0 in tl.range(0, D_FF, BLOCK_H):
            offsets_h = h0 + tl.arange(0, BLOCK_H)
            valid_h = offsets_h < D_FF
            up_dot = tl.zeros((BLOCK_H,), tl.float32)
            gate_dot = tl.zeros((BLOCK_H,), tl.float32)
            for r in range(0, RANK_BLOCK):
                if r < RANK:
                    q_u = tl.load(q_up + token * RANK + r).to(tl.float32)
                    q_g = tl.load(q_gate + token * RANK + r).to(tl.float32)
                    up_vals = tl.load(
                        up_b + r * D_FF + offsets_h,
                        mask=valid_h,
                        other=0.0,
                    ).to(tl.float32)
                    gate_vals = tl.load(
                        gate_b + r * D_FF + offsets_h,
                        mask=valid_h,
                        other=0.0,
                    ).to(tl.float32)
                    up_dot += q_u * up_vals
                    gate_dot += q_g * gate_vals

            norm = tl.load(wd_norm + offsets_h, mask=valid_h, other=0.0).to(tl.float32)
            gate_act = gate_dot / (1.0 + tl.exp(-gate_dot))
            tile_up_scores = tl.where(valid_h, tl.abs(up_dot) * norm, -float("inf"))
            tile_gate_scores = tl.where(valid_h, tl.abs(gate_act) * norm, -float("inf"))
            tile_product_scores = tl.where(valid_h, tl.abs(up_dot * gate_act) * norm, -float("inf"))

            live_up = valid_h
            live_gate = valid_h
            live_product = valid_h
            for m in range(0, TOP_M):
                masked_up = tl.where(live_up, tile_up_scores, -float("inf"))
                tile_best_up_score = tl.max(masked_up, axis=0)
                tile_up_winner = masked_up == tile_best_up_score
                tile_best_up_id = tl.max(tl.where(tile_up_winner, offsets_h, 0), axis=0)
                global_min_up = tl.min(best_up_scores, axis=0)
                global_up_winner = best_up_scores == global_min_up
                replace_up = tl.max(tl.where(global_up_winner, offsets_m, 0), axis=0)
                up_better = tile_best_up_score > global_min_up
                best_up_scores = tl.where(
                    (offsets_m == replace_up) & up_better,
                    tile_best_up_score,
                    best_up_scores,
                )
                best_up_ids = tl.where(
                    (offsets_m == replace_up) & up_better,
                    tile_best_up_id,
                    best_up_ids,
                )
                live_up = live_up & (offsets_h != tile_best_up_id)

                masked_gate = tl.where(live_gate, tile_gate_scores, -float("inf"))
                tile_best_gate_score = tl.max(masked_gate, axis=0)
                tile_gate_winner = masked_gate == tile_best_gate_score
                tile_best_gate_id = tl.max(tl.where(tile_gate_winner, offsets_h, 0), axis=0)
                global_min_gate = tl.min(best_gate_scores, axis=0)
                global_gate_winner = best_gate_scores == global_min_gate
                replace_gate = tl.max(tl.where(global_gate_winner, offsets_m, 0), axis=0)
                gate_better = tile_best_gate_score > global_min_gate
                best_gate_scores = tl.where(
                    (offsets_m == replace_gate) & gate_better,
                    tile_best_gate_score,
                    best_gate_scores,
                )
                best_gate_ids = tl.where(
                    (offsets_m == replace_gate) & gate_better,
                    tile_best_gate_id,
                    best_gate_ids,
                )
                live_gate = live_gate & (offsets_h != tile_best_gate_id)

                if VIEWS == 3:
                    masked_product = tl.where(live_product, tile_product_scores, -float("inf"))
                    tile_best_product_score = tl.max(masked_product, axis=0)
                    tile_product_winner = masked_product == tile_best_product_score
                    tile_best_product_id = tl.max(
                        tl.where(tile_product_winner, offsets_h, 0),
                        axis=0,
                    )
                    global_min_product = tl.min(best_product_scores, axis=0)
                    global_product_winner = best_product_scores == global_min_product
                    replace_product = tl.max(
                        tl.where(global_product_winner, offsets_m, 0),
                        axis=0,
                    )
                    product_better = tile_best_product_score > global_min_product
                    best_product_scores = tl.where(
                        (offsets_m == replace_product) & product_better,
                        tile_best_product_score,
                        best_product_scores,
                    )
                    best_product_ids = tl.where(
                        (offsets_m == replace_product) & product_better,
                        tile_best_product_id,
                        best_product_ids,
                    )
                    live_product = live_product & (offsets_h != tile_best_product_id)

        base = token * VIEWS * TOP_M
        tl.store(candidate_ids + base + offsets_m, best_up_ids)
        tl.store(candidate_ids + base + TOP_M + offsets_m, best_gate_ids)
        if VIEWS == 3:
            tl.store(candidate_ids + base + 2 * TOP_M + offsets_m, best_product_ids)

    @triton.jit
    def _selector_tile_all_views_kernel(
        q_up,
        q_gate,
        up_b,
        gate_b,
        wd_norm,
        partial_ids,
        partial_scores,
        D_FF: tl.constexpr,
        RANK: tl.constexpr,
        TILES: tl.constexpr,
        VIEWS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        TOP_M: tl.constexpr,
        RANK_BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        tile = tl.program_id(1)
        offsets_h = tile * BLOCK_H + tl.arange(0, BLOCK_H)
        valid_h = offsets_h < D_FF

        up_dot = tl.zeros((BLOCK_H,), tl.float32)
        gate_dot = tl.zeros((BLOCK_H,), tl.float32)
        for r in range(0, RANK_BLOCK):
            if r < RANK:
                q_u = tl.load(q_up + token * RANK + r).to(tl.float32)
                q_g = tl.load(q_gate + token * RANK + r).to(tl.float32)
                up_vals = tl.load(up_b + r * D_FF + offsets_h, mask=valid_h, other=0.0).to(tl.float32)
                gate_vals = tl.load(
                    gate_b + r * D_FF + offsets_h,
                    mask=valid_h,
                    other=0.0,
                ).to(tl.float32)
                up_dot += q_u * up_vals
                gate_dot += q_g * gate_vals

        norm = tl.load(wd_norm + offsets_h, mask=valid_h, other=0.0).to(tl.float32)
        gate_act = gate_dot / (1.0 + tl.exp(-gate_dot))
        scores_up = tl.where(valid_h, tl.abs(up_dot) * norm, -float("inf"))
        scores_gate = tl.where(valid_h, tl.abs(gate_act) * norm, -float("inf"))
        scores_product = tl.where(valid_h, tl.abs(up_dot * gate_act) * norm, -float("inf"))

        live_up = valid_h
        live_gate = valid_h
        live_product = valid_h
        for m in range(0, TOP_M):
            masked_up = tl.where(live_up, scores_up, -float("inf"))
            best_score_up = tl.max(masked_up, axis=0)
            winner_up = masked_up == best_score_up
            best_id_up = tl.max(tl.where(winner_up, offsets_h, 0), axis=0)
            out_offset_up = (((token * VIEWS + 0) * TILES + tile) * TOP_M) + m
            tl.store(partial_ids + out_offset_up, best_id_up)
            tl.store(partial_scores + out_offset_up, best_score_up)
            live_up = live_up & (offsets_h != best_id_up)

            masked_gate = tl.where(live_gate, scores_gate, -float("inf"))
            best_score_gate = tl.max(masked_gate, axis=0)
            winner_gate = masked_gate == best_score_gate
            best_id_gate = tl.max(tl.where(winner_gate, offsets_h, 0), axis=0)
            out_offset_gate = (((token * VIEWS + 1) * TILES + tile) * TOP_M) + m
            tl.store(partial_ids + out_offset_gate, best_id_gate)
            tl.store(partial_scores + out_offset_gate, best_score_gate)
            live_gate = live_gate & (offsets_h != best_id_gate)

            if VIEWS == 3:
                masked_product = tl.where(live_product, scores_product, -float("inf"))
                best_score_product = tl.max(masked_product, axis=0)
                winner_product = masked_product == best_score_product
                best_id_product = tl.max(tl.where(winner_product, offsets_h, 0), axis=0)
                out_offset_product = (((token * VIEWS + 2) * TILES + tile) * TOP_M) + m
                tl.store(partial_ids + out_offset_product, best_id_product)
                tl.store(partial_scores + out_offset_product, best_score_product)
                live_product = live_product & (offsets_h != best_id_product)

    @triton.jit
    def _selector_merge_kernel(
        partial_ids,
        partial_scores,
        candidate_ids,
        N_TILES: tl.constexpr,
        VIEWS: tl.constexpr,
        TOP_M: tl.constexpr,
        MERGE_BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        view = tl.program_id(1)
        offsets = tl.arange(0, MERGE_BLOCK)
        valid = offsets < (N_TILES * TOP_M)
        base = ((token * VIEWS + view) * N_TILES * TOP_M)
        scores = tl.load(partial_scores + base + offsets, mask=valid, other=-float("inf")).to(
            tl.float32
        )
        ids = tl.load(partial_ids + base + offsets, mask=valid, other=0)

        live = valid
        for m in range(0, TOP_M):
            masked = tl.where(live, scores, -float("inf"))
            best_score = tl.max(masked, axis=0)
            winner = masked == best_score
            best_pos = tl.max(tl.where(winner, offsets, 0), axis=0)
            best_id = tl.load(partial_ids + base + best_pos)
            tl.store(candidate_ids + (token * VIEWS + view) * TOP_M + m, best_id)
            live = live & (offsets != best_pos)

    @triton.jit
    def _candidate_activation_kernel(
        x,
        candidate_ids,
        w_up,
        w_gate,
        wd_norm,
        z_values,
        candidate_scores,
        D_MODEL: tl.constexpr,
        CANDIDATES: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        c_block = tl.program_id(1)
        offsets_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        valid_c = offsets_c < CANDIDATES
        ids = tl.load(candidate_ids + token * CANDIDATES + offsets_c, mask=valid_c, other=0)

        duplicate = tl.zeros((BLOCK_C,), tl.int32)
        for p in range(0, CANDIDATES):
            prev = tl.load(candidate_ids + token * CANDIDATES + p)
            duplicate += tl.where((p < offsets_c) & (prev == ids) & valid_c, 1, 0)

        up_acc = tl.zeros((BLOCK_C,), tl.float32)
        gate_acc = tl.zeros((BLOCK_C,), tl.float32)
        for d0 in tl.range(0, D_MODEL, BLOCK_D):
            offsets_d = d0 + tl.arange(0, BLOCK_D)
            valid_d = offsets_d < D_MODEL
            x_vals = tl.load(x + token * D_MODEL + offsets_d, mask=valid_d, other=0.0).to(
                tl.float32
            )
            up_rows = tl.load(
                w_up + ids[:, None] * D_MODEL + offsets_d[None, :],
                mask=valid_c[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_rows = tl.load(
                w_gate + ids[:, None] * D_MODEL + offsets_d[None, :],
                mask=valid_c[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            up_acc += tl.sum(up_rows * x_vals[None, :], axis=1)
            gate_acc += tl.sum(gate_rows * x_vals[None, :], axis=1)

        gate_act = gate_acc / (1.0 + tl.exp(-gate_acc))
        z = up_acc * gate_act
        norm = tl.load(wd_norm + ids, mask=valid_c, other=0.0).to(tl.float32)
        scores = tl.where(valid_c & (duplicate == 0), tl.abs(z) * norm, -float("inf"))
        tl.store(z_values + token * CANDIDATES + offsets_c, z, mask=valid_c)
        tl.store(candidate_scores + token * CANDIDATES + offsets_c, scores, mask=valid_c)

    @triton.jit
    def _candidate_select_kernel(
        candidate_ids,
        z_values,
        candidate_scores,
        selected_ids,
        selected_z,
        CANDIDATES: tl.constexpr,
        TOP_K: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, CANDIDATE_BLOCK)
        valid = offsets < CANDIDATES
        scores = tl.load(
            candidate_scores + token * CANDIDATES + offsets,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        live = valid
        for k in range(0, TOP_K):
            masked = tl.where(live, scores, -float("inf"))
            best_score = tl.max(masked, axis=0)
            winner = masked == best_score
            best_pos = tl.max(tl.where(winner, offsets, 0), axis=0)
            best_id = tl.load(candidate_ids + token * CANDIDATES + best_pos)
            best_z = tl.load(z_values + token * CANDIDATES + best_pos).to(tl.float32)
            tl.store(selected_ids + token * TOP_K + k, best_id)
            tl.store(selected_z + token * TOP_K + k, best_z)
            live = live & (offsets != best_pos)

    @triton.jit
    def _candidate_activation_select_kernel(
        x,
        candidate_ids,
        w_up,
        w_gate,
        wd_norm,
        selected_ids,
        selected_z,
        D_MODEL: tl.constexpr,
        CANDIDATES: tl.constexpr,
        TOP_K: tl.constexpr,
        CANDIDATE_BLOCK: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets_c = tl.arange(0, CANDIDATE_BLOCK)
        valid_c = offsets_c < CANDIDATES
        ids = tl.load(candidate_ids + token * CANDIDATES + offsets_c, mask=valid_c, other=0)

        duplicate = tl.zeros((CANDIDATE_BLOCK,), tl.int32)
        for p in range(0, CANDIDATES):
            prev = tl.load(candidate_ids + token * CANDIDATES + p)
            duplicate += tl.where((p < offsets_c) & (prev == ids) & valid_c, 1, 0)

        up_acc = tl.zeros((CANDIDATE_BLOCK,), tl.float32)
        gate_acc = tl.zeros((CANDIDATE_BLOCK,), tl.float32)
        for d0 in tl.range(0, D_MODEL, BLOCK_D):
            offsets_d = d0 + tl.arange(0, BLOCK_D)
            valid_d = offsets_d < D_MODEL
            x_vals = tl.load(x + token * D_MODEL + offsets_d, mask=valid_d, other=0.0).to(
                tl.float32
            )
            up_rows = tl.load(
                w_up + ids[:, None] * D_MODEL + offsets_d[None, :],
                mask=valid_c[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_rows = tl.load(
                w_gate + ids[:, None] * D_MODEL + offsets_d[None, :],
                mask=valid_c[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            up_acc += tl.sum(up_rows * x_vals[None, :], axis=1)
            gate_acc += tl.sum(gate_rows * x_vals[None, :], axis=1)

        gate_act = gate_acc / (1.0 + tl.exp(-gate_acc))
        z = up_acc * gate_act
        norm = tl.load(wd_norm + ids, mask=valid_c, other=0.0).to(tl.float32)
        scores = tl.where(valid_c & (duplicate == 0), tl.abs(z) * norm, -float("inf"))
        live = valid_c
        for k in range(0, TOP_K):
            masked = tl.where(live, scores, -float("inf"))
            best_score = tl.max(masked, axis=0)
            winner = masked == best_score
            best_pos = tl.max(tl.where(winner, offsets_c, 0), axis=0)
            best_id = tl.load(candidate_ids + token * CANDIDATES + best_pos)
            best_z = tl.max(tl.where(winner, z, -float("inf")), axis=0)
            tl.store(selected_ids + token * TOP_K + k, best_id)
            tl.store(selected_z + token * TOP_K + k, best_z)
            live = live & (offsets_c != best_pos)

    @triton.jit
    def _downsum_kernel(
        selected_ids,
        selected_z,
        w_down,
        out,
        D_MODEL: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        d_block = tl.program_id(1)
        offsets_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offsets_d < D_MODEL
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for k in range(0, TOP_K):
            neuron_id = tl.load(selected_ids + token * TOP_K + k)
            z = tl.load(selected_z + token * TOP_K + k).to(tl.float32)
            down = tl.load(
                w_down + neuron_id * D_MODEL + offsets_d,
                mask=valid_d,
                other=0.0,
            ).to(tl.float32)
            acc += z * down
        tl.store(out + token * D_MODEL + offsets_d, acc, mask=valid_d)


def triton_svd_sparse_ffn_forward(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    up_a: torch.Tensor,
    up_b: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    wd_norm: torch.Tensor,
    *,
    top_k: int,
    up_m: int,
    gate_m: int | None = None,
    product_m: int = 0,
    block_h: int = 1024,
    block_d: int = 64,
    low_launch: bool = True,
) -> torch.Tensor:
    """CUDA/Triton SVD sparse FFN forward.

    This is the real NVIDIA path: cuBLAS computes the low-rank selector queries,
    Triton streams FFN neurons in H tiles to keep top-M candidate slots, then
    Triton evaluates exact candidate SwiGLU activations and sparse downsum.
    """

    _require_available()
    gate_m = up_m if gate_m is None else gate_m
    if gate_m != up_m:
        raise ValueError("Triton SVD sparse FFN currently requires gate_m == up_m")
    if product_m not in {0, up_m}:
        raise ValueError("Triton SVD sparse FFN currently requires product_m == 0 or product_m == up_m")
    if x.ndim != 2:
        raise ValueError("Triton SVD sparse FFN expects flattened x with shape [tokens, d_model]")
    tensors = (x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
    if any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError("Triton SVD sparse FFN requires all tensors on CUDA")
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("Triton SVD sparse FFN is an inference/eval kernel; backward is not implemented")
    if any(not tensor.is_contiguous() for tensor in tensors):
        tensors = tuple(tensor.contiguous() for tensor in tensors)
        x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm = tensors

    n_tokens, d_model = x.shape
    d_ff = w_up.shape[0]
    rank = up_a.shape[1]
    top_m = min(int(up_m), d_ff)
    if top_m <= 0:
        raise ValueError("Triton SVD sparse FFN requires up_m > 0")
    top_k = min(int(top_k), top_m * (2 + int(product_m > 0)))
    views = 2 + int(product_m > 0)
    tiles = triton.cdiv(d_ff, block_h)
    rank_block = _next_power_of_2(rank)
    merge_block = _next_power_of_2(tiles * top_m)
    candidate_slots = views * top_m
    candidate_block = _next_power_of_2(candidate_slots)
    block_d = _next_power_of_2(block_d)

    q_up = x @ up_a
    q_gate = x @ gate_a

    candidate_ids = torch.empty((n_tokens, candidate_slots), device=x.device, dtype=torch.int32)
    if low_launch:
        _selector_full_kernel[(n_tokens,)](
            q_up,
            q_gate,
            up_b,
            gate_b,
            wd_norm,
            candidate_ids,
            d_ff,
            rank,
            views,
            block_h,
            top_m,
            rank_block,
            num_warps=8,
        )
    else:
        partial_ids = torch.empty(
            (n_tokens, views, tiles, top_m),
            device=x.device,
            dtype=torch.int32,
        )
        partial_scores = torch.empty(
            (n_tokens, views, tiles, top_m),
            device=x.device,
            dtype=torch.float32,
        )
        _selector_tile_all_views_kernel[(n_tokens, tiles)](
            q_up,
            q_gate,
            up_b,
            gate_b,
            wd_norm,
            partial_ids,
            partial_scores,
            d_ff,
            rank,
            tiles,
            views,
            block_h,
            top_m,
            rank_block,
            num_warps=8,
        )
        _selector_merge_kernel[(n_tokens, views)](
            partial_ids,
            partial_scores,
            candidate_ids,
            tiles,
            views,
            top_m,
            merge_block,
            num_warps=8,
        )

    selected_ids = torch.empty((n_tokens, top_k), device=x.device, dtype=torch.int32)
    selected_z = torch.empty((n_tokens, top_k), device=x.device, dtype=torch.float32)
    if low_launch:
        _candidate_activation_select_kernel[(n_tokens,)](
            x,
            candidate_ids,
            w_up,
            w_gate,
            wd_norm,
            selected_ids,
            selected_z,
            d_model,
            candidate_slots,
            top_k,
            candidate_block,
            block_d,
            num_warps=8,
        )
    else:
        block_c = 32
        z_values = torch.empty((n_tokens, candidate_slots), device=x.device, dtype=torch.float32)
        candidate_scores = torch.empty(
            (n_tokens, candidate_slots),
            device=x.device,
            dtype=torch.float32,
        )
        _candidate_activation_kernel[(n_tokens, triton.cdiv(candidate_slots, block_c))](
            x,
            candidate_ids,
            w_up,
            w_gate,
            wd_norm,
            z_values,
            candidate_scores,
            d_model,
            candidate_slots,
            _next_power_of_2(block_c),
            block_d,
            num_warps=4,
        )
        _candidate_select_kernel[(n_tokens,)](
            candidate_ids,
            z_values,
            candidate_scores,
            selected_ids,
            selected_z,
            candidate_slots,
            top_k,
            candidate_block,
            num_warps=4,
        )

    out = torch.empty((n_tokens, d_model), device=x.device, dtype=x.dtype)
    _downsum_kernel[(n_tokens, triton.cdiv(d_model, block_d))](
        selected_ids,
        selected_z,
        w_down,
        out,
        d_model,
        top_k,
        block_d,
        num_warps=4,
    )
    return out
