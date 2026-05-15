# Kernel Notes

This file tracks the current fused/kernel-shaped paths in the project and the
CUDA/Triton kernels that should replace PyTorch fallback prototypes later.

## Current Runtime Kernel Wrappers

These symbols live under `src/recursive_training_engine/kernels`. On macOS/MPS
they are mostly PyTorch reference/fallback implementations. CUDA deployments can
replace the same call sites with Triton/CUDA kernels.

| Symbol | Current behavior | CUDA/Triton target |
| --- | --- | --- |
| `k_fused_rmsnorm` | Vectorized PyTorch RMSNorm, optional residual add. | Fused residual add + RMSNorm + scale. |
| `k_rope_apply` | PyTorch rotary embedding application. | Fused RoPE inside Q/K load path. |
| `k_qkv_dense` | Dense Q/K/V matmuls. | Fused QKV projection or vendor GEMM path. |
| `k_flash_causal_dense` | Torch SDPA, flash only when CUDA supports it. | FlashAttention-only causal attention. |
| `k_qkv_grouped` | Selects active Q/K/V columns then matmuls. | Gather-free grouped QKV projection for active heads. |
| `k_flash_causal_grouped` | Torch SDPA over active grouped heads. | FlashAttention over packed active heads. |
| `k_out_proj_grouped` | Grouped output projection through selected rows. | Fused grouped output projection. |
| `k_swiglu_dense` | Dense SwiGLU using PyTorch matmuls. | Fused dense SwiGLU if useful, otherwise vendor GEMMs. |
| `k_swiglu_grouped` | Group-indexed SwiGLU slab projection. | Packed grouped SwiGLU slab kernel. |
| `k_pack_by_recipe` / `k_unpack_from_recipe` | Sort/scatter tensors by route recipe. | Route-packing kernel with prefix offsets. |
| `k_macro_phi` | PyTorch macro operator. | Fused MacroV2 operator. |
| `k_recurrent_exact_loop` | Python recurrence loop. | CUDA graph / persistent recurrence dispatch. |
| `k_logits_full` | Full vocabulary matmul. | Vendor GEMM or shortlist-aware fused output. |
| `k_logits_shortlist` | Gathers shortlist rows and dot-products. | Fused shortlist logits kernel. |
| `k_cross_entropy_unreduced` | PyTorch CE. | Fused logits/CE for shortlist or full vocab path. |
| `k_sample_audit_mask` | Torch Bernoulli sampling. | Lightweight CUDA random/audit mask kernel. |
| `k_metrics_reduce` | CPU-readable reductions. | Device-side metric reductions with compact host copy. |

## New FFN Prototype: `SVDFactorSparseFFN`

The new hot-path prototype lives in `src/recursive_training_engine/layers.py`.
It is intentionally still a PyTorch implementation, but its stages correspond
to kernels we should write once the math is locked.

An optional MLX fused-graph version lives in
`src/recursive_training_engine/mlx_svd_ffn.py`. It compiles the fixed-slot SVD
sparse FFN as one MLX graph for Apple Silicon experiments. It is still expressed
with MLX array primitives rather than a custom Metal kernel, but it fuses the
whole logical FFN path into a single compiled function:

```text
selector scores
→ fixed candidate slots
→ duplicate validity mask
→ exact candidate up/gate activation
→ norm rerank
→ sparse down projection
```

Run it with:

```bash
uv run --extra metal rte benchmark-mlx-svd-sparse-ffn
```

or, without installing the optional extra into the project environment:

```bash
uv run --with mlx rte benchmark-mlx-svd-sparse-ffn
```

### Logical Stages

| Stage | Current implementation | Why it is slow now | CUDA/Triton target |
| --- | --- | --- | --- |
| `selector_score_time` | Computes `q_up = x @ A_up`, `q_gate = x @ A_gate`, then full low-rank scores `q @ B`. | Materializes full `[tokens, d_ff]` score tensors. | Blockwise top-M selector that streams `B` tiles and keeps per-token top-M in registers/shared memory. |
| `candidate_union_dedup_time` (`candidate_mode=mask`) | Builds a full boolean `[tokens, d_ff]` mask, scatters top-M ids, then topks the mask. | Full mask is memory-heavy and not a hot-path shape. | Avoid full masks; emit fixed candidate slots or compact unique slots directly. |
| `candidate_union_dedup_time` (`candidate_mode=slots`) | Concats `topM(up)`, `topM(gate)`, optional `topM(product)`, then cheap duplicate suppression inside candidate slots. | Still uses PyTorch topk and pairwise duplicate masking. | Fixed-slot candidate kernel; allow duplicates or suppress duplicates during candidate write. |
| `candidate_mode=triton` | CUDA-only inference path using real Triton kernels for streaming selector top-M, candidate exact activation, rerank, and down projection. | Requires NVIDIA CUDA + Triton and has no backward yet. | This is the first actual CUDA target for FFN-side speed work. |
| `exact_candidate_activation_time` | Advanced-index gathers `W_up/W_gate` rows into `[tokens, candidates, d]`, then `einsum`. | Gather creates large irregular tensors and many small dot products. | Fused candidate activation: for each token and candidate id, compute exact up/gate dot products directly from weight rows. |
| `rerank_topk_time` | Computes exact contribution norm scores and PyTorch topk. | Separate topk pass over candidate tensor. | Fuse score calculation with candidate activation and maintain top-k selected candidates. |
| `down_sum_time` | Gathers selected `W_down` rows and sums `z_j * W_down[j]`. | Irregular gather plus reduction over selected rows. | Fused selected-neuron down projection accumulation. |

### MLX Fused Graph

| Function | Current behavior | Notes |
| --- | --- | --- |
| `build_mlx_svd_sparse_ffn` | Builds an MLX function and optionally wraps it in `mx.compile`. | This is the Mac/Metal prototype for the entire sparse FFN path. It is graph-fused but still relies on MLX implementations of `argpartition`, `take`, and gather-like indexing. |
| `build_mlx_svd_candidate_slots` | Builds only the MLX SVD factor-union selector and returns fixed candidate id slots. | Used by the hybrid backend so selector top-M stays in MLX while candidate activation/down projection move to a custom Metal kernel. |
| `build_metal_fused_svd_sparse_ffn` | Builds one custom MLX Metal kernel for the entire sparse FFN forward. | This avoids materializing score/candidate tensors, but the current prototype assigns one serial Metal program per token, so it is a correctness/shape test rather than a performant parallel kernel. |
| `build_metal_candidate_swiglu_downsum` | Builds a custom MLX Metal kernel for candidate activation, duplicate suppression, norm rerank, and sparse down projection. | One threadgroup handles one token. Lanes split exact up/gate dot products and the final down projection across `d_model`, so this is the first Mac-side kernel that exposes real parallelism inside the sparse FFN hot path. |
| `build_mlx_dense_swiglu` | MLX dense SwiGLU baseline. | Used to compare dense versus sparse FFN timing on the same backend. |
| `benchmark-mlx-svd-sparse-ffn` | CLI benchmark for MLX dense and MLX SVD sparse FFN. | Optional dependency; does not affect the PyTorch path. |

### Candidate Modes

| Mode | Purpose | Notes |
| --- | --- | --- |
| `mask` | Exact oracle-style dedup path. | Best for equivalence and quality checks. It still creates a full `[tokens, d_ff]` mask and is not the desired hot path. |
| `slots` | Fixed-slot no-full-mask path. | Uses `topM(up) ∪ topM(gate) ∪ topM(product)` candidate slots and suppresses duplicate slots cheaply. This is closer to the CUDA kernel shape. |

## CUDA/Triton Kernels

The first CUDA implementation lives in
`src/recursive_training_engine/kernels/svd_sparse_ffn_triton.py` and is selected
with:

```bash
uv run --extra cuda rte benchmark-svd-sparse-ffn \
  --candidate-mode triton \
  --size 2048x8192x1024 \
  --rank 48 \
  --factor-m 64 \
  --product-factor-m 64 \
  --k 64
```

This path is intentionally CUDA-only. It does not silently fall back to PyTorch
when `candidate_mode=triton` is requested.

### Implemented CUDA/Triton Kernels

| Kernel | Role | Shape |
| --- | --- | --- |
| `_selector_tile_all_views_kernel` | Streams a tile of FFN neurons once and keeps top-M up, gate, and optional product candidates for one token. | Grid: `[tokens, ceil(d_ff / block_h)]`. Does not materialize `[tokens, d_ff]` scores and does not recompute the same tile for each view. |
| `_selector_full_kernel` | Low-launch selector that streams all FFN tiles inside one program per token and writes final candidate slots directly. | Grid: `[tokens]`. Reduces launch count and intermediate partial buffers; best when token count is high enough to cover occupancy. |
| `_selector_merge_kernel` | Merges per-tile top-M candidates into final fixed slots for each token/view. | Grid: `[tokens, views]`. Small top-M reduction over tile winners. |
| `_candidate_activation_kernel` | Computes exact `x·W_up_j`, `x·W_gate_j`, SwiGLU activation, duplicate suppression, and contribution-norm scores for candidate slots. | Grid: `[tokens, ceil(candidate_slots / block_c)]`. Uses vectorized candidate blocks, not one serial program per token. |
| `_candidate_select_kernel` | Selects final top-k candidate neurons from exact candidate scores. | Grid: `[tokens]`. Small top-k over candidate slots. |
| `_candidate_activation_select_kernel` | Low-launch candidate kernel that computes exact candidate activations and final top-k selection in one program per token. | Avoids materializing candidate `z` and score tensors before selection. |
| `_downsum_kernel` | Accumulates `sum_j z_j * W_down[j]` into the FFN residual. | Grid: `[tokens, ceil(d_model / block_d)]`. Parallel over output channels. |

The default Triton forward now uses a GEMM selector path:

```text
cuBLAS q_up GEMM
cuBLAS q_gate GEMM
cuBLAS up_hat = q_up @ B_up
cuBLAS gate_hat = q_gate @ B_gate
torch topk candidate ids
_candidate_activation_kernel
_candidate_select_kernel
_downsum_kernel
```

The hand-written Triton selector paths remain in the file as experiments, but
they are not the default. Both the low-launch and tile-parallel Triton selectors
lost badly on T4 because they evaluate rank-48 dot products with scalar/vector
Triton loops instead of tensor-core GEMM scheduling. The selector is a low-rank
matrix multiply problem, so the fast path uses cuBLAS GEMMs and lets Triton do
only the irregular sparse work that cuBLAS cannot express.

### CUDA/Triton Kernels To Build Later

These are the next kernels after the initial Triton forward path.

1. **Persistent / grouped blockwise SVD selector**
   - Inputs: token hidden `x`, low-rank factors `A_up/B_up`, `A_gate/B_gate`.
   - Output: fixed candidate id slots for up, gate, and optional product views.
   - Initial Triton version exists. Next pass should tune `block_h`, use more
     persistent tiling, and reduce launch count for the three selector views.

2. **Fixed-slot candidate union / duplicate handling**
   - Inputs: top-M id lists from factor views.
   - Output: candidate slot table plus optional validity mask.
   - Initial duplicate suppression happens in the candidate activation kernel.
     Later kernels can suppress earlier if recomputation becomes measurable.

3. **More fused sparse SwiGLU candidate activation**
   - Inputs: `x`, candidate ids, `W_up`, `W_gate`.
   - Output: exact candidate activations `z_j = (x·W_up_j) * silu(x·W_gate_j)`.
   - Initial Triton version exists. Next pass should tune candidate blocking,
     vector widths, and FP16/BF16 accumulation tradeoffs.

4. **Fused candidate rerank**
   - Inputs: candidate `z_j`, `||W_down_j||`, optional candidate validity.
   - Output: selected top-k candidate ids and activations.
   - Start with contribution-norm rerank. OMP is an oracle/training target, not
     a first hot-path kernel.

5. **Fused sparse down projection**
   - Inputs: selected ids, selected activations, `W_down`.
   - Output: FFN residual.
   - Accumulate `sum_j z_j * W_down_j` per token directly.

6. **Single fused SVD sparse FFN / fewer launches**
   - Combines candidate activation, rerank, and down projection. The selector
     may remain a separate kernel if top-M over `d_ff` is the dominant stage.
   - Initial Triton forward currently uses several kernels plus cuBLAS query
     GEMMs. A production path should reduce launch count and keep intermediate
     ids/scores in compact buffers.
   - MLX custom Metal serial-token prototype now exists. It proves whole-path
     custom Metal correctness, but it does not expose enough parallelism to be
     the final fast kernel.

7. **Sparse FFN backward**
   - Needed once continuation training uses this path.
   - Only selected/candidate rows receive gradients on the sparse path; coverage
     or dense-audit updates may be separate.

## Current Quality/Speed Status

- Quality gate passed:
  - `rank48, m64, k64`: hot NLL matches oracle (`3.1077236`).
  - `rank64, m64, product64, k64`: hot NLL matches oracle (`3.1059841`).
- Speed gate failed in PyTorch:
  - `2048x8192`: estimated FFN speedup ~`26.5x`, measured MPS prototype
    speedup ~`0.49x`.
- MLX graph backend:
  - Added as optional `metal` extra and `benchmark-mlx-svd-sparse-ffn`.
  - This is the best Mac-side place to test whole-path graph fusion, but it is
    not a substitute for a hand-written Metal/CUDA kernel if gather/topk remains
    dominant.
  - Switching top-M selection from full sort to `argpartition` improved the
    `2048x8192x16` MLX prototype from ~`0.73x` dense speed to ~`0.84x` dense
    speed, still short of the estimated ~`26.5x` math speedup.
- MLX custom Metal backend:
  - Added as `benchmark-mlx-svd-sparse-ffn --backend metal`.
  - Current kernel fuses the whole forward but is serial per token. The next
    Metal/CUDA step is a parallel threadgroup design that splits candidate
    activation and down projection across lanes/warps.
- MLX hybrid parallel backend:
  - Added as `benchmark-mlx-svd-sparse-ffn --backend hybrid` or `--backend parallel`.
  - Uses MLX compiled ops for factor top-M candidate slots, then a custom Metal
    threadgroup-per-token kernel for exact candidate SwiGLU activation, duplicate
    suppression, norm rerank, and sparse down projection.
  - This is closer to the desired CUDA/Triton shape than the serial Metal
    kernel, but it still pays MLX selector materialization/topk overhead and
    lacks a fused blockwise selector over `d_ff`.
  - Current measured `2048x8192x16` result: dense `6.78ms`, hybrid sparse
    `7.37ms`, measured speedup `0.92x`, estimated math speedup `26.48x`,
    max diff vs MLX eager oracle `4.77e-7`.
- Interpretation:
  - The algorithm is viable.
  - The current implementation is a correctness prototype.
  - Real speed needs packed/fused Metal/CUDA/Triton kernels, not more
    architecture tuning.
