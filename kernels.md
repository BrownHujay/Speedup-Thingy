# Kernel Notes

This file tracks fused/kernel-shaped paths in the project. It records the
actual runtime shape, measured bottleneck, and the CUDA/Triton substitution for
places where the local MPS/PyTorch path is only a prototype.

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
It is a PyTorch implementation whose stages map directly onto the CUDA/Triton
kernel boundaries below.

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

### Cluster-Shared Candidate Pool Oracle

The per-token sparse FFN path is now treated as a correctness/quality oracle,
not the production kernel shape. On T4 it lost badly because every token gets a
different sparse row set, which turns the FFN into `topk + gather + tiny
irregular reductions` while dense FFN remains a few Tensor-Core GEMMs.

The new no-training diagnostic is:

```bash
uv run rte deferred-neuron-cluster-pool-oracle \
  --config configs/proof_filter_depth4_dense.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --clusters 8 16 32 64 \
  --candidate-m 128 192 256 \
  --ranks 64
```

Current implementation is a PyTorch oracle in `cli.py`:

```text
SVD selector features q_up/q_gate
→ cosine k-means token clusters
→ aggregate factor scores per cluster
→ one shared candidate pool per cluster
→ X_cluster @ W_up[:, pool]
→ X_cluster @ W_gate[:, pool]
→ SwiGLU
→ activation @ W_down[pool, :]
```

This executes all `M` candidates for a cluster and intentionally skips
per-token final `k`. That is the point: the hot shape becomes a small set of
cluster GEMMs instead of token-specific sparse gathers. The oracle reports NLL,
KL, hidden cosine, cluster imbalance, candidate recall against reference
top-`k`, and the ideal FFN FLOP ratio `M / d_ff`.

CUDA/Triton kernel target:

| Kernel | Role | Shape |
| --- | --- | --- |
| `k_cluster_assign` | Assign tokens to a small number of clusters from SVD selector features. | One block per token or tiled GEMM/argmax over cluster centers. |
| `k_cluster_score_reduce` | Aggregate up/gate/product scores into one pool score per cluster/neuron. | Reduction over tokens grouped by cluster; can be approximate/top-heavy. |
| `k_cluster_pool_gemm` | Execute `X_cluster @ W_up_pool`, `X_cluster @ W_gate_pool`, and `Z @ W_down_pool`. | Grouped GEMM over clusters, the GPU-friendly replacement for per-token row gather. |
| `k_cluster_scatter` | Write clustered FFN outputs back to original token order. | Lightweight permutation/scatter; keep below GEMM time. |

The production-shaped CUDA prototype now lives in
`src/recursive_training_engine/kernels/cluster_pool_ffn.py` and can be
benchmarked with:

```bash
uv run rte benchmark-cluster-pool-ffn \
  --size 2048x8192x4096 \
  --clusters 8 16 32 \
  --candidate-m 96 128 192 \
  --rank 64
```

This path uses static cluster centers/candidate pools and removes the Python
per-cluster loop:

```text
cuBLAS q_up/q_gate route features
→ fixed-center assignment
→ Triton slot assignment
→ Triton token pack
→ cuBLAS batched GEMM for up/gate/down pooled FFN
→ Triton gather back to token order
```

Implemented kernels:

| Kernel | Role | Shape |
| --- | --- | --- |
| `_assign_slots_kernel` | Uses atomic per-cluster counters to give each token a padded slot. | Grid: `[tokens]`; no Python loop and no `one_hot/cumsum` packing. |
| `_pack_x_kernel` | Copies token rows into `[clusters, max_tokens_per_cluster, d_model]`. | Grid: `[tokens, ceil(d_model / block_d)]`. |
| `_gather_y_kernel` | Copies padded cluster outputs back to `[tokens, d_model]`. | Grid: `[tokens, ceil(d_model / block_d)]`. |

The pooled FFN math deliberately uses cuBLAS `torch.bmm` rather than hand-written
Triton matmul because this is the GEMM-shaped part we want NVIDIA libraries to
run on Tensor Cores. The missing CUDA follow-up, if this still does not hit the
target on a real GPU, is a single persistent kernel or CUDA Graph wrapper that
combines slot assignment, pack, three grouped GEMMs, and gather with less launch
overhead. The current code is the first non-notebook implementation of the
right execution shape.

Static/precomputed pool quality is now tested with:

```bash
uv run rte deferred-neuron-static-cluster-pool-oracle \
  --config configs/proof_filter_depth4_dense.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --calibration-tokens 8192 32768 65536 \
  --clusters 8 16 \
  --candidate-m 192 \
  --ranks 64
```

This command moves candidate-pool planning out of the hot path:

```text
offline/train-stream calibration:
  dense hidden states per layer
  → SVD selector features
  → fixed cluster centroids
  → fixed candidate pools per layer/cluster

held-out eval/hot shape:
  token hidden
  → cheap route to fixed centroid
  → use precomputed pool
  → cluster-pool FFN
```

The first 65,536-token Mac eval showed that global `C=1` pools are not viable,
but static `C=16, M=192` pools hold quality:

```text
dense:                         3.087743
dynamic C16 M192:              3.103140
static C16 M192, calib 8k:     3.107310
static C16 M192, calib 32k:    3.107661
static C16 M192, calib 65k:    3.107783
```

CUDA target: fixed/slow-refresh codebook execution with cheap routing and
packed cluster GEMMs. Dynamic pool planning stays out of the hot path.

Training-side diagnostics now exist:

```bash
uv run rte benchmark-static-cluster-pool-ffn-train \
  --size 2048x8192x1024 \
  --clusters 16 \
  --candidate-m 192

uv run rte static-cluster-pool-staleness \
  --config configs/proof_filter_depth4_dense.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --calibration-tokens 8192 \
  --clusters 16 \
  --candidate-m 192 \
  --perturb-pct 0 0.1 0.5 1 2 5
```

`benchmark-static-cluster-pool-ffn-train` uses the differentiable static
pack/GEMM/gather path:

```text
precomputed assignments
→ precomputed pack/gather indices
→ index_select pack
→ batched up/gate/down GEMMs
→ index_select gather
→ backward
→ gradient scatter from cluster pools back to dense FFN rows
```

The full-pool gradient check verifies output and gradients match dense exactly
when `M = d_ff`. On Mac/MPS smoke sizes, forward/backward is faster than dense,
but gradient scatter is currently a major tax. The CUDA follow-up is therefore
not another math change; it is a fused/persistent gradient scatter-add kernel or
a training parameterization that treats cluster pools as the trainable weights
and refreshes/merges periodically.

The first staleness smoke on 16k eval tokens showed stale pools barely moved
under synthetic FFN perturbations up to 5% RMS noise:

```text
perturb  stale NLL   refreshed NLL   stale recall
0.0%     2.97147     2.97147         0.96489
0.5%     2.97127     2.97119         0.96487
1.0%     2.97158     2.97208         0.96489
2.0%     2.97153     2.97280         0.96485
5.0%     2.97204     2.97387         0.96480
```

This is a short smoke, not the final training answer, but it says the fixed
codebook is not immediately fragile.

Static-pool continuation diagnostics now exist too:

```bash
uv run rte static-cluster-pool-continuation \
  --config configs/proof_filter_depth4_exact_teacher.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --steps 50 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --resume-optimizer-state \
  --eval-batches 8 \
  --calibration-tokens 8192 \
  --clusters 16 \
  --candidate-m 192 \
  --refresh-intervals 0 50
```

This command starts from the dense checkpoint, replaces only the FFNs with
`StaticClusterPoolSwiGLU`, keeps dense attention/head intact, and trains the
sparse FFN path through the original dense FFN row weights. Candidate pools are
static unless a positive refresh interval is requested. A bounded 20-step Mac
run on 8,192 eval tokens produced:

```text
variant                 initial NLL   final NLL   train tokens
dense continuation       3.00750       3.03713     40,960
static C16 M192          3.02818       3.04981     40,960
static C16 M192 refresh  3.02818       3.05132     40,960
```

The short run did not show refresh helping; that is consistent with the
staleness smoke. The important CUDA-side training issue remains the same:
forward/backward through packed cluster GEMMs is fine, but packing/gather and
duplicate-row gradient handling need fused kernels or a pool-native optimizer.

The dense checkpoint includes AdamW optimizer state, so the cleaner
continuation setup resumes it and uses a lower LR with no decay. A 50-step
65,536-token eval run produced:

```text
variant                 initial NLL   final NLL   gap vs dense
dense continuation       3.08774       3.00059     -
static C16 M192          3.10731       3.01401     +0.01342
static C16 M192 refresh  3.10731       3.01362     +0.01303
```

The initial sparse gap was `+0.01957`, so the gap shrank during the clean
continuation run. This does not prove long-run training yet, but it clears the
first trainability gate.

There is also a local gradient-alignment diagnostic:

```bash
uv run rte static-cluster-pool-gradient-alignment \
  --config configs/proof_filter_depth4_exact_teacher.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --batch-size 32 \
  --eval-batches 1 \
  --calibration-tokens 8192 \
  --clusters 16 \
  --candidate-m 192
```

For `C16 M192`, the one-batch mean results were:

```text
FFN output cosine:       0.99712
FFN input-grad cosine:   0.98696
W_up row-grad cosine:    0.97558
W_gate row-grad cosine:  0.97826
W_down row-grad cosine:  0.99049
```

So the sparse FFN path is not merely forward-close; it sends fairly aligned
gradients to the FFN input and selected dense rows.

The 300-step stability run used the same clean setup:

```bash
uv run rte static-cluster-pool-continuation \
  --config configs/proof_filter_depth4_exact_teacher.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --steps 300 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --resume-optimizer-state \
  --eval-batches 8 \
  --eval-steps 0 50 100 200 300 \
  --calibration-tokens 8192 \
  --clusters 16 \
  --candidate-m 192 \
  --refresh-intervals 0 100 50
```

Result:

```text
variant         step 0    step 50   step 100  step 200  step 300  final gap
dense           3.08774   3.00059   3.01334   2.97915   2.95914   -
static          3.10731   3.01401   3.02470   2.98917   2.96843   +0.00929
refresh@100     3.10731   3.01401   3.02554   2.99107   2.96947   +0.01033
refresh@50      3.10731   3.01362   3.02600   2.99001   2.96960   +0.01046
```

The initial sparse gap was `+0.01957`, so all three sparse variants passed the
300-step gap gate and no-refresh was slightly best. Coverage stayed nearly
complete:

```text
unique selected neurons/layer: ~255.8 / 256
coverage fraction:             ~0.9993
dead selected rows/layer:       ~0.18-0.36
```

Input-gradient alignment remained stable:

```text
initial input-grad cosine: ~0.9868-0.9870
final input-grad cosine:   ~0.9855-0.9860
```

Current MPS forward/backward benchmark:

```text
shape                 dense fwd+bwd   sparse fwd+bwd   +grad scatter   speedup fwd+bwd   speedup with scatter
d=512,H=2048,N=1024   8.76 ms         1.95 ms          4.24 ms         4.50x             2.06x
d=2048,H=8192,N=1024  116.30 ms       7.86 ms          22.64 ms        14.80x            5.14x
```

The `d=2048,H=8192` row passes the first training-speed gate (`>=5x`) even
including the current unfused gradient scatter. The CUDA path for this variant
is fused scatter-add or pool-native optimizer state; otherwise the GEMM body is
fast but row updates dominate.

Notebook benchmark variants:

| Variant | Current behavior | What it measures |
| --- | --- | --- |
| `cluster_execute_prepacked_bmm` | Uses already packed `[clusters, max_tokens, d]` inputs and pooled weights, then runs three `torch.bmm` calls. | Lower bound for the cluster GEMM body once packing/routing are solved. This hit the desired 20-50x range in the Colab synthetic benchmark. |
| `cluster_execute_pack_bmm_scatter` | Packs tokens with `one_hot/cumsum/scatter`, runs the same `bmm` body, then gathers outputs back. | Generic PyTorch dynamic packing overhead. This is much better than the Python loop but still far from the lower bound. |
| `cluster_execute_preindexed_pack_bmm_gather` | Uses precomputed static pack indices and flat gather indices, then runs `index_select + bmm + index_select`. | Best PyTorch-side proxy for a custom pack/gather kernel. It separates the remaining pack/scatter cost from the GEMM body without using a Python loop. |

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

The exact candidate activation path is also not the default benchmark path
anymore. Recomputing exact `x·W_up_j` and `x·W_gate_j` for dynamic per-token
candidate rows is too irregular and lost badly on T4. The fast path uses the
SVD factor activations from the selector GEMMs as runtime activations
(`triton_exact_activation=False`) and only applies the sparse down projection
over selected neurons. Exact candidate activation remains available for
correctness/reference checks via `triton_exact_activation=True`.

### Per-Token SVD Sparse FFN Kernel Map

This path is retained as a correctness/reference implementation. T4 timing
showed that per-token dynamic candidate rows are memory/latency bound even when
individual Triton pieces are fused. The CUDA lesson from this path is: keep
cuBLAS for low-rank selector GEMMs and avoid per-token unique row sets in the
training hot path.

| Stage | Existing implementation | CUDA mapping |
| --- | --- | --- |
| Factor selector | cuBLAS query/score GEMMs plus `torch.topk`. | Keep GEMM-based; do not replace with scalar Triton dot loops. |
| Candidate activation | Triton candidate kernels or factor-activation shortcut. | Use only for oracle/reference or small candidate diagnostics. |
| Candidate rerank | Norm/top-k over slots. | Fused with candidate activation when the per-token path is used. |
| Sparse down projection | Triton `_downsum_kernel`. | Parallel over output channels; still row-gather heavy. |

The training path moved away from this shape toward cluster pools and active
union because those recover large GEMMs.

## Current Quality/Speed Status

- Quality gate passed:
  - `rank48, m64, k64`: hot NLL matches oracle (`3.1077236`).
  - `rank64, m64, product64, k64`: hot NLL matches oracle (`3.1059841`).
- Speed gate failed in PyTorch:
  - `2048x8192`: estimated FFN speedup ~`26.5x`, measured MPS prototype
    speedup ~`0.49x`.
- MLX graph backend:
  - Added as optional `metal` extra and `benchmark-mlx-svd-sparse-ffn`.
  - Mac-side whole-path graph fusion still exposes gather/topk as the dominant
    bottleneck.
  - Switching top-M selection from full sort to `argpartition` improved the
    `2048x8192x16` MLX prototype from ~`0.73x` dense speed to ~`0.84x` dense
    speed, still short of the estimated ~`26.5x` math speedup.
- MLX custom Metal backend:
  - Added as `benchmark-mlx-svd-sparse-ffn --backend metal`.
  - Current kernel fuses the whole forward but is serial per token.
  - CUDA/Metal mapping: parallel threadgroup design splitting candidate
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
  - The measured bottleneck is implementation shape, not architecture quality.
  - Packed/fused Metal/CUDA/Triton kernels are the speed path; more architecture
    tuning is not the answer for this specific slowdown.

## Static Cluster-Pool Union FFN

The `ActiveUnionSwiGLU` path tests the simplification after static
cluster-pool training: build the normal `C16 M192` codebook, then use the
unique union of all candidate neurons in each layer as one global active FFN.

This is **not** the previously failed single global top-M pool. The active set
is the union of all neurons selected by the cluster pools:

```text
cluster codebook: C pools × M neurons
active union:     unique(cluster_candidate_ids)
hot forward:      X @ W_up[active].T, X @ W_gate[active].T, act @ W_down[active]
```

### Local MPS Smoke Results

Command:

```bash
rte static-cluster-pool-union-eval \
  --config configs/proof_filter_depth4_dense.yaml \
  --dense-checkpoint runs/proof_d4_dense/checkpoint.pt \
  --eval-batches 1 \
  --calibration-tokens 8192 \
  --rank 64 \
  --clusters 16 \
  --candidate-m 192
```

Results on the small proof model (`d_ff=256`, so the union is nearly dense):

```text
dense:          3.007499
static cluster: 3.031404
global union:   3.007487
union active:   255.7 / 256 neurons
```

This validates the union execution math, but it is not a scale-quality proof
because the small proof model's union covers almost the whole FFN.

### Active Union Forward/Backward Probe

Command:

```bash
rte benchmark-active-union-ffn-train \
  --size 512x2048x512 \
  --size 2048x8192x128 \
  --active-m 320
```

MPS results:

```text
512x2048, active 320:
  indexed master fwd+bwd: 2.84x dense
  packed active fwd+bwd:  5.19x dense

2048x8192, active 320:
  indexed master fwd:     9.25x dense
  indexed master fwd+bwd: 1.67x dense
  packed active fwd+bwd:  15.24x dense
```

Interpretation:

- The active-union forward shape is GPU-friendly.
- Backward through indexed master weights is still update/scatter-bound.
- Packed active trainable weights are dramatically faster than indexed master
  rows and are the active-union training shape used by
  `--sparse-ffn-kind active_union_packed`.
- A tiny all-neuron active-union MPS check matched dense forward/backward to
  numerical tolerance (`~1e-8` forward, `~1e-9` grads).

### Fair Benchmark Baseline

Dense FFN timing must use the same fused SwiGLU projection shape as the model:

```text
ug = X @ W_ug.T
up, gate = split(ug)
Y = (up * silu(gate)) @ W_down
```

The active-union benchmark now reports against this fused dense baseline, not
against two separate dense `W_up`/`W_gate` launches. Sparse variants report:

```text
indexed fused:      gather active rows from dense W_ug every step
packed split:       train W_up_active/W_gate_active separately
packed fused WUG:   train contiguous W_ug_active directly
```

Use `packed fused WUG` for the active-union training speed path. The indexed
version is a diagnostic for the scatter/gather tax.

The executable `ActiveUnionSwiGLU` path now also uses one fused active `W_ug`
projection for up/gate rows instead of two separate active GEMMs. For the
trainable no-reconcile path, `PackedActiveUnionSwiGLU` stores:

```text
W_ug_active:   [2M, d]
W_down_active: [d, M]
```

This is the FFN-v2 systems shape: no dense-master row gather and no per-step
reconcile in the hot path. Any reconcile/foldback to dense master weights is a
separate refresh/audit operation, not part of the default training step.

### Capped Union Quality Gate

`static-cluster-pool-union-eval` can now evaluate capped active unions:

```bash
rte static-cluster-pool-union-eval ... --union-caps 160 192 256 320
```

Capping is a quality-changing experiment. It ranks candidate ids only from the
static cluster codebook frequency/rank, not from eval labels, but it still
changes which neurons the sparse FFN can use. Keep it behind the notebook flag
`RUN_UNION_CAP_EVAL` so the uncapped union path remains the clean baseline.

The same command also supports one layer-specific active-size variant:

```bash
rte static-cluster-pool-union-eval ... --union-layer-caps 192 192 256 256 320 320
```

This tests the "do not pay max active size in every layer" idea. It is also a
quality-changing experiment and must remain opt-in.

### Active-Union CUDA Mapping

The active-union path is intentionally GEMM-shaped:

```text
forward:
  ug = X @ W_ug_active.T
  up, gate = split(ug)
  z = up * silu(gate)
  y = z @ W_down_active.T

backward:
  dW_down_active = dY.T @ z
  dZ = dY @ W_down_active
  dUp, dGate = swiglu_backward(dZ, up, gate)
  dW_ug_active = concat(dUp, dGate).T @ X
  dX = concat(dUp, dGate) @ W_ug_active
```

`src/recursive_training_engine/kernels/active_swiglu_triton.py` implements the
packed SwiGLU activation as a custom Triton autograd path. The GEMM pieces stay
as normal matmuls, while Triton fuses:

```text
forward:  z = up * silu(gate)
backward: dUp, dGate = swiglu_backward(dZ, up, gate)
```

`benchmark-active-union-ffn-train --triton-swiglu-backward` applies that custom
path to both dense fused `W_ug` and sparse active-union `W_ug_active`, so the
generic SwiGLU backward optimization is not sparse-only. Whole-model graphing or
`torch.compile` is a fairness-sensitive runtime setting: apply it to both dense
and sparse model shells, while packed active rows remain the sparse-specific
FFN primitive.

### CUDA Graph Benchmark Path

The FFN training benchmarks now have CUDA graph replay enabled by default:

```bash
rte benchmark-static-cluster-pool-ffn-train ... --cuda-graphs
rte benchmark-active-union-ffn-train ... --cuda-graphs
```

Graph capture is applied symmetrically:

```text
dense fused W_ug forward+backward graph
sparse static-cluster forward+backward graph
sparse active-union indexed forward+backward graph
sparse active-union packed W_ug forward+backward graph
```

The notebook exposes this as `RUN_CUDA_GRAPHS=True` and reports the dense graph
time next to the sparse graph time, so graph launch reduction cannot silently
favor only the sparse path.

### Full-Model Active-Union Benchmark Mapping

`benchmark-active-union-model-train-step` measures the full train step with the
same packed SwiGLU treatment on both sides:

```text
dense baseline:
  DenseSwiGLU -> PackedDenseSwiGLU
  W_ug:       [2H, D]
  W_down:    [H, D] row-oriented parameter

sparse FFN:
  DenseSwiGLU -> PackedActiveUnionSwiGLU
  W_ug_active:    [2M, D]
  W_down_active:  [M, D] row-oriented parameter
```

When `--triton-swiglu-backward` is enabled, both packed dense FFNs and packed
active-union FFNs call `triton_packed_swiglu_ffn`. The sparse-specific change is
only the active row count `M`; the dense baseline still gets the same fused
SwiGLU activation/backward helper and row-oriented down projection layout.
