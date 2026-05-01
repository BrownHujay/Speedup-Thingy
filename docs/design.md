# Design Notes

## Objective And Data

Every mode trains the same next-token language-modeling objective over packed
`tokens -> targets` batches. The dataset loader emits deterministic token streams
for a fixed seed. The default local preset uses synthetic tokens for fast tests;
the small, medium, and large presets use TinyStories with GPT-2 BPE when the
tokenizer can be downloaded, and a trained byte-level BPE fallback when offline.

## Dense Baseline

`DenseModel` is a pre-norm decoder-only Transformer:

- tied embedding / LM head by default
- RMSNorm
- causal self-attention
- RoPE
- SwiGLU
- full dense residual blocks

Losses are returned per sample as sums over token positions so the training
engines can assemble exact control-variate corrections.

## Recursive Model

`RecursiveModel` implements:

- token embedding
- Prelude dense blocks
- shared banked recurrent core
- Coda dense blocks
- final RMSNorm
- full-vocab LM head

The exact recurrent path is the source of truth. Sparse recipes slice active
attention head groups and FFN slabs from selected banks. The dense fallback
recipe activates all banks and all groups.

## Macro Path And Audit Correction

The macro path applies `Phi_{r,k}` operators. The default hot path is `direct`
mode: one selected macro operator per sample. Optional `binary`, `greedy`, and
`consistency_tree` decomposition modes are available for experiments and report
their physical pass count and decomposition error. The hot loss is never trusted
alone in approximate audited modes.
`AuditEngine` samples exact replay subsets, reuses hot-path router decisions, and
assembles:

```text
hot + audit_mask / p_audit * (exact - hot)
```

When `audit_gradient_correction` is true, exact audited losses keep their graph
and the expression implements the stochastic gradient estimator. When it is
false, the residual is detached and the path is labeled as loss correction only.
Macro hidden, cosine, KL, and consistency losses are added when exact audit data
is available.

## Shortlist Output

`recursive_macro_shortlist` routes each hidden token to vocabulary clusters,
builds deduplicated per-token shortlists containing the target token at position
0, cluster candidates, hard negatives, and random negatives, and uses gathered
row-block logits for the hot path. Audited samples always use exact full-vocab
logits. The shortlist objective is approximate unless full-softmax gradient
correction is enabled and validated for the run.

## Kernels

Every required kernel has a public reference and dispatch symbol. The reference
module is pure PyTorch and is used by local CPU/MPS tests. The optimized module
is the dispatch surface for CUDA/Triton deployments and currently reports
whether it is using CUDA/Triton dispatch or PyTorch fallback. `strict_cuda`
raises instead of silently falling back. CUDA benchmark runs expose whether
Triton is present and skip CUDA-required tests on non-CUDA machines.

## Hot-Path Fusion

The production path avoids avoidable Python and launch overhead:

- dense attention uses one fused QKV projection
- dense SwiGLU uses one fused up/gate projection
- banked attention stores fused QKV banks and cached recipe column maps
- banked SwiGLU stores fused up/gate banks and cached slab maps
- shortlist construction is vectorized; the target is fixed at shortlist position 0
- active-touch accounting is a device-side table lookup
- CUDA runs use fused AdamW, TF32 matmuls, bf16 autocast presets, and optional
  `torch.compile` hot-path compilation
- `strict_cuda` speed presets reject CPU/MPS tensors in optimized kernels instead
  of silently falling back
- CUDA attention is dispatched through PyTorch SDPA with FlashAttention forced
  when `require_flash_attention` is enabled
- tests assert that the macro hot forward and shortlist builder contain no
  Python `for`/`while` loops
- active compute budgets can be enforced with
  `target_speedup_vs_dense`, `max_active_param_equiv_per_token`, and
  `max_hotpath_flops_per_token`
- every training run writes a resolved config, manifest, non-empty metrics log,
  and can save/resume checkpoints

The exact recurrent audit path remains a sequential recurrence by definition.
For speed runs it runs on CUDA tensors under the same strict kernel checks, while
the approximate macro hot path is fully batched.

## Failure Handling

The failure playbook from the build spec maps to configuration controls:

- macro drift: raise audit rate, reduce depth choices, increase macro losses
- router collapse: increase load/coverage penalties or force coverage batches
- sparse quality collapse: increase active groups or enable dense fallback
- grouped path slower than dense: reduce recipe count or increase batch size
- audit variance high: increase audit probability or cap residuals externally
- output head dominates: use shortlist mode and tune shortlist size
- exact subset OOMs: cap audit subset size or microbatch exact audits
