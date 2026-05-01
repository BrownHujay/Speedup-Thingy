# Recursive Training Engine

This repository is a research harness for comparing a same-size dense
decoder-only Transformer against a recursive sparse macro-trained Transformer.
It is not yet proof of a 500x training speedup.

The implementation includes:

- `dense_exact`
- `recursive_exact`
- `recursive_macro`
- `recursive_macro_shortlist`
- fairness checks for stored parameter count, data stream, objective, optimizer,
  sequence length, and evaluation protocol
- exact audited recurrent execution with control-variate correction
- PyTorch reference kernels and CUDA/Triton dispatch points with explicit
  fallback reporting
- fused dense QKV, fused dense SwiGLU, fused banked QKV/up-gate storage,
  vectorized shortlist construction, fused/foreach AdamW switches, bf16/TF32
  presets, strict CUDA checks, forced FlashAttention dispatch, and
  `torch.compile` hot-path hooks
- unit, parity, regression, audit, and benchmark harnesses

Local Mac/CPU runs verify the PyTorch reference system. CUDA/Triton performance
benchmarks are skipped unless a CUDA GPU and Triton are available. Any run using
the PyTorch fallback reports that fallback mode in its manifest/backend status.

## Current Local Results And Audit Notes

These are measured numbers from the local Apple MPS machine. They are useful
for development sanity checks, not final CUDA claims. CUDA/Triton runs are still
required before claiming GPU speedups. The local backend is currently
`torch_reference_fallback`, so any speed result below is a PyTorch/MPS reference
result, not a proof of the intended optimized engine.

Important: some historical runs used different token projections. Do not compare
their NLLs as if they came from the same data stream.

- `filter`: keep GPT-2 token IDs below the configured small vocab size. This
  makes the local 2048-vocab proof stream easier and historically produced
  lower NLLs.
- `modulo`: map all GPT-2 token IDs into the configured small vocab by modulo.
  This keeps the requested token count but changes the distribution and produced
  higher NLLs.

The current clean comparison uses a full 6M-token TinyStories/GPT-2 stream,
batch size 128, sequence length 64, 5,005,312 train tokens per run, and 65,536
eval tokens. Recursive rows use fixed semantic depth 4, one macro physical pass,
and exact recurrent evaluation as the source-of-truth metric.

### Clean 5M-Token Matrix

Run artifact: `runs/clean_filter_vs_modulo_dense_vs_thingy_5m.jsonl`

Labels in the artifact may include the old ad-hoc word `thingy`; future labels
should use `recursive_macro_*`.

#### Filter Lane

Dataset fingerprint: `tinystories:gpt2_bpe:6000000`

| Mode | Train tokens/sec | Speedup vs dense | Exact eval NLL/token | Hot eval NLL/token | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense_exact` | 21,246 | 1.00x | 3.411 | n/a | Dense baseline, LR 0.001 |
| `recursive_macro` no audit | 77,225 | 3.63x | 4.012 | 3.158 | Fast, but exact/hot gap is large |
| `recursive_macro` fixed audit 1/128 | 45,399 | 2.14x | 3.588 | 3.372 | Faster than dense, closer exact loss, still worse exact NLL |

#### Modulo Lane

Dataset fingerprint: `tinystories:gpt2_bpe_mod2048:6000000`

| Mode | Train tokens/sec | Speedup vs dense | Exact eval NLL/token | Hot eval NLL/token | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `dense_exact` | 18,617 | 1.00x | 4.085 | n/a | Dense baseline, LR 0.001 |
| `recursive_macro` no audit | 68,004 | 3.65x | 4.518 | 4.245 | Fast, but exact/hot gap remains |
| `recursive_macro` fixed audit 1/128 | 35,135 | 1.89x | 4.298 | 4.479 | Exact loss improves vs no-audit, still worse than dense |

Current interpretation:

- Raw macro training is locally faster than dense on MPS/reference.
- Exact-path validation is still worse than dense.
- Fixed gradient-corrected audit reduces exact-path damage, but does not yet
  close the dense quality gap.
- Hot-path NLL can look much better than exact-path NLL. Hot-path NLL is not the
  proof metric.
- The main model failure is exact/hot divergence: the macro/coda path can learn
  a representation that decodes well on the hot path but does not match the
  exact recurrent path.

### Isolated Speed Diagnostic

Run context: filtered TinyStories/GPT-2 stream, batch 128, sequence 64, no eval,
20 warmup steps, 80 measured synchronized train steps.

| Mode | Median train tokens/sec | Mean train tokens/sec | Notes |
| --- | ---: | ---: | --- |
| `dense_exact` | 14,462 | 14,712 | MPS speed was noisy in this session |
| `recursive_macro` no audit | 103,076 | 97,511 | Raw hot path remains fast |
| `recursive_macro` fixed audit 1/128 | 47,894 | 46,430 | Audit tax is substantial but still faster than dense in this diagnostic |

This diagnostic confirms that the raw macro path was not globally nerfed. The
speed numbers vary significantly on Apple MPS, especially across long sequential
runs, so local speed should be treated as approximate.

### Historical No-Audit Hot-Path Training Speed

These numbers are kept for context only and should not be mixed with the clean
matrix above.

Config: `configs/medium_mac_real_speed.yaml`

Dataset: TinyStories with GPT-2 BPE, capped to a 2048-token vocab for this
local proof run. Batch size 4, sequence length 64, real token stream, 5 warmup
steps, 40 measured synchronized training steps. Audit probability was forced to
zero to measure the hot training path only.

| Mode | Tokens/sec | Speedup vs dense | Speedup vs recursive exact |
| --- | ---: | ---: | ---: |
| `dense_exact` | 3,710 | 1.00x | 2.37x |
| `recursive_exact` | 1,563 | 0.42x | 1.00x |
| `recursive_macro` | 10,974 | 2.96x | 7.02x |
| `recursive_macro_shortlist` | 9,692 | 2.61x | 6.20x |

The same no-audit macro path scales strongly with batch size on MPS. In the
larger-batch probe below, the table uses median synchronized step throughput
because MPS had occasional step-time jitter:

| Batch size | Dense tokens/sec | Macro tokens/sec | Macro speedup |
| ---: | ---: | ---: | ---: |
| 32 | 22,665 | 79,380 | 3.50x |
| 64 | 39,265 | 114,149 | 2.91x |
| 128 | 42,918 | 151,408 | 3.53x |
| 256 | 28,176 | 123,309 | 4.38x |

The best local macro throughput in this probe was batch 128, which is now saved
as `configs/medium_mac_real_hotpath.yaml`. Batch 256 still ran, but throughput
fell off, so it is not the current default.

### Historical 1M-Token Loss Snapshot

Config: `configs/medium_mac_real_1m.yaml`

This run uses real TinyStories/GPT-2 BPE data with a 1M-token training budget.
The recursive macro run uses exact audits at probability 0.25 with at most one
audited sample per batch. Evaluation for recursive models reports both the
exact recurrent path and the macro hot path.

| Train tokens | Dense exact eval NLL/token | Recursive macro exact eval NLL/token | Recursive macro hot eval NLL/token |
| ---: | ---: | ---: | ---: |
| 250,112 | 4.526 | 4.482 | 4.386 |
| 500,224 | 3.938 | 4.122 | 4.011 |
| 750,336 | 3.859 | 4.218 | 3.902 |
| 1,000,192 | 3.232 | 3.601 | 3.353 |

At 1M tokens, dense exact had the best exact-path validation loss in this Mac
run: 3.232 NLL/token in 265 seconds, about 3,773 tokens/sec since start. The
fixed audited recursive macro run was stable and converging, ending at 3.601
exact-path NLL/token and 3.353 hot-path NLL/token in 338 seconds, about 2,962
tokens/sec since start.

These historical 1M-token runs used older configs and should not be treated as
current proof. They are useful for regression hunting only.

## Recent Fixes For Auditability

Recent local changes made after audit feedback:

- `recursive_macro_shortlist` no longer computes full-vocab logits before the
  shortlist branch. The shortlist path now runs coda hidden only and keeps
  `meta.logits=None` for hot shortlist outputs.
- Added a regression test that fails if the shortlist path calls the full logits
  path.
- Audit sampling now has explicit modes:
  `metric_only`, `gradient_corrected`, and `distill_only`.
- Fixed audit-cap correction math. When Bernoulli sampling is capped, the
  correction now divides by the effective two-stage inclusion probability rather
  than the original Bernoulli probability.
- Added fixed-count audit sampling for exactly one audited sample per batch,
  used by the current 1/128 audit configs.
- Added `rte train-macro-teacher` for fixed-route macro endpoint distillation.
- Added `benchmarks/speed_audit.py --proof`, which refuses to report speed proof
  when the backend is using the reference fallback.

## Known Problems / Audit Targets

This repo is still a research harness, not a proven speed engine.

Known model and system problems:

- Exact/hot divergence remains the central model failure. No-audit macro can
  produce attractive hot NLL while exact recurrent eval is much worse.
- The router is still hard argmax for active recipe/depth execution. The
  straight-through one-hot tensors exist, but the main execution path does not
  yet use a differentiable mixture or policy-gradient training signal.
- Macro operators are still weak relative to the intended compiled recurrence.
  They need teacher pretraining against exact recurrent endpoints before being
  trusted as the main training path.
- Shortlist is now logically separated from full logits, but the local PyTorch
  shortlist implementation is not fast on MPS. It is correctness scaffolding
  until a fused CUDA/Triton shortlist kernel exists.
- Optimized kernels are still mostly dispatch wrappers over PyTorch reference
  implementations. The CUDA/Triton grouped kernels required for the real speed
  engine are not implemented yet.
- Active parameter-equivalent accounting exists, but the full budget ledger is
  incomplete. Prelude, coda, output, optimizer bytes, audit amortization, and
  kernel wall time all need to be reported together before any 500x claim.
- Apple MPS timings are noisy and should not be used as final proof.

Recommended next proof loop:

1. Use fixed routing first: `fixed_recipe=1`, `fixed_depth=4`.
2. Train exact recurrent teacher on the same stream.
3. Run `rte train-macro-teacher` to train `Phi_4` against exact endpoints.
4. Run macro LM training only after macro endpoint cosine/KL improves.
5. Compare exact-path validation NLL only.
6. Repeat for depth 8, then 16, before returning to learned routing.
7. Do not report speed proof unless `benchmarks/speed_audit.py --proof` passes
   on a non-reference CUDA/Triton backend.

## Current Status

Proven locally:

- no-audit macro hot-path training can run faster than the same-size dense
  baseline on Apple MPS/reference
- exact replay audits are stable after the NaN fix
- config validation, audit-cap sampling, gradient accumulation, active-budget
  assertions, run manifests, summaries, and checkpoint/resume are tested

Not proven yet:

- CUDA/Triton kernel speedups
- audited time-to-quality win over the dense baseline
- unbiased shortlist/full-softmax output-gradient correction at scale
- 500x training speedup over a strong same-size dense Transformer

The next proof target is: audited `recursive_macro` reaches the same exact-path
validation NLL as `dense_exact` faster on a controlled real-token experiment
while enforcing an active compute budget.

```bash
uv run pytest
uv run rte fairness --config configs/tiny.yaml
uv run rte train --config configs/tiny.yaml --steps 2
uv run rte train --config configs/tiny_mac_fused.yaml --steps 5
uv run rte train --config configs/medium_mac_real_hotpath.yaml --steps 12
uv run rte benchmark-kernels --config configs/tiny.yaml
uv run rte summarize-run runs/tiny
```
