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

## Current Local Results

These are the current measured numbers from the local Apple MPS machine on
April 30, 2026. They are useful for development sanity checks, not final CUDA
claims. CUDA/Triton runs are still required before claiming GPU speedups.

### No-Audit Hot-Path Training Speed

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

### 1M-Token Loss Snapshot

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

Current interpretation: the no-audit macro path is genuinely faster locally,
especially versus the full recursive exact loop. The audited convergence path is
now stable after the NaN fix, but this small Mac run does not yet show a
time-to-quality win over dense exact. Larger CUDA/Triton runs, bigger vocabularies,
and better audit/distillation scheduling are the next places to look for a real
end-to-end training speedup.

## Current Status

Proven locally:

- no-audit macro hot-path training can run faster than the same-size dense
  baseline on Apple MPS
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
