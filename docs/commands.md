# Commands

```bash
uv run pytest
uv run rte fairness --config configs/tiny.yaml --strict
uv run rte train --config configs/tiny.yaml --mode dense_exact --steps 2
uv run rte train --config configs/tiny.yaml --mode recursive_exact --steps 2
uv run rte train --config configs/tiny.yaml --mode recursive_macro --steps 2
uv run rte train --config configs/tiny.yaml --mode recursive_macro_shortlist --steps 2
uv run rte train --config configs/tiny_mac_fused.yaml --steps 5
uv run rte evaluate --config configs/tiny.yaml --mode recursive_macro_shortlist
uv run rte benchmark-kernels --config configs/tiny.yaml
uv run python benchmarks/benchmark_kernels.py --config configs/tiny.yaml
uv run rte run-ablations --config configs/tiny.yaml
uv run rte summarize-run runs/tiny
uv run rte summarize-comparison runs/a runs/b
uv run python benchmarks/time_to_quality.py --config configs/tiny.yaml --target-loss 100 --max-steps 10
```

Speedup claims should be made from `time_to_quality.py` or `rte compare-ttq`
only after `rte fairness --strict` passes for the compared configs.

For CUDA speed runs, start from `configs/small.yaml`, `configs/medium.yaml`, or
`configs/large.yaml`; those presets enable bf16, TF32, fused AdamW, and compile
hooks, and `strict_cuda`. The local tiny config keeps strict CUDA disabled so CI
and smoke tests stay quick and deterministic on CPU/MPS.
