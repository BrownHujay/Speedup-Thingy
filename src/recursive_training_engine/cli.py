from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from recursive_training_engine.artifacts import summarize_metrics
from recursive_training_engine.config import ExperimentConfig, ModelConfig, load_config
from recursive_training_engine.ablations import build_ablation_configs
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.kernels import optimized, reference
from recursive_training_engine.metrics import (
    build_fairness_report,
    dense_param_count,
    recursive_param_count,
    solve_banks_for_fairness,
)
from recursive_training_engine.models import DenseModel, RecursiveModel
from recursive_training_engine.training import TrainEngine
from recursive_training_engine.utils import default_device, set_seed


def _model_for_mode(config: ExperimentConfig, mode: str | None) -> ExperimentConfig:
    if mode is None:
        return config
    cfg = dataclasses.replace(config.training, mode=mode)
    topology = "dense" if mode == "dense_exact" else "recursive"
    model = dataclasses.replace(config.model, topology=topology)
    return dataclasses.replace(config, training=cfg, model=model)


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def cmd_fairness(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dense_cfg = dataclasses.replace(config.model, topology="dense")
    rec_cfg = dataclasses.replace(config.model, topology="recursive")
    report = build_fairness_report(
        dense_cfg,
        rec_cfg,
        tolerance=config.model.fairness_tolerance,
    )
    solved = solve_banks_for_fairness(rec_cfg, max_banks=args.max_banks)
    _print_json(
        {
            **dataclasses.asdict(report),
            "passed": report.passed,
            "suggested_attn_banks": solved[0],
            "suggested_ffn_banks": solved[1],
            "suggested_relative_delta": solved[2],
        }
    )
    if args.strict and not report.passed:
        raise SystemExit(2)


def cmd_train(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), args.mode)
    print(json.dumps({"event": "resolved_config", "config": dataclasses.asdict(config)}, sort_keys=True))
    set_seed(config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    engine = TrainEngine(config)
    engine.write_run_manifest(
        {
            "data_fingerprint": streams.data_fingerprint,
            "tokenizer": streams.tokenizer_name,
            "train_tokens": int(streams.train.numel()),
            "eval_tokens": int(streams.eval.numel()),
            "command": sys.argv,
        }
    )
    if args.resume:
        engine.load_checkpoint(args.resume)
    batches = streams.train_batches(config.training)
    try:
        for step in range(args.steps):
            result = engine.train_step(next(batches))
            if step % config.training.log_every == 0:
                printable = {
                    key: (float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else value)
                    for key, value in result.metrics.items()
                }
                printable["step"] = step + 1
                print(json.dumps(printable, sort_keys=True))
        if args.save_checkpoint:
            engine.save_checkpoint(args.save_checkpoint)
    finally:
        engine.close()


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), args.mode)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, targets = next(streams.eval_batches(config.training))
    device = default_device()
    tokens = tokens.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        if config.training.mode == "dense_exact":
            model = DenseModel(dataclasses.replace(config.model, topology="dense")).to(device)
            out = model(tokens, targets, return_loss_per_sample=True)
        else:
            model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
            if config.training.mode == "recursive_exact":
                out = model.forward_exact(tokens, targets, return_loss_per_sample=True)
            elif config.training.mode == "recursive_macro":
                out = model.forward_macro(tokens, targets, return_loss_per_sample=True)
            else:
                out = model.forward_macro(tokens, targets, return_loss_per_sample=True, shortlist=True)
    assert out.loss_per_sample is not None
    _print_json({"mode": config.training.mode, "eval_loss_per_sample": float(out.loss_per_sample.mean().cpu())})


def cmd_benchmark_kernels(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    torch.manual_seed(config.training.seed)
    b, s, d = config.training.batch_size, config.training.seq_len, config.model.d_model
    x = torch.randn(b, s, d, device=device)
    w = torch.ones(d, device=device)
    q = torch.randn(b, config.model.n_heads, s, d // config.model.n_heads, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rows = []
    for name, fn, args_tuple in [
        ("k_fused_rmsnorm", optimized.k_fused_rmsnorm, (x, w)),
        ("k_flash_causal_dense", optimized.k_flash_causal_dense, (q, k, v)),
    ]:
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start is not None and end is not None:
            start.record()
            for _ in range(args.iters):
                fn(*args_tuple)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / args.iters
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn(*args_tuple)
            ms = (time.perf_counter() - t0) * 1000.0 / args.iters
        rows.append({"kernel": name, "device": str(device), "ms": ms})
    rows.append({"kernel": "triton_available", "value": optimized.triton_available()})
    _print_json(rows)


def cmd_run_ablations(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    results = []
    specs = build_ablation_configs(base)
    if args.max_ablations is not None:
        specs = specs[: args.max_ablations]
    for spec in specs:
        cfg = spec.config
        streams = load_token_streams(cfg.data, cfg.training, cfg.model.vocab_size)
        engine = TrainEngine(cfg)
        try:
            batches = streams.train_batches(cfg.training)
            result = None
            for _ in range(args.steps):
                result = engine.train_step(next(batches))
            assert result is not None
            results.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "mode": cfg.training.mode,
                    "loss": float(result.loss.detach().float().cpu()),
                    "stored_params": dense_param_count(cfg.model)
                    if cfg.training.mode == "dense_exact"
                    else recursive_param_count(cfg.model),
                }
            )
        finally:
            engine.close()
    _print_json(results)


def cmd_compare_ttq(args: argparse.Namespace) -> None:
    rows = []
    for path in Path(args.runs_dir).glob("*/metrics.jsonl"):
        mode_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        reached = [r for r in mode_rows if r.get("loss", float("inf")) <= args.target_loss]
        rows.append(
            {
                "run": path.parent.name,
                "target_loss": args.target_loss,
                "reached": bool(reached),
                "first_step": reached[0]["step"] if reached else None,
            }
        )
    _print_json(rows)


def cmd_summarize_run(args: argparse.Namespace) -> None:
    _print_json(summarize_metrics(args.run))


def cmd_summarize_comparison(args: argparse.Namespace) -> None:
    summaries = [summarize_metrics(path) for path in args.runs]
    rows = []
    for summary in summaries:
        last = summary["last"]
        rows.append(
            {
                "run": summary["metrics_path"],
                "rows": summary["rows"],
                "last_step": summary["last_step"],
                "mode": last.get("mode"),
                "nll_per_token": last.get("nll_per_token"),
                "tokens_per_sec": last.get("tokens_per_sec"),
                "active_param_equiv_per_token": last.get("active_param_equiv_per_token"),
                "audit_rate": last.get("audit_audit_rate"),
                "backend": summary.get("manifest", {}).get("backend", {}).get("mode"),
            }
        )
    _print_json(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rte")
    sub = parser.add_subparsers(required=True)
    fairness = sub.add_parser("fairness")
    fairness.add_argument("--config", required=True)
    fairness.add_argument("--max-banks", type=int, default=128)
    fairness.add_argument("--strict", action="store_true")
    fairness.set_defaults(func=cmd_fairness)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=["dense_exact", "recursive_exact", "recursive_macro", "recursive_macro_shortlist"])
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--save-checkpoint")
    train.add_argument("--resume")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=["dense_exact", "recursive_exact", "recursive_macro", "recursive_macro_shortlist"])
    evaluate.set_defaults(func=cmd_evaluate)

    bench = sub.add_parser("benchmark-kernels")
    bench.add_argument("--config", required=True)
    bench.add_argument("--iters", type=int, default=10)
    bench.set_defaults(func=cmd_benchmark_kernels)

    ablate = sub.add_parser("run-ablations")
    ablate.add_argument("--config", required=True)
    ablate.add_argument("--steps", type=int, default=1)
    ablate.add_argument("--max-ablations", type=int)
    ablate.set_defaults(func=cmd_run_ablations)

    ttq = sub.add_parser("compare-ttq")
    ttq.add_argument("--runs-dir", default="runs")
    ttq.add_argument("--target-loss", type=float, required=True)
    ttq.set_defaults(func=cmd_compare_ttq)

    summarize = sub.add_parser("summarize-run")
    summarize.add_argument("run")
    summarize.set_defaults(func=cmd_summarize_run)

    summarize_cmp = sub.add_parser("summarize-comparison")
    summarize_cmp.add_argument("runs", nargs="+")
    summarize_cmp.set_defaults(func=cmd_summarize_comparison)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
