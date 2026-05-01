from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

from recursive_training_engine.artifacts import build_manifest, summarize_metrics, write_json
from recursive_training_engine.config import ExperimentConfig, ModelConfig, load_config
from recursive_training_engine.ablations import build_ablation_configs
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.kernels import optimized, reference
from recursive_training_engine.macro import macro_distill_loss
from recursive_training_engine.metrics import (
    build_fairness_report,
    dense_param_count,
    hidden_cosine,
    logit_kl,
    recursive_param_count,
    solve_banks_for_fairness,
)
from recursive_training_engine.models import DenseModel, RecursiveModel, lm_loss_per_sample
from recursive_training_engine.training import TrainEngine
from recursive_training_engine.utils import default_device, set_seed


def _model_for_mode(config: ExperimentConfig, mode: str | None) -> ExperimentConfig:
    if mode is None:
        return config
    cfg = dataclasses.replace(config.training, mode=mode)
    topology = "dense" if mode == "dense_exact" else "recursive"
    model = dataclasses.replace(config.model, topology=topology)
    return dataclasses.replace(config, training=cfg, model=model)


def _with_run_dir(config: ExperimentConfig, run_dir: str | None) -> ExperimentConfig:
    if run_dir is None:
        return config
    path = Path(run_dir)
    return dataclasses.replace(
        config,
        output_dir=str(path.parent if str(path.parent) != "" else Path(".")),
        run_name=path.name,
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


TRAIN_MODES = [
    "dense_exact",
    "recursive_exact",
    "recursive_macro",
    "recursive_macro_shortlist",
    "recursive_macro_distill_only",
    "recursive_macro_lm_aligned",
    "recursive_macro_shadow_coda",
]


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
    config = _with_run_dir(_model_for_mode(load_config(args.config), args.mode), args.run_dir)
    print(json.dumps({"event": "resolved_config", "config": dataclasses.asdict(config)}, sort_keys=True))
    set_seed(config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    engine = TrainEngine(config)
    engine.write_run_manifest(
        {
            "data_fingerprint": streams.data_fingerprint,
            "tokenizer": streams.tokenizer_name,
            "projection_lane": config.data.vocab_projection,
            "vocab_size": config.model.vocab_size,
            "seq_len": config.training.seq_len,
            "batch_size": config.training.batch_size,
            "mode": config.training.mode,
            "semantic_depth": config.training.fixed_depth or config.model.t_max,
            "audit_mode": config.training.audit_mode,
            "audit_probability": config.training.audit_p_min
            if config.training.audit_p_min == config.training.audit_p_max
            else [config.training.audit_p_min, config.training.audit_p_max],
            "audit_cap": config.training.audit_cap,
            "audit_sampler": config.training.audit_sampler,
            "audit_fixed_count": config.training.audit_fixed_count_per_batch,
            "macro_checkpoint_source": args.resume,
            "teacher_checkpoint_source": None,
            "backend_status": optimized.backend_status(),
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
        save_checkpoint = args.save_checkpoint
        if save_checkpoint is None and args.run_dir is not None:
            save_checkpoint = str(engine.run_dir / "checkpoint.pt")
        if save_checkpoint:
            engine.save_checkpoint(save_checkpoint)
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
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"], strict=True)
            out = model(tokens, targets, return_loss_per_sample=True)
        else:
            model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"], strict=True)
            if config.training.mode == "recursive_exact":
                out = model.forward_exact(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=config.training.fixed_recipe,
                    fixed_depth=config.training.fixed_depth,
                )
            elif config.training.mode in {"recursive_macro", "recursive_macro_lm_aligned", "recursive_macro_distill_only"}:
                out = model.forward_macro(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=config.training.fixed_recipe,
                    fixed_depth=config.training.fixed_depth,
                )
            else:
                out = model.forward_macro(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    shortlist=True,
                    fixed_recipe=config.training.fixed_recipe,
                    fixed_depth=config.training.fixed_depth,
                )
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


def cmd_train_macro_teacher(args: argparse.Namespace) -> None:
    config = _with_run_dir(load_config(args.config), args.run_dir)
    fixed_depth = args.fixed_depth if args.fixed_depth is not None else config.training.fixed_depth
    fixed_recipe = args.fixed_recipe if args.fixed_recipe is not None else config.training.fixed_recipe
    if fixed_depth is None or fixed_recipe is None:
        raise SystemExit("train-macro-teacher requires fixed_recipe and fixed_depth")
    if args.teacher_checkpoint is None and not args.allow_random_teacher:
        raise SystemExit("train-macro-teacher requires --teacher-checkpoint")
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, topology="recursive"),
        training=dataclasses.replace(
            config.training,
            mode="recursive_macro_distill_only",
            fixed_depth=fixed_depth,
            fixed_recipe=fixed_recipe,
            audit_p_min=0.0,
            audit_p_max=0.0,
        ),
    )
    set_seed(config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    device = default_device()
    model = RecursiveModel(config.model, config.output).to(device)
    teacher_loaded = False
    if args.teacher_checkpoint is not None:
        payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        if payload.get("model") is None:
            raise SystemExit(f"teacher checkpoint has no model state: {args.teacher_checkpoint}")
        model.load_state_dict(payload["model"], strict=True)
        teacher_loaded = True
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.macro.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.macro.parameters(),
        lr=args.lr if args.lr is not None else config.training.effective_lr_macro,
        weight_decay=config.training.weight_decay,
    )
    batches = streams.train_batches(config.training)
    boundary_batches = None
    if args.boundary_cache:
        boundary_payload = torch.load(args.boundary_cache, map_location="cpu", weights_only=False)
        boundary_batches = boundary_payload.get("batches") or []
        if not boundary_batches:
            raise SystemExit(f"boundary cache is empty: {args.boundary_cache}")
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from recursive_training_engine.config import save_config
    from recursive_training_engine.reporting import JsonlLogger

    save_config(config, run_dir / "resolved_config.yaml")
    write_json(
        run_dir / "manifest.json",
        build_manifest(
            config,
            extra={
                "data_fingerprint": streams.data_fingerprint,
                "projection_lane": config.data.vocab_projection,
                "tokenizer": streams.tokenizer_name,
                "train_tokens": int(streams.train.numel()),
                "eval_tokens": int(streams.eval.numel()),
                "teacher_checkpoint_source": args.teacher_checkpoint,
                "macro_checkpoint_source": None,
                "teacher_loaded": teacher_loaded,
                "boundary_cache": args.boundary_cache,
            },
        ),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    rows = []
    try:
        for step in range(1, args.steps + 1):
            if boundary_batches is not None:
                entry = boundary_batches[(step - 1) % len(boundary_batches)]
                tokens = entry["tokens"].to(device)
                targets = entry["targets"].to(device)
                cached_endpoint = entry.get("recurrent_hidden")
                if cached_endpoint is None:
                    cached_endpoint = entry["states"][fixed_depth]
                exact_endpoint = cached_endpoint.to(device)
                with torch.no_grad():
                    _, exact_logits = model._coda_logits(exact_endpoint)
                    exact_loss_per_sample = lm_loss_per_sample(exact_logits, targets)
            else:
                tokens, targets = next(batches)
                tokens = tokens.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    exact = model.forward_exact(
                        tokens,
                        targets,
                        return_loss_per_sample=True,
                        fixed_recipe=fixed_recipe,
                        fixed_depth=fixed_depth,
                    )
                exact_endpoint = (
                    exact.meta.recurrent_hidden
                    if exact.meta.recurrent_hidden is not None
                    else exact.meta.hidden
                )
                exact_logits = exact.meta.logits
                exact_loss_per_sample = exact.loss_per_sample
            hot = model.forward_macro(
                tokens,
                targets,
                return_loss_per_sample=True,
                fixed_recipe=fixed_recipe,
                fixed_depth=fixed_depth,
            )
            hot_endpoint = (
                hot.meta.recurrent_hidden if hot.meta.recurrent_hidden is not None else hot.meta.hidden
            )
            if hot_endpoint is None or exact_endpoint is None:
                raise RuntimeError("macro teacher requires hidden states")
            if exact_loss_per_sample is None:
                raise RuntimeError("macro teacher requires exact per-sample loss")
            losses = macro_distill_loss(
                hot_endpoint,
                exact_endpoint,
                hot.meta.logits,
                exact_logits,
                lambda_hid=args.lambda_hid,
                lambda_cos=args.lambda_cos,
                lambda_kl=args.lambda_kl,
                lambda_norm=args.lambda_norm,
                temperature=args.temperature,
            )
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            macro_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.macro.parameters(),
                config.training.grad_clip_norm
                if config.training.grad_clip_norm is not None
                else float("inf"),
            )
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                with torch.no_grad():
                    refreshed_hot = model.forward_macro(
                        tokens,
                        targets,
                        return_loss_per_sample=True,
                        fixed_recipe=fixed_recipe,
                        fixed_depth=fixed_depth,
                    )
                    assert refreshed_hot.loss_per_sample is not None
                    refreshed_endpoint = (
                        refreshed_hot.meta.recurrent_hidden
                        if refreshed_hot.meta.recurrent_hidden is not None
                        else refreshed_hot.meta.hidden
                    )
                    cos = hidden_cosine(refreshed_endpoint, exact_endpoint).mean()
                    kl = (
                        logit_kl(exact_logits, refreshed_hot.meta.logits).mean()
                        if exact_logits is not None and refreshed_hot.meta.logits is not None
                        else torch.zeros((), device=device)
                    )
                    hidden_mse = (refreshed_endpoint - exact_endpoint).square().mean()
                    hot_nll = refreshed_hot.loss_per_sample.mean() / float(targets.shape[1])
                    exact_nll = exact_loss_per_sample.mean() / float(targets.shape[1])
                row = {
                    "event": "macro_teacher",
                    "mode": "recursive_macro_distill_only",
                    "step": step,
                    "fixed_recipe": fixed_recipe,
                    "fixed_depth": fixed_depth,
                    "teacher_checkpoint_source": args.teacher_checkpoint,
                    "boundary_cache": args.boundary_cache,
                    "coda_trainable": any(param.requires_grad for param in model.coda.parameters()),
                    "macro_lr": optimizer.param_groups[0]["lr"],
                    "coda_lr": 0.0,
                    "macro_grad_norm": float(macro_grad_norm.detach().float().cpu()),
                    "coda_grad_norm": 0.0,
                    "loss": float(loss.detach().float().cpu()),
                    "macro_distill_loss": float(loss.detach().float().cpu()),
                    "macro_hidden_loss": float(losses["hid"].detach().float().cpu()),
                    "macro_logit_kl_loss": float(losses["kl"].detach().float().cpu()),
                    "macro_norm_loss": float(losses["norm"].detach().float().cpu()),
                    "exact_eval_nll": float(exact_nll.detach().float().cpu()),
                    "hot_eval_nll": float(hot_nll.detach().float().cpu()),
                    "hot_exact_nll_gap": float((hot_nll - exact_nll).detach().float().cpu()),
                    "hidden_mse_exact_macro": float(hidden_mse.detach().float().cpu()),
                    "hidden_cosine_exact_macro": float(cos.detach().float().cpu()),
                    "logit_kl_exact_macro": float(kl.detach().float().cpu()),
                    **{
                        f"macro_{key}": float(value.detach().float().cpu())
                        for key, value in losses.items()
                    },
                }
                rows.append(row)
                logger.write(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        logger.close()
    save_checkpoint = args.save_checkpoint
    if save_checkpoint is None and args.run_dir is not None:
        save_checkpoint = str(run_dir / "checkpoint.pt")
    if save_checkpoint:
        path = Path(save_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rows": rows,
                "teacher_checkpoint_source": args.teacher_checkpoint,
            },
            path,
        )


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


def _metrics_rows(run: str | Path) -> list[dict[str, Any]]:
    path = Path(run)
    metrics = path if path.is_file() else path / "metrics.jsonl"
    return [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]


def cmd_summarize_alignment(args: argparse.Namespace) -> None:
    rows = _metrics_rows(args.run)
    numeric = [
        row
        for row in rows
        if any(key in row for key in ("exact_eval_nll", "eval_exact_nll_per_token", "hot_eval_nll"))
    ]
    if not numeric:
        raise SystemExit(f"no alignment metrics found in {args.run}")

    def values(*keys: str) -> list[float]:
        out = []
        for row in numeric:
            for key in keys:
                if isinstance(row.get(key), (int, float)):
                    out.append(float(row[key]))
                    break
        return out

    exact_vals = values("exact_eval_nll", "eval_exact_nll_per_token")
    hot_vals = values("hot_eval_nll", "eval_hot_nll_per_token")
    hidden_mse_vals = values("hidden_mse_exact_macro", "audit_hidden_mse_exact_macro")
    hidden_cos_vals = values("hidden_cosine_exact_macro", "audit_hidden_cosine_exact_macro")
    kl_vals = values("logit_kl_exact_macro", "audit_logit_kl_exact_macro")
    residual_vals = values("audit_residual_var", "audit_audit_residual_var")
    last = numeric[-1]
    summary = {
        "run": str(args.run),
        "best_exact_eval_nll": min(exact_vals) if exact_vals else None,
        "best_hot_eval_nll": min(hot_vals) if hot_vals else None,
        "final_exact_hot_gap": last.get("hot_exact_nll_gap"),
        "minimum_hidden_mse": min(hidden_mse_vals) if hidden_mse_vals else None,
        "maximum_hidden_cosine": max(hidden_cos_vals) if hidden_cos_vals else None,
        "minimum_logit_kl": min(kl_vals) if kl_vals else None,
        "audit_residual_trend": {
            "first_var": residual_vals[0] if residual_vals else None,
            "last_var": residual_vals[-1] if residual_vals else None,
        },
        "last_tokens_per_sec": last.get("tokens_per_sec"),
        "time_to_dense_target_nll": None,
    }
    _print_json(summary)


@torch.no_grad()
def cmd_collect_boundaries(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    batches = streams.train_batches(config.training)
    cache: dict[str, Any] = {
        "config": dataclasses.asdict(config),
        "checkpoint": args.checkpoint,
        "fixed_recipe": config.training.fixed_recipe,
        "fixed_depth": config.training.fixed_depth,
        "data_fingerprint": streams.data_fingerprint,
        "batches": [],
    }
    for _ in range(args.num_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        out = model.forward_exact(
            tokens,
            targets,
            return_states=True,
            return_loss_per_sample=True,
            fixed_recipe=config.training.fixed_recipe,
            fixed_depth=config.training.fixed_depth,
        )
        cache["batches"].append(
            {
                "tokens": tokens.detach().cpu(),
                "targets": targets.detach().cpu(),
                "h0": out.meta.h0.detach().cpu() if out.meta.h0 is not None else None,
                "states": {
                    depth: state.detach().cpu()
                    for depth, state in (out.meta.states or {}).items()
                },
                "recurrent_hidden": out.meta.recurrent_hidden.detach().cpu()
                if out.meta.recurrent_hidden is not None
                else None,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    _print_json({"output": str(output), "num_batches": args.num_batches})


@torch.no_grad()
def cmd_diagnose_coda_collusion(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    teacher = None
    if args.teacher_checkpoint:
        teacher = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
        teacher_payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher.load_state_dict(teacher_payload["model"], strict=True)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, targets = next(streams.eval_batches(config.training))
    tokens = tokens.to(device)
    targets = targets.to(device)
    exact = model.forward_exact(
        tokens,
        targets,
        return_loss_per_sample=True,
        fixed_recipe=config.training.fixed_recipe,
        fixed_depth=config.training.fixed_depth,
    )
    macro = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        fixed_recipe=config.training.fixed_recipe,
        fixed_depth=config.training.fixed_depth,
    )
    assert exact.loss_per_sample is not None and macro.loss_per_sample is not None
    exact_h = exact.meta.recurrent_hidden
    macro_h = macro.meta.recurrent_hidden
    if exact_h is None or macro_h is None:
        raise SystemExit("checkpoint did not expose recurrent hidden endpoints")
    teacher_exact_ce = None
    teacher_macro_ce = None
    if teacher is not None:
        _, teacher_exact_logits = teacher._coda_logits(exact_h)
        _, teacher_macro_logits = teacher._coda_logits(macro_h)
        from recursive_training_engine.models import lm_loss_per_sample

        teacher_exact_ce = float(
            (lm_loss_per_sample(teacher_exact_logits, targets).mean() / targets.shape[1]).cpu()
        )
        teacher_macro_ce = float(
            (lm_loss_per_sample(teacher_macro_logits, targets).mean() / targets.shape[1]).cpu()
        )
    _print_json(
        {
            "ce_coda_h_exact": float((exact.loss_per_sample.mean() / targets.shape[1]).cpu()),
            "ce_coda_h_macro": float((macro.loss_per_sample.mean() / targets.shape[1]).cpu()),
            "ce_teacher_coda_h_exact": teacher_exact_ce,
            "ce_teacher_coda_h_macro": teacher_macro_ce,
            "hidden_mse_exact_macro": float((exact_h - macro_h).square().mean().cpu()),
            "hidden_cosine_exact_macro": float(hidden_cosine(macro_h, exact_h).mean().cpu()),
            "logit_kl_exact_macro": float(logit_kl(exact.logits, macro.logits).mean().cpu())
            if exact.logits is not None and macro.logits is not None
            else None,
            "coda_collusion_suspected": bool(
                macro.loss_per_sample.mean() + 0.05 * targets.shape[1]
                < exact.loss_per_sample.mean()
            ),
        }
    )


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
    train.add_argument("--mode", choices=TRAIN_MODES)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--save-checkpoint")
    train.add_argument("--resume")
    train.add_argument("--run-dir")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=TRAIN_MODES)
    evaluate.add_argument("--checkpoint")
    evaluate.set_defaults(func=cmd_evaluate)

    bench = sub.add_parser("benchmark-kernels")
    bench.add_argument("--config", required=True)
    bench.add_argument("--iters", type=int, default=10)
    bench.set_defaults(func=cmd_benchmark_kernels)

    teacher = sub.add_parser("train-macro-teacher")
    teacher.add_argument("--config", required=True)
    teacher.add_argument("--steps", type=int, default=100)
    teacher.add_argument("--teacher-checkpoint")
    teacher.add_argument("--allow-random-teacher", action="store_true")
    teacher.add_argument("--boundary-cache")
    teacher.add_argument("--run-dir")
    teacher.add_argument("--fixed-recipe", type=int)
    teacher.add_argument("--fixed-depth", type=int)
    teacher.add_argument("--lr", type=float)
    teacher.add_argument("--lambda-hid", type=float, default=1.0)
    teacher.add_argument("--lambda-cos", type=float, default=0.5)
    teacher.add_argument("--lambda-kl", type=float, default=0.25)
    teacher.add_argument("--lambda-norm", type=float, default=0.05)
    teacher.add_argument("--temperature", type=float, default=2.0)
    teacher.add_argument("--log-every", type=int, default=10)
    teacher.add_argument("--save-checkpoint")
    teacher.set_defaults(func=cmd_train_macro_teacher)

    collect = sub.add_parser("collect-boundaries")
    collect.add_argument("--config", required=True)
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--num-batches", type=int, default=256)
    collect.set_defaults(func=cmd_collect_boundaries)

    diagnose = sub.add_parser("diagnose-coda-collusion")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--checkpoint", required=True)
    diagnose.add_argument("--teacher-checkpoint")
    diagnose.set_defaults(func=cmd_diagnose_coda_collusion)

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

    summarize_alignment = sub.add_parser("summarize-alignment")
    summarize_alignment.add_argument("run")
    summarize_alignment.set_defaults(func=cmd_summarize_alignment)

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
