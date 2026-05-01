from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any

import torch

from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.config import load_config
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.kernels import optimized
from recursive_training_engine.models import RecursiveModel
from recursive_training_engine.training import TrainEngine
from recursive_training_engine.utils import default_device, synchronize_device


def measure(name: str, fn: Callable[[], Any], *, iters: int, warmup: int, tokens: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    synchronize_device()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize_device()
    elapsed = time.perf_counter() - start
    seconds = elapsed / max(iters, 1)
    return {
        "component": name,
        "seconds": seconds,
        "ms": seconds * 1000.0,
        "tokens_per_sec": tokens / max(seconds, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium_mac_real_clamped4_8_cheap_audit.yaml")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--proof", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = optimized.backend_status()
    if args.proof and backend["uses_reference_fallback"]:
        raise SystemExit(
            "speed proof refused: optimized backend is using reference fallback "
            f"({backend['mode']})"
        )
    device = default_device()
    streams = load_token_streams(cfg.data, cfg.training, cfg.model.vocab_size)
    tokens_cpu, targets_cpu = next(streams.train_batches(cfg.training))
    tokens = tokens_cpu.to(device)
    targets = targets_cpu.to(device)
    batch_tokens = int(tokens.numel())
    subset_tokens = int(tokens.shape[1])

    model = RecursiveModel(cfg.model, cfg.output).to(device)
    model.eval()
    audit = AuditEngine(cfg.training)

    with torch.no_grad():
        h0 = model._prelude(tokens)
        route = model.router(tokens.new_zeros(h0.shape, dtype=h0.dtype), fixed_depth=cfg.model.depth_choices[0])
        route = model.router(h0)
        h_macro, _ = model.macro(h0, h0, route.recipe_id, route.depth, return_states=True)
        hidden = model._coda_hidden_impl(h_macro)
        hot_full = model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True)
        mask = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
        mask[0] = True

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        rows.append(measure("prelude", lambda: model._prelude(tokens), iters=args.iters, warmup=args.warmup, tokens=batch_tokens))
        rows.append(measure("router", lambda: model.router(h0), iters=args.iters, warmup=args.warmup, tokens=batch_tokens))
        rows.append(
            measure(
                "macro_phi",
                lambda: model.macro(h0, h0, route.recipe_id, route.depth, return_states=True),
                iters=args.iters,
                warmup=args.warmup,
                tokens=batch_tokens,
            )
        )
        rows.append(measure("coda_hidden", lambda: model._coda_hidden_impl(h_macro), iters=args.iters, warmup=args.warmup, tokens=batch_tokens))
        rows.append(measure("full_vocab_logits", lambda: hidden @ model.vocab_weight.t(), iters=args.iters, warmup=args.warmup, tokens=batch_tokens))
        rows.append(
            measure(
                "shortlist_loss",
                lambda: model.shortlist_head.loss(hidden, targets, model.vocab_weight, seed=cfg.training.seed),
                iters=args.iters,
                warmup=args.warmup,
                tokens=batch_tokens,
            )
        )
        rows.append(
            measure(
                "hot_forward_full",
                lambda: model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True),
                iters=args.iters,
                warmup=args.warmup,
                tokens=batch_tokens,
            )
        )
        rows.append(
            measure(
                "hot_forward_shortlist",
                lambda: model.forward_macro(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    return_states=True,
                    shortlist=True,
                    seed=cfg.training.seed,
                ),
                iters=args.iters,
                warmup=args.warmup,
                tokens=batch_tokens,
            )
        )
        rows.append(
            measure(
                "exact_subset_1_sequence",
                lambda: audit.run_exact_subset(model, tokens, targets, mask, hot_full.meta),
                iters=max(3, args.iters // 2),
                warmup=min(args.warmup, 2),
                tokens=subset_tokens,
            )
        )
        rows.append(
            measure(
                "exact_full_batch",
                lambda: model.forward_exact(tokens, targets, return_loss_per_sample=True),
                iters=max(2, args.iters // 4),
                warmup=min(args.warmup, 2),
                tokens=batch_tokens,
            )
        )

    no_audit_cfg = dataclasses.replace(
        cfg,
        run_name=f"{cfg.run_name}_speed_audit_no_audit",
        training=dataclasses.replace(
            cfg.training,
            audit_p_min=0.0,
            audit_p_max=0.0,
            audit_fixed_count_per_batch=0,
            lambda_hid=0.0,
            lambda_cos=0.0,
            lambda_kl=0.0,
            lambda_cons=0.0,
        ),
    )
    no_audit_engine = TrainEngine(no_audit_cfg, device=device)
    try:
        rows.append(
            measure(
                "train_step_no_audit_full",
                lambda: no_audit_engine.train_step((tokens_cpu, targets_cpu)),
                iters=max(3, args.iters // 2),
                warmup=min(args.warmup, 2),
                tokens=batch_tokens,
            )
        )
    finally:
        no_audit_engine.close()

    train_engine = TrainEngine(cfg, device=device)
    try:
        rows.append(
            measure(
                "train_step_fixed_audit_full",
                lambda: train_engine.train_step((tokens_cpu, targets_cpu)),
                iters=max(3, args.iters // 2),
                warmup=min(args.warmup, 2),
                tokens=batch_tokens,
            )
        )
    finally:
        train_engine.close()

    shortlist_cfg = dataclasses.replace(
        cfg,
        run_name=f"{cfg.run_name}_speed_audit_shortlist",
        training=dataclasses.replace(cfg.training, mode="recursive_macro_shortlist"),
        output=dataclasses.replace(cfg.output, mode="shortlist"),
    )
    shortlist_engine = TrainEngine(shortlist_cfg, device=device)
    try:
        rows.append(
            measure(
                "train_step_fixed_audit_shortlist",
                lambda: shortlist_engine.train_step((tokens_cpu, targets_cpu)),
                iters=max(3, args.iters // 2),
                warmup=min(args.warmup, 2),
                tokens=batch_tokens,
            )
        )
    finally:
        shortlist_engine.close()

    hot = next(row for row in rows if row["component"] == "hot_forward_full")
    for row in rows:
        row["relative_to_hot_forward_full"] = row["seconds"] / max(hot["seconds"], 1e-12)
    print(
        json.dumps(
            {"device": str(device), "backend": backend, "batch_tokens": batch_tokens, "rows": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
