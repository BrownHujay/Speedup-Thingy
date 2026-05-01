from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.models import DenseModel, RecursiveModel
from recursive_training_engine.training import TrainEngine


def tensor_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


@torch.no_grad()
def evaluate(engine: TrainEngine, batches, mode: str, eval_batches: int) -> dict[str, float]:
    model = engine.dense_model if mode == "dense_exact" else engine.recursive_model
    assert model is not None
    tr = engine.config.training
    was_training = model.training
    model.eval()
    losses = []
    hot_losses = []
    tokens_seen = 0
    for _ in range(eval_batches):
        tokens, targets = engine._move_batch(next(batches))
        tokens_seen += int(tokens.numel())
        if mode == "dense_exact":
            out = model(tokens, targets, return_loss_per_sample=True)
            assert out.loss_per_sample is not None
            losses.append(out.loss_per_sample.sum())
        else:
            exact = model.forward_exact(
                tokens,
                targets,
                return_loss_per_sample=True,
                fixed_recipe=tr.fixed_recipe,
                fixed_depth=tr.fixed_depth,
            )
            assert exact.loss_per_sample is not None
            losses.append(exact.loss_per_sample.sum())
            hot = model.forward_macro(
                tokens,
                targets,
                return_loss_per_sample=True,
                shortlist=False,
                fixed_recipe=tr.fixed_recipe,
                fixed_depth=tr.fixed_depth,
            )
            assert hot.loss_per_sample is not None
            hot_losses.append(hot.loss_per_sample.sum())
    if was_training:
        model.train()
    result = {"eval_exact_nll_per_token": tensor_float(torch.stack(losses).sum()) / tokens_seen}
    if hot_losses:
        result["eval_hot_nll_per_token"] = tensor_float(torch.stack(hot_losses).sum()) / tokens_seen
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/medium_mac_real_speed.yaml")
    parser.add_argument("--modes", nargs="+", default=["dense_exact", "recursive_macro"])
    parser.add_argument("--token-budget", type=int, default=1_000_000)
    parser.add_argument("--eval-every-tokens", type=int, default=250_000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--output", default="runs/convergence_compare.jsonl")
    parser.add_argument("--audit-p-min", type=float)
    parser.add_argument("--audit-p-max", type=float)
    parser.add_argument("--audit-cap", type=int)
    args = parser.parse_args()

    base = load_config(args.config)
    streams = load_token_streams(base.data, base.training, base.model.vocab_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "data",
                "fingerprint": streams.data_fingerprint,
                "tokenizer": streams.tokenizer_name,
                "train_tokens": int(streams.train.numel()),
                "eval_tokens": int(streams.eval.numel()),
            }
        ),
        flush=True,
    )
    with output.open("w") as f:
        for mode in args.modes:
            topology = "dense" if mode == "dense_exact" else "recursive"
            training = dataclasses.replace(base.training, mode=mode)
            if args.audit_p_min is not None:
                training = dataclasses.replace(training, audit_p_min=args.audit_p_min)
            if args.audit_p_max is not None:
                training = dataclasses.replace(training, audit_p_max=args.audit_p_max)
            if args.audit_cap is not None:
                training = dataclasses.replace(training, audit_cap=args.audit_cap)
            cfg = dataclasses.replace(
                base,
                run_name=f"{base.run_name}-convergence-{mode}",
                model=dataclasses.replace(base.model, topology=topology),
                training=training,
            )
            engine = TrainEngine(cfg)
            train_batches = streams.train_batches(cfg.training)
            eval_batches = streams.eval_batches(cfg.training)
            tokens_per_step = cfg.training.batch_size * cfg.training.seq_len
            total_steps = math.ceil(args.token_budget / tokens_per_step)
            eval_every_steps = max(1, math.ceil(args.eval_every_tokens / tokens_per_step))
            start = time.perf_counter()
            rows = []
            try:
                for step in range(1, total_steps + 1):
                    result = engine.train_step(next(train_batches))
                    if step == 1 or step % eval_every_steps == 0 or step == total_steps:
                        elapsed = time.perf_counter() - start
                        metrics = {
                            k: (tensor_float(v) if isinstance(v, torch.Tensor) else v)
                            for k, v in result.metrics.items()
                        }
                        row = {
                            "event": "checkpoint",
                            "mode": mode,
                            "step": step,
                            "train_tokens": step * tokens_per_step,
                            "elapsed_sec": elapsed,
                            "train_nll_per_token": metrics.get("nll_per_token", tensor_float(result.loss)),
                            "tokens_per_sec_since_start": (step * tokens_per_step)
                            / max(elapsed, 1e-9),
                            **evaluate(engine, eval_batches, mode, args.eval_batches),
                            "last_step_tokens_per_sec": metrics.get("tokens_per_sec"),
                            "last_audit_rate": metrics.get("audit_audit_rate", 0.0),
                        }
                        rows.append(row)
                        line = json.dumps(row)
                        print(line, flush=True)
                        f.write(line + "\n")
                        f.flush()
            finally:
                engine.close()
            summary = {
                "event": "summary",
                "mode": mode,
                "checkpoints": rows,
            }
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
