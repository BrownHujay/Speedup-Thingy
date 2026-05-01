from __future__ import annotations

import argparse
import dataclasses
import json
import time

import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.training import TrainEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--target-loss", type=float, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dense_exact", "recursive_exact", "recursive_macro", "recursive_macro_shortlist"],
    )
    args = parser.parse_args()
    base = load_config(args.config)
    results = []
    for mode in args.modes:
        topology = "dense" if mode == "dense_exact" else "recursive"
        cfg = dataclasses.replace(
            base,
            model=dataclasses.replace(base.model, topology=topology),
            training=dataclasses.replace(base.training, mode=mode),
            run_name=f"{base.run_name}-ttq-{mode}",
        )
        streams = load_token_streams(cfg.data, cfg.training, cfg.model.vocab_size)
        batches = streams.train_batches(cfg.training)
        engine = TrainEngine(cfg)
        start = time.perf_counter()
        reached = False
        first_step = None
        try:
            for step in range(1, args.max_steps + 1):
                result = engine.train_step(next(batches))
                loss = float(result.loss.detach().float().cpu())
                if loss <= args.target_loss:
                    reached = True
                    first_step = step
                    break
        finally:
            engine.close()
        results.append(
            {
                "mode": mode,
                "target_loss": args.target_loss,
                "reached": reached,
                "first_step": first_step,
                "time_to_target_loss": time.perf_counter() - start if reached else None,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
