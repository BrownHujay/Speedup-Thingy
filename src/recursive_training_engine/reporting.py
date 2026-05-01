from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a")
        self.rows_written = 0

    def write(self, row: dict[str, Any]) -> None:
        clean = {}
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                clean[key] = float(value.detach().float().mean().cpu())
            else:
                clean[key] = value
        self.file.write(json.dumps(clean, sort_keys=True) + "\n")
        self.file.flush()
        self.rows_written += 1

    def close(self) -> None:
        self.file.close()


def summarize_time_to_target(rows: list[dict[str, Any]], target_loss: float) -> dict[str, Any]:
    reached = [row for row in rows if row.get("val_loss", float("inf")) <= target_loss]
    if not reached:
        return {"target_loss": target_loss, "reached": False, "time_to_target_loss": None}
    first = min(reached, key=lambda x: x.get("elapsed_train_time", float("inf")))
    return {
        "target_loss": target_loss,
        "reached": True,
        "time_to_target_loss": first.get("elapsed_train_time"),
        "step": first.get("step"),
    }
