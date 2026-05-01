from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from recursive_training_engine.config import ExperimentConfig
from recursive_training_engine.kernels import optimized


def json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().float().cpu())
        return value.detach().cpu().tolist()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return str(value)


def config_hash(config: ExperimentConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_info(cwd: str | Path = ".") -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines() if dirty else [],
    }


def hardware_info() -> dict[str, Any]:
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    device = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": device,
        "cuda_device": cuda_name,
        "mps_available": mps_available,
    }


def build_manifest(
    config: ExperimentConfig,
    *,
    command: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "config_hash": config_hash(config),
        "config": dataclasses.asdict(config),
        "git": git_info(Path.cwd()),
        "hardware": hardware_info(),
        "backend": optimized.backend_status(),
        "command": command or sys.argv,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def summarize_metrics(path: str | Path) -> dict[str, Any]:
    run_path = Path(path)
    metrics_path = run_path if run_path.is_file() else run_path / "metrics.jsonl"
    rows = read_jsonl(metrics_path)
    if not rows:
        raise ValueError(f"no metrics found in {metrics_path}")
    numeric_keys: set[str] = set()
    for row in rows:
        numeric_keys.update(k for k, v in row.items() if isinstance(v, (int, float)))
    summary: dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "rows": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "last": rows[-1],
    }
    means = {}
    for key in sorted(numeric_keys):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            means[f"mean_{key}"] = sum(vals) / len(vals)
    summary["means"] = means
    manifest_path = metrics_path.parent / "manifest.json"
    if manifest_path.exists():
        summary["manifest"] = json.loads(manifest_path.read_text())
    return summary
