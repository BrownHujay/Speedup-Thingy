from __future__ import annotations

import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def greedy_decompose_depth(depth: int, strides: list[int]) -> list[int]:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    strides = sorted(strides, reverse=True)
    out: list[int] = []
    remaining = depth
    for stride in strides:
        while remaining >= stride:
            out.append(stride)
            remaining -= stride
    if remaining != 0:
        raise ValueError(f"cannot decompose depth={depth} with strides={strides}")
    return out


def expand_masked(values: torch.Tensor, mask: torch.Tensor, total: int) -> torch.Tensor:
    out = values.new_zeros((total, *values.shape[1:]))
    out[mask] = values
    return out


def detach_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu())
    return float(value)


def maybe_peak_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


@dataclass(slots=True)
class WallTimer:
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "WallTimer":
        synchronize_device()
        self.start = perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        synchronize_device()
        self.elapsed = perf_counter() - self.start


def batch_iterator(tokens: torch.Tensor, batch_size: int, seq_len: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("not enough tokens for requested sequence length")
    cursor = 0
    while True:
        xs = []
        ys = []
        for _ in range(batch_size):
            if cursor > max_start:
                cursor = 0
            chunk = tokens[cursor : cursor + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            cursor += seq_len
        yield torch.stack(xs), torch.stack(ys)


def ceiled_div(a: int, b: int) -> int:
    return int(math.ceil(a / b))
