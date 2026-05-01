from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap

import pytest
import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.macro import MacroOperators
from recursive_training_engine.output import ShortlistHead
from recursive_training_engine.training import TrainEngine


def _loop_nodes(fn) -> list[ast.AST]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]


def test_macro_hot_forward_has_no_python_loops() -> None:
    assert _loop_nodes(MacroOperators.forward) == []


def test_shortlist_builder_has_no_python_loops() -> None:
    assert _loop_nodes(ShortlistHead.build_shortlist) == []


def test_strict_cuda_rejects_non_cuda_device() -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, strict_cuda=True),
    )
    with pytest.raises(RuntimeError, match="strict_cuda requires a CUDA device"):
        TrainEngine(cfg, device=torch.device("cpu"))
