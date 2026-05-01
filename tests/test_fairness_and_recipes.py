from __future__ import annotations

import dataclasses
import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.metrics import (
    build_fairness_report,
    dense_param_count,
    recursive_param_count,
)
from recursive_training_engine.recipes import RecipeBank
from recursive_training_engine.utils import greedy_decompose_depth
from recursive_training_engine.models import RecursiveModel


def test_tiny_fairness_passes() -> None:
    cfg = load_config("configs/tiny.yaml").model
    dense = dataclasses.replace(cfg, topology="dense")
    rec = dataclasses.replace(cfg, topology="recursive")
    report = build_fairness_report(dense, rec, tolerance=cfg.fairness_tolerance)
    assert report.passed
    assert dense_param_count(dense) > 0
    assert recursive_param_count(rec) > 0


def test_tiny_mac_fused_fairness_passes() -> None:
    cfg = load_config("configs/tiny_mac_fused.yaml").model
    dense = dataclasses.replace(cfg, topology="dense")
    rec = dataclasses.replace(cfg, topology="recursive")
    report = build_fairness_report(dense, rec, tolerance=cfg.fairness_tolerance)
    assert report.passed


def test_recipe_bank_valid_and_balanced() -> None:
    cfg = load_config("configs/tiny.yaml").model
    bank = RecipeBank(cfg)
    fallback = bank.dense_fallback_recipe()
    assert fallback.dense_fallback
    assert set(fallback.head_groups) == set(range(cfg.head_groups))
    assert set(fallback.ffn_groups) == set(range(cfg.ffn_groups))
    for idx in range(1, cfg.recipe_count):
        spec = bank.get_recipe(idx)
        assert not isinstance(spec, list)
        assert len(spec.attention_banks) >= 1
        assert len(spec.ffn_banks) >= 1
        assert len(spec.head_groups) == cfg.active_head_groups
        assert len(spec.ffn_groups) == cfg.active_ffn_groups
    spread = bank.validate_balance()
    assert spread["attn_bank_spread"] <= 1


def test_usage_stats_update() -> None:
    cfg = load_config("configs/tiny.yaml").model
    bank = RecipeBank(cfg)
    bank.update_usage(torch.tensor([0, 1, 1, 2]), beta=0.0)
    stats = bank.usage_stats()
    assert stats["max_usage"] == 0.5


def test_depth_decomposition() -> None:
    assert greedy_decompose_depth(12, [1, 2, 4, 8]) == [8, 4]
    assert greedy_decompose_depth(4, [1, 2, 4]) == [4]


def test_binary_macro_decomposition_reports_extra_physical_passes() -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, macro_decomposition="binary"),
        training=dataclasses.replace(cfg.training, batch_size=2, seq_len=8),
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens = torch.arange(cfg.training.batch_size * cfg.training.seq_len).view(
        cfg.training.batch_size,
        cfg.training.seq_len,
    ) % cfg.model.vocab_size
    targets = (tokens + 1) % cfg.model.vocab_size
    out = model.forward_macro(
        tokens,
        targets,
        return_states=True,
        return_loss_per_sample=True,
        fixed_recipe=1,
        fixed_depth=4,
    )
    assert out.meta.macro_trace is not None
    assert torch.equal(out.meta.macro_trace.physical_passes, torch.full((2,), 2.0))
    assert out.meta.macro_trace.decomposition_error is not None

