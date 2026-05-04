from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.cli import main as cli_main
from recursive_training_engine.config import load_config, save_config
from recursive_training_engine.macro import (
    MacroOperators,
    apply_macro_rms_clamp,
    macro_alignment_metrics,
    macro_distill_loss,
    macro_rms_trust_penalty,
)
from recursive_training_engine.models import RecursiveModel
from recursive_training_engine.training import TrainEngine


def _v2_model_cfg(**overrides):
    cfg = load_config("configs/tiny.yaml")
    model_overrides = {
        "macro_type": "v2_delta_radius",
        "macro_rank": 4,
        "macro_hidden_mult": 2,
        "macro_update_scale": 0.001,
        "macro_update_scale_init": 0.25,
        "macro_radius_init_from_teacher": True,
        "macro_use_depth_embedding": True,
        "macro_use_recipe_embedding": True,
        "macro_use_delta_to_h0": True,
    }
    model_overrides.update(overrides)
    return dataclasses.replace(cfg.model, **model_overrides)


def test_macrov2_initial_radius_matches_teacher_delta_rms() -> None:
    model_cfg = _v2_model_cfg()
    macro = MacroOperators(model_cfg)
    macro.initialize_radius_from_teacher_delta(1, 4, torch.tensor(3.0))
    recipe_ids = torch.ones(2, dtype=torch.long)
    stride_ids = torch.full((2,), macro.stride_to_idx[4], dtype=torch.long)
    assert torch.allclose(macro.current_radius(recipe_ids, stride_ids), torch.full((2,), 3.0), atol=1e-5)


def test_macrov2_delta_not_limited_by_old_update_scale() -> None:
    model_cfg = _v2_model_cfg(macro_update_scale=0.001)
    macro = MacroOperators(model_cfg)
    macro.initialize_radius_from_teacher_delta(1, 4, torch.tensor(2.0))
    h0 = torch.randn(3, 5, model_cfg.d_model)
    out, _ = macro(h0, h0, torch.ones(3, dtype=torch.long), torch.full((3,), 4))
    delta_rms = (out - h0).float().pow(2).mean().sqrt()
    assert delta_rms > 1.0
    assert delta_rms > 100 * model_cfg.macro_update_scale


def test_macrov2_forward_shape_matches_old_macro() -> None:
    model_cfg = _v2_model_cfg()
    macro = MacroOperators(model_cfg)
    h0 = torch.randn(3, 7, model_cfg.d_model)
    out, trace = macro(h0, h0, torch.ones(3, dtype=torch.long), torch.full((3,), 4))
    assert out.shape == h0.shape
    assert trace.physical_passes.shape == (3,)


def test_macrov2_can_represent_large_teacher_delta() -> None:
    model_cfg = _v2_model_cfg(macro_radius_clamp_mult_max=4.0)
    macro = MacroOperators(model_cfg)
    macro.initialize_radius_from_teacher_delta(1, 4, torch.tensor(6.0))
    h0 = torch.randn(2, 6, model_cfg.d_model)
    out, _ = macro(h0, h0, torch.ones(2, dtype=torch.long), torch.full((2,), 4))
    assert (out - h0).float().pow(2).mean().sqrt() > 5.0


def test_macrov2_radius_clamp_applies_only_when_enabled() -> None:
    clamped_cfg = _v2_model_cfg(macro_radius_clamp_mult_max=2.0)
    clamped = MacroOperators(clamped_cfg)
    unclamped = MacroOperators(dataclasses.replace(clamped_cfg, macro_radius_init_from_teacher=False))
    for macro in (clamped, unclamped):
        macro.initialize_radius_from_teacher_delta(1, 4, torch.tensor(1.0))
        with torch.no_grad():
            macro.rho_base_raw[1, macro.stride_to_idx[4]].fill_(10.0)
    h0 = torch.randn(2, 5, clamped_cfg.d_model)
    recipe = torch.ones(2, dtype=torch.long)
    depth = torch.full((2,), 4)
    clamped_out, _ = clamped(h0, h0, recipe, depth)
    unclamped_out, _ = unclamped(h0, h0, recipe, depth)
    assert (clamped_out - h0).float().pow(2).mean().sqrt() <= 2.05
    assert (unclamped_out - h0).float().pow(2).mean().sqrt() > 5.0


def test_delta_rms_loss_catches_scale_mismatch() -> None:
    h0 = torch.zeros(2, 4, 3)
    exact = torch.ones_like(h0)
    hot = 10.0 * exact
    losses = macro_distill_loss(
        hot,
        exact,
        None,
        None,
        h0=h0,
        lambda_delta_dir=1.0,
        lambda_delta_rms=1.0,
    )
    assert losses["delta_dir"] < 1e-5
    assert losses["delta_rms"] > 1.0


def test_cosine_one_with_bad_scale_still_fails_rms_gate() -> None:
    h0 = torch.zeros(2, 4, 3)
    exact = torch.ones_like(h0)
    hot = 3.0 * exact
    metrics = macro_alignment_metrics(hot, exact, h0=h0)
    assert metrics["delta_cosine_exact_macro"] > 0.999
    assert float(metrics["delta_rms_ratio"]) == pytest.approx(3.0)


def test_distill_loss_reports_all_components() -> None:
    h0 = torch.zeros(2, 4, 3)
    exact = torch.ones_like(h0)
    hot = torch.zeros_like(h0).requires_grad_(True)
    losses = macro_distill_loss(
        hot,
        exact,
        None,
        None,
        h0=h0,
        lambda_delta_dir=1.0,
        lambda_delta_rms=1.0,
        lambda_endpoint_normed=1.0,
        lambda_endpoint_raw=1.0,
        lambda_macro_rms_trust=1.0,
    )
    for key in ("delta_dir", "delta_rms", "endpoint_normed", "endpoint_raw", "rms_trust", "kl"):
        assert key in losses


def test_delta_distill_loss_decreases_on_toy_target() -> None:
    h0 = torch.zeros(2, 4, 3)
    exact = torch.randn_like(h0)
    hot = torch.nn.Parameter(0.1 * torch.randn_like(h0))
    opt = torch.optim.Adam([hot], lr=0.05)
    first = None
    last = None
    for _ in range(40):
        losses = macro_distill_loss(
            hot,
            exact,
            None,
            None,
            h0=h0,
            lambda_delta_dir=1.0,
            lambda_delta_rms=1.0,
            lambda_endpoint_raw=1.0,
        )
        loss = sum(losses.values())
        first = float(loss.detach()) if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    assert last is not None and first is not None and last < first


def test_macro_rms_trust_penalty_positive_on_scale_mismatch() -> None:
    exact = torch.ones(2, 4, 3)
    hot = exact * 4.0
    assert macro_rms_trust_penalty(hot, exact) > 0.5


def test_macro_rms_clamp_prevents_large_norm_explosion() -> None:
    exact = torch.ones(2, 4, 3)
    hot = exact * 10.0
    clamped = apply_macro_rms_clamp(hot, exact, enabled=True, min_scale=0.5, max_scale=2.0)
    assert clamped.float().pow(2).mean().sqrt() < hot.float().pow(2).mean().sqrt()


def test_macro_rms_clamp_can_be_disabled() -> None:
    exact = torch.ones(2, 4, 3)
    hot = exact * 10.0
    unclamped = apply_macro_rms_clamp(hot, exact, enabled=False, min_scale=0.5, max_scale=2.0)
    assert torch.equal(unclamped, hot)


def _scheduled_audit(**training_overrides) -> AuditEngine:
    cfg = load_config("configs/tiny.yaml")
    training = dataclasses.replace(
        cfg.training,
        batch_size=8,
        audit_sampler="fixed_count",
        audit_fixed_count_per_batch=4,
        audit_schedule_enabled=True,
        audit_schedule_min_count=1,
        **training_overrides,
    )
    return AuditEngine(training)


def _good_schedule_metrics(**overrides):
    metrics = {
        "hidden_cosine_exact_macro": torch.tensor(0.99),
        "hot_exact_nll_gap": torch.tensor(0.01),
        "delta_rms_ratio": torch.tensor(1.0),
        "hidden_mse_exact_macro": torch.tensor(0.1),
        "macro_norm": torch.tensor(1.0),
        "exact_norm": torch.tensor(1.0),
        "audit_residual_var": torch.tensor(0.0),
    }
    metrics.update(overrides)
    return metrics


def test_audit_scheduler_refuses_to_lower_when_mse_huge() -> None:
    audit = _scheduled_audit()
    audit.maybe_update_fixed_count_schedule(
        _good_schedule_metrics(hidden_mse_exact_macro=torch.tensor(10.0)),
        batch_size=8,
    )
    assert audit.config.audit_fixed_count_per_batch == 8


def test_audit_scheduler_refuses_to_lower_when_norm_explodes() -> None:
    audit = _scheduled_audit()
    audit.maybe_update_fixed_count_schedule(
        _good_schedule_metrics(macro_norm=torch.tensor(5.0), exact_norm=torch.tensor(1.0)),
        batch_size=8,
    )
    assert audit.config.audit_fixed_count_per_batch == 8


def test_audit_scheduler_refuses_cosine_only_success() -> None:
    audit = _scheduled_audit()
    audit.maybe_update_fixed_count_schedule(
        {
            "hidden_cosine_exact_macro": torch.tensor(0.999),
            "hot_exact_nll_gap": torch.tensor(0.0),
            "audit_residual_var": torch.tensor(0.0),
        },
        batch_size=8,
    )
    assert audit.config.audit_fixed_count_per_batch == 8


def test_audit_scheduler_holds_when_delta_rms_ratio_bad() -> None:
    audit = _scheduled_audit()
    audit.maybe_update_fixed_count_schedule(
        _good_schedule_metrics(delta_rms_ratio=torch.tensor(1.8)),
        batch_size=8,
    )
    assert audit.config.audit_fixed_count_per_batch == 8


def test_phase_a_only_macro_trainable(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="phase-a",
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro_lm_aligned",
            batch_size=2,
            seq_len=8,
            log_every=10_000,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        assert engine.recursive_model is not None
        assert any(p.requires_grad for p in engine.recursive_model.macro.parameters())
        assert all(not p.requires_grad for p in engine.recursive_model.prelude.parameters())
        assert all(not p.requires_grad for p in engine.recursive_model.core.parameters())
        assert all(not p.requires_grad for p in engine.recursive_model.coda.parameters())
    finally:
        engine.close()


def test_phase_transition_requires_scale_metrics(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="phase-gate",
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro_lm_aligned",
            batch_size=2,
            seq_len=8,
            coda_warmup_steps=0,
            log_every=10_000,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        bad_scale = _good_schedule_metrics(delta_rms_ratio=torch.tensor(0.5))
        engine.global_step = 1
        engine._maybe_update_aligned_lm_phase(bad_scale)
        assert engine.aligned_lm_phase == "A"
        engine._set_coda_trainable_for_schedule()
        assert engine.recursive_model is not None
        assert all(not p.requires_grad for p in engine.recursive_model.coda.parameters())

        engine._maybe_update_aligned_lm_phase(_good_schedule_metrics())
        assert engine.aligned_lm_phase == "B"
        engine._set_coda_trainable_for_schedule()
        assert any(p.requires_grad for p in engine.recursive_model.coda.parameters())
    finally:
        engine.close()


def test_phase_b_regresses_to_macro_only_when_scale_breaks(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="phase-regress",
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro_lm_aligned",
            batch_size=2,
            seq_len=8,
            coda_warmup_steps=0,
            log_every=10_000,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        engine.global_step = 1
        engine.aligned_lm_phase = "B"
        engine._maybe_update_aligned_lm_phase(_good_schedule_metrics(delta_rms_ratio=torch.tensor(0.5)))
        assert engine.aligned_lm_phase == "A"
    finally:
        engine.close()


def test_aligned_lm_uses_frozen_teacher_checkpoint(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    teacher_model = RecursiveModel(cfg.model, cfg.output)
    with torch.no_grad():
        teacher_model.embed.weight.add_(0.5)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": teacher_model.state_dict(), "optimizer": {}}, checkpoint)
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="teacher",
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro_lm_aligned",
            aligned_lm_teacher_checkpoint=str(checkpoint),
            batch_size=2,
            seq_len=8,
            log_every=10_000,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        assert engine.teacher_model is not None
        assert torch.allclose(engine.teacher_model.embed.weight, teacher_model.embed.weight)
        assert all(not p.requires_grad for p in engine.teacher_model.parameters())
    finally:
        engine.close()


def test_aligned_lm_does_not_update_teacher(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    teacher_model = RecursiveModel(cfg.model, cfg.output)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": teacher_model.state_dict(), "optimizer": {}}, checkpoint)
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="teacher-stable",
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro_lm_aligned",
            aligned_lm_teacher_checkpoint=str(checkpoint),
            batch_size=2,
            seq_len=8,
            audit_p_min=1.0,
            audit_p_max=1.0,
            audit_fixed_count_per_batch=2,
            log_every=10_000,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        assert engine.teacher_model is not None
        before = {k: v.clone() for k, v in engine.teacher_model.state_dict().items()}
        tokens = torch.arange(16).view(2, 8) % cfg.model.vocab_size
        engine.train_step((tokens, (tokens + 1) % cfg.model.vocab_size))
        after = engine.teacher_model.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)
    finally:
        engine.close()


def test_diagnose_macro_range_reports_delta_capacity(tmp_path: Path, capsys) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="diag",
        training=dataclasses.replace(cfg.training, batch_size=2, seq_len=8, fixed_recipe=1, fixed_depth=4),
    )
    config_path = tmp_path / "tiny.yaml"
    save_config(cfg, config_path)
    model = RecursiveModel(cfg.model, cfg.output)
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}}, checkpoint)
    cli_main(
        [
            "diagnose-macro-range",
            "--config",
            str(config_path),
            "--teacher-checkpoint",
            str(checkpoint),
            "--fixed-depth",
            "4",
            "--fixed-recipe",
            "1",
            "--batches",
            "1",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert "macro_delta_capacity_estimate" in report
    assert "ratio_delta_exact_to_macro" in report


def test_macro_range_detects_bounded_delta_failure() -> None:
    model_cfg = dataclasses.replace(
        load_config("configs/tiny.yaml").model,
        macro_update_scale=0.001,
        macro_update_scale_init=1.0,
    )
    macro = MacroOperators(model_cfg)
    h0 = torch.zeros(2, 4, model_cfg.d_model)
    out, _ = macro(h0, h0, torch.ones(2, dtype=torch.long), torch.full((2,), 4))
    capacity = model_cfg.macro_update_scale * macro.update_scale[1, macro.stride_to_idx[4]].abs().max()
    exact_delta_rms = torch.tensor(1.0)
    assert exact_delta_rms / capacity > 100
    assert (out - h0).abs().max() <= capacity + 1e-6
