from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.cli import main as cli_main
from recursive_training_engine.config import load_config, save_config
from recursive_training_engine.macro import MacroOperators
from recursive_training_engine.models import DenseModel, RecursiveModel
from recursive_training_engine.training import TrainEngine


def _cfg(tmp_path: Path | None = None, **training_overrides):
    cfg = load_config("configs/tiny.yaml")
    overrides = {
        "batch_size": 2,
        "seq_len": 8,
        "fixed_recipe": 1,
        "fixed_depth": 4,
        "audit_alpha": 0.0,
        "audit_beta": 0.0,
        "audit_gamma": 0.0,
        "lambda_hid": 0.0,
        "lambda_cos": 0.0,
        "lambda_kl": 0.0,
        "lambda_norm": 0.0,
        "lambda_cons": 0.0,
        "log_every": 1000,
    }
    overrides.update(training_overrides)
    return dataclasses.replace(
        cfg,
        output_dir=str(tmp_path) if tmp_path is not None else cfg.output_dir,
        run_name="proof-test",
        training=dataclasses.replace(cfg.training, **overrides),
    )


def _batch(cfg):
    tokens = torch.arange(cfg.training.batch_size * cfg.training.seq_len).view(
        cfg.training.batch_size,
        cfg.training.seq_len,
    ) % cfg.model.vocab_size
    return tokens, (tokens + 1) % cfg.model.vocab_size


def test_train_macro_teacher_requires_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="teacher-checkpoint"):
        cli_main(
            [
                "train-macro-teacher",
                "--config",
                "configs/tiny.yaml",
                "--run-dir",
                str(tmp_path / "macro"),
                "--fixed-recipe",
                "1",
                "--fixed-depth",
                "4",
                "--steps",
                "0",
            ]
        )


def test_train_macro_teacher_loads_checkpoint_weights(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    model = RecursiveModel(cfg.model, cfg.output)
    with torch.no_grad():
        model.embed.weight.add_(0.123)
    teacher = tmp_path / "teacher.pt"
    torch.save({"model": model.state_dict(), "optimizer": {}}, teacher)
    distilled = tmp_path / "distilled.pt"

    cli_main(
        [
            "train-macro-teacher",
            "--config",
            "configs/tiny.yaml",
            "--teacher-checkpoint",
            str(teacher),
            "--save-checkpoint",
            str(distilled),
            "--run-dir",
            str(tmp_path / "macro"),
            "--fixed-recipe",
            "1",
            "--fixed-depth",
            "4",
            "--steps",
            "0",
        ]
    )
    loaded = torch.load(distilled, map_location="cpu", weights_only=False)["model"]
    assert torch.allclose(loaded["embed.weight"], model.state_dict()["embed.weight"])


def test_transplant_and_operator_clone_cli_smoke(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        output_dir=str(tmp_path),
        run_name="clone-test",
        model=dataclasses.replace(
            cfg.model,
            topology="recursive",
            use_global_lowrank_corrector=True,
            global_corrector_rank=4,
        ),
        training=dataclasses.replace(
            cfg.training,
            batch_size=2,
            seq_len=8,
            fixed_recipe=1,
            fixed_recipe_schedule=[1, 2, 3, 1],
            fixed_depth=4,
            log_every=1000,
        ),
    )
    config_path = tmp_path / "tiny-recursive.yaml"
    save_config(cfg, config_path)
    dense = DenseModel(dataclasses.replace(cfg.model, topology="dense"))
    dense_checkpoint = tmp_path / "dense.pt"
    torch.save({"model": dense.state_dict()}, dense_checkpoint)
    init_checkpoint = tmp_path / "init.pt"

    cli_main(
        [
            "transplant-dense-to-recursive",
            "--dense-checkpoint",
            str(dense_checkpoint),
            "--recursive-config",
            str(config_path),
            "--output",
            str(init_checkpoint),
            "--use-global-lowrank-corrector",
            "--global-corrector-rank",
            "4",
        ]
    )
    loaded = torch.load(init_checkpoint, map_location="cpu", weights_only=False)["model"]
    assert torch.allclose(loaded["embed.weight"], dense.state_dict()["embed.weight"])

    cli_main(
        [
            "operator-clone",
            "--dense-checkpoint",
            str(dense_checkpoint),
            "--recursive-checkpoint",
            str(init_checkpoint),
            "--config",
            str(config_path),
            "--run-dir",
            str(tmp_path / "clone"),
            "--steps",
            "1",
            "--depth",
            "4",
            "--use-global-lowrank-corrector",
            "--global-corrector-rank",
            "4",
            "--log-every",
            "1000",
        ]
    )
    assert (tmp_path / "clone" / "checkpoint.pt").exists()


def test_macro_distill_freezes_coda(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, mode="recursive_macro_distill_only")
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        assert engine.recursive_model is not None
        assert all(not p.requires_grad for p in engine.recursive_model.coda.parameters())
        assert any(p.requires_grad for p in engine.recursive_model.macro.parameters())
    finally:
        engine.close()


def test_metric_only_audit_keeps_hot_loss_value() -> None:
    cfg = _cfg(
        mode="recursive_macro",
        audit_p_min=1.0,
        audit_p_max=1.0,
        audit_mode="metric_only",
        audit_gradient_correction=False,
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = _batch(cfg)
    hot = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        return_states=True,
        fixed_recipe=1,
        fixed_depth=4,
    )
    assert hot.loss_per_sample is not None
    result = AuditEngine(cfg.training).correction(model, tokens, targets, hot, seed=1)
    assert torch.allclose(result.corrected_loss_per_sample, hot.loss_per_sample)
    assert "corrected_metric_loss" in result.metrics
    assert result.metrics["audit_is_gradient_corrected"].item() == 0.0


def test_audit_reuses_prelude_when_cached_h0_available() -> None:
    cfg = _cfg(
        mode="recursive_macro",
        audit_p_min=1.0,
        audit_p_max=1.0,
        audit_mode="gradient_corrected",
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = _batch(cfg)
    hot = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        return_states=True,
        fixed_recipe=1,
        fixed_depth=4,
    )

    def forbidden_prelude(_tokens):
        raise AssertionError("audit replay recomputed prelude")

    model._prelude = forbidden_prelude
    result = AuditEngine(cfg.training).correction(model, tokens, targets, hot, seed=2)
    assert result.exact is not None


def test_fixed_route_depth_is_respected_and_router_aux_can_be_disabled(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        mode="recursive_exact",
        fixed_recipe=1,
        fixed_depth=4,
        disable_router_aux=True,
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        result = engine.train_step(_batch(cfg))
        out = result.model_output
        assert out.meta.recipe_ids is not None and out.meta.depths is not None
        assert torch.equal(out.meta.recipe_ids, torch.ones_like(out.meta.recipe_ids))
        assert torch.equal(out.meta.depths, torch.full_like(out.meta.depths, 4))
        assert result.metrics["aux_load"].item() == 0.0
        assert result.metrics["aux_cover"].item() == 0.0
        assert result.metrics["aux_depth"].item() == 0.0
    finally:
        engine.close()


def test_macro_initial_update_nonzero_but_small() -> None:
    cfg = load_config("configs/tiny.yaml")
    model_cfg = dataclasses.replace(
        cfg.model,
        macro_rank=4,
        macro_hidden_mult=2,
        macro_use_gated_update=True,
        macro_update_scale=0.05,
        macro_update_scale_init=0.1,
        macro_include_delta_to_h0=True,
        macro_use_depth_embedding=True,
    )
    macro = MacroOperators(model_cfg)
    h0 = torch.randn(3, 5, model_cfg.d_model)
    recipe_ids = torch.ones(3, dtype=torch.long)
    depths = torch.full((3,), 4, dtype=torch.long)
    out, _ = macro(h0, h0, recipe_ids, depths)
    delta_rms = (out - h0).float().pow(2).mean().sqrt()
    hidden_rms = h0.float().pow(2).mean().sqrt()
    assert delta_rms > 0
    assert delta_rms < 0.1 * hidden_rms
