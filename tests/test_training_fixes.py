from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch
import yaml

from recursive_training_engine.config import load_config
from recursive_training_engine.training import TrainEngine


def _dense_cfg(tmp_path: Path, *, batch_size: int, grad_accum_steps: int, run_name: str):
    cfg = load_config("configs/tiny.yaml")
    return dataclasses.replace(
        cfg,
        run_name=run_name,
        output_dir=str(tmp_path),
        model=dataclasses.replace(cfg.model, topology="dense"),
        training=dataclasses.replace(
            cfg.training,
            mode="dense_exact",
            batch_size=batch_size,
            seq_len=8,
            grad_accum_steps=grad_accum_steps,
            grad_clip_norm=None,
            audit_p_min=0.0,
            audit_p_max=0.0,
            log_every=10_000,
        ),
    )


def _batch(batch_size: int, seq_len: int, vocab_size: int, *, offset: int = 0):
    tokens = (torch.arange(batch_size * seq_len).view(batch_size, seq_len) + offset) % vocab_size
    return tokens, (tokens + 1) % vocab_size


def _params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/tiny.yaml").read_text())
    raw["training"]["typo_learning_rate"] = 1.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="training.typo_learning_rate"):
        load_config(path)


def test_grad_accum_matches_one_large_batch(tmp_path: Path) -> None:
    full_cfg = _dense_cfg(tmp_path, batch_size=4, grad_accum_steps=1, run_name="full")
    accum_cfg = _dense_cfg(tmp_path, batch_size=2, grad_accum_steps=2, run_name="accum")
    full_batch = _batch(4, full_cfg.training.seq_len, full_cfg.model.vocab_size)
    micro_a = (full_batch[0][:2], full_batch[1][:2])
    micro_b = (full_batch[0][2:], full_batch[1][2:])

    full_engine = TrainEngine(full_cfg, device=torch.device("cpu"))
    accum_engine = TrainEngine(accum_cfg, device=torch.device("cpu"))
    try:
        full_engine.train_step(full_batch)
        accum_engine.train_step(micro_a)
        accum_engine.train_step(micro_b)
        assert full_engine.dense_model is not None and accum_engine.dense_model is not None
        assert torch.allclose(
            _params(full_engine.dense_model),
            _params(accum_engine.dense_model),
            atol=1e-6,
            rtol=1e-6,
        )
    finally:
        full_engine.close()
        accum_engine.close()


def test_active_budget_is_enforced(tmp_path: Path) -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        run_name="budget",
        output_dir=str(tmp_path),
        training=dataclasses.replace(
            cfg.training,
            mode="recursive_macro",
            batch_size=2,
            seq_len=8,
            audit_p_min=0.0,
            audit_p_max=0.0,
            max_active_param_equiv_per_token=1.0,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        with pytest.raises(RuntimeError, match="active parameter budget"):
            engine.train_step(_batch(2, cfg.training.seq_len, cfg.model.vocab_size))
    finally:
        engine.close()


def test_linear_lr_schedule_updates_by_tokens(tmp_path: Path) -> None:
    cfg = _dense_cfg(tmp_path, batch_size=2, grad_accum_steps=1, run_name="lr-schedule")
    tokens_per_step = cfg.training.batch_size * cfg.training.seq_len
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training,
            lr=0.01,
            lr_schedule="linear_decay_after",
            lr_decay_start_tokens=0,
            lr_decay_end_tokens=tokens_per_step,
            lr_final_scale=0.25,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        first = engine.train_step(_batch(2, cfg.training.seq_len, cfg.model.vocab_size))
        second = engine.train_step(_batch(2, cfg.training.seq_len, cfg.model.vocab_size, offset=3))
        assert first.metrics["lr"] == pytest.approx(0.01)
        assert second.metrics["lr"] == pytest.approx(0.0025)
    finally:
        engine.close()


def test_checkpoint_resume_matches_continuous(tmp_path: Path) -> None:
    cfg = _dense_cfg(tmp_path, batch_size=2, grad_accum_steps=1, run_name="continuous")
    batch1 = _batch(2, cfg.training.seq_len, cfg.model.vocab_size)
    batch2 = _batch(2, cfg.training.seq_len, cfg.model.vocab_size, offset=11)

    continuous = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        continuous.train_step(batch1)
        continuous.train_step(batch2)
        assert continuous.dense_model is not None
        continuous_params = _params(continuous.dense_model)
    finally:
        continuous.close()

    resumed_cfg = dataclasses.replace(cfg, run_name="resumed")
    checkpoint = tmp_path / "resume.pt"
    first = TrainEngine(resumed_cfg, device=torch.device("cpu"))
    try:
        first.train_step(batch1)
        first.save_checkpoint(checkpoint)
    finally:
        first.close()

    second = TrainEngine(resumed_cfg, device=torch.device("cpu"))
    try:
        second.load_checkpoint(checkpoint)
        second.train_step(batch2)
        assert second.dense_model is not None
        assert torch.allclose(continuous_params, _params(second.dense_model), atol=1e-6, rtol=1e-6)
    finally:
        second.close()
