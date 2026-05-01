from __future__ import annotations

import dataclasses

import torch

from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.config import load_config
from recursive_training_engine.models import RecursiveModel


def test_audit_correction_replayable_and_shapes() -> None:
    cfg = load_config("configs/tiny.yaml")
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training,
            audit_p_min=1.0,
            audit_p_max=1.0,
            batch_size=2,
            seq_len=8,
        ),
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens = torch.arange(cfg.training.batch_size * cfg.training.seq_len).view(
        cfg.training.batch_size, cfg.training.seq_len
    ) % cfg.model.vocab_size
    targets = (tokens + 1) % cfg.model.vocab_size
    hot = model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True)
    audit = AuditEngine(cfg.training)
    result = audit.correction(model, tokens, targets, hot, seed=42)
    assert result.audit_mask.all()
    assert result.corrected_loss_per_sample.shape == (cfg.training.batch_size,)
    assert result.exact is not None
    assert result.exact.loss_per_sample is not None
    expected = result.exact.loss_per_sample
    assert torch.allclose(result.corrected_loss_per_sample, expected, atol=1e-5)
