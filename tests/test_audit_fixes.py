from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.config import load_config
from recursive_training_engine.models import RecursiveModel


def _recursive_cfg(**training_overrides):
    cfg = load_config("configs/tiny.yaml")
    base_overrides = {
        "batch_size": 4,
        "seq_len": 6,
        "lambda_hid": 0.0,
        "lambda_cos": 0.0,
        "lambda_kl": 0.0,
        "lambda_cons": 0.0,
        "audit_alpha": 0.0,
        "audit_beta": 0.0,
        "audit_gamma": 0.0,
    }
    base_overrides.update(training_overrides)
    return dataclasses.replace(
        cfg,
        training=dataclasses.replace(cfg.training, **base_overrides),
    )


def _batch(cfg):
    tokens = torch.arange(cfg.training.batch_size * cfg.training.seq_len).view(
        cfg.training.batch_size,
        cfg.training.seq_len,
    ) % cfg.model.vocab_size
    return tokens, (tokens + 1) % cfg.model.vocab_size


def test_audit_cap_random_subsample_is_position_balanced() -> None:
    cfg = _recursive_cfg(audit_p_min=1.0, audit_p_max=1.0, audit_cap=2)
    audit = AuditEngine(cfg.training)
    p = torch.ones(8)
    counts = torch.zeros_like(p)
    for seed in range(400):
        counts += audit.sample_mask(p, seed=seed).float()
    assert counts.max() - counts.min() < 60
    assert torch.allclose(counts.mean(), torch.tensor(100.0), atol=10.0)


def test_audit_cap_reports_two_stage_inclusion_probability() -> None:
    cfg = _recursive_cfg(audit_p_min=1.0, audit_p_max=1.0, audit_cap=2)
    audit = AuditEngine(cfg.training)
    p = torch.ones(8)
    mask, inclusion_p = audit.sample_mask_with_prob(p, seed=123)
    assert int(mask.sum()) == 2
    assert torch.allclose(inclusion_p[mask], torch.full((2,), 0.25))
    assert audit.last_capped_fraction == 0.75


def test_fixed_count_audit_sampler_has_exact_count_and_correct_probability() -> None:
    cfg = _recursive_cfg(
        batch_size=8,
        audit_p_min=0.125,
        audit_p_max=0.125,
        audit_cap=None,
        audit_fixed_count_per_batch=1,
    )
    audit = AuditEngine(cfg.training)
    p = torch.full((8,), 0.125)
    counts = torch.zeros_like(p)
    for seed in range(800):
        mask, inclusion_p = audit.sample_mask_with_prob(p, seed=seed)
        assert int(mask.sum()) == 1
        assert torch.allclose(inclusion_p, torch.full_like(p, 0.125))
        counts += mask.float()
    assert counts.max() - counts.min() < 45
    assert torch.allclose(counts.mean(), torch.tensor(100.0), atol=1e-6)


def test_distill_only_audit_keeps_hot_loss_value() -> None:
    cfg = _recursive_cfg(
        audit_p_min=1.0,
        audit_p_max=1.0,
        audit_mode="distill_only",
        audit_gradient_correction=False,
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = _batch(cfg)
    hot = model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True)
    assert hot.loss_per_sample is not None
    audit = AuditEngine(cfg.training)
    result = audit.correction(model, tokens, targets, hot, seed=3)
    assert torch.allclose(result.corrected_loss_per_sample, hot.loss_per_sample)
    assert result.metrics["audit_is_gradient_corrected"].item() == 0.0


def test_coverage_deficit_increases_audit_probability() -> None:
    cfg = _recursive_cfg(
        audit_p_min=0.1,
        audit_p_max=0.9,
        audit_gamma=0.5,
        coverage_min=0.2,
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = _batch(cfg)
    hot = model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True)
    audit = AuditEngine(cfg.training)
    cold_p = audit.compute_audit_prob(hot.meta)
    audit.recipe_audit_ema = torch.full((cfg.model.recipe_count,), cfg.training.coverage_min)
    covered_p = audit.compute_audit_prob(hot.meta)
    assert cold_p.mean() > covered_p.mean()


def test_audit_gradient_estimator_tracks_exact_gradient() -> None:
    cfg = _recursive_cfg(
        audit_p_min=0.5,
        audit_p_max=0.5,
        audit_cap=None,
        audit_gradient_correction=True,
    )
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = _batch(cfg)
    param = model.final_norm.weight

    hot_for_route = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        return_states=True,
        fixed_recipe=1,
        fixed_depth=1,
    )
    exact = model.forward_exact(
        tokens,
        targets,
        return_loss_per_sample=True,
        router_decisions=hot_for_route.meta.router,
    )
    assert exact.loss_per_sample is not None
    exact_loss = exact.loss_per_sample.mean() / cfg.training.seq_len
    exact_grad = torch.autograd.grad(exact_loss, param, retain_graph=False)[0].detach()

    grads = []
    for seed in range(120):
        hot = model.forward_macro(
            tokens,
            targets,
            return_loss_per_sample=True,
            return_states=True,
            fixed_recipe=1,
            fixed_depth=1,
        )
        audit = AuditEngine(cfg.training)
        result = audit.correction(model, tokens, targets, hot, seed=seed)
        loss = result.corrected_loss_per_sample.mean() / cfg.training.seq_len
        grads.append(torch.autograd.grad(loss, param, retain_graph=False)[0].detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    cosine = F.cosine_similarity(avg_grad.flatten(), exact_grad.flatten(), dim=0)
    assert cosine > 0.95
    assert torch.allclose(avg_grad.norm(), exact_grad.norm(), rtol=0.25, atol=0.25)
