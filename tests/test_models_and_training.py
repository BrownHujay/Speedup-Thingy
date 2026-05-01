from __future__ import annotations

import dataclasses

import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.models import DenseModel, RecursiveModel
from recursive_training_engine.output import ShortlistHead
from recursive_training_engine.training import TrainEngine
from recursive_training_engine.utils import set_seed


def tiny_config(mode: str = "recursive_macro_shortlist"):
    cfg = load_config("configs/tiny.yaml")
    topology = "dense" if mode == "dense_exact" else "recursive"
    return dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, topology=topology),
        training=dataclasses.replace(cfg.training, mode=mode, batch_size=2, seq_len=8),
    )


def sample_batch(cfg):
    tokens = torch.arange(cfg.training.batch_size * cfg.training.seq_len).view(
        cfg.training.batch_size, cfg.training.seq_len
    )
    tokens = tokens % cfg.model.vocab_size
    targets = (tokens + 1) % cfg.model.vocab_size
    return tokens, targets


def test_dense_forward_is_deterministic() -> None:
    cfg = tiny_config("dense_exact")
    set_seed(123)
    model_a = DenseModel(cfg.model)
    out_a = model_a(*sample_batch(cfg), return_loss_per_sample=True)
    set_seed(123)
    model_b = DenseModel(cfg.model)
    out_b = model_b(*sample_batch(cfg), return_loss_per_sample=True)
    assert out_a.loss_per_sample is not None and out_b.loss_per_sample is not None
    assert torch.allclose(out_a.loss_per_sample, out_b.loss_per_sample)


def test_recursive_exact_matches_manual_loop_with_fixed_route() -> None:
    cfg = tiny_config("recursive_exact")
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = sample_batch(cfg)
    out = model.forward_exact(
        tokens,
        targets,
        return_loss_per_sample=True,
        fixed_recipe=0,
        fixed_depth=2,
    )
    h0 = model._prelude(tokens)
    recipe_ids = torch.zeros(tokens.shape[0], dtype=torch.long)
    h = h0
    for _ in range(2):
        h = model.core.forward_step(h, h0, recipe_ids)
    hidden, logits = model._coda_logits(h)
    assert out.logits is not None
    assert torch.allclose(out.logits, logits, atol=1e-5)
    assert hidden.shape == h.shape


def test_recursive_macro_returns_per_sample_loss() -> None:
    cfg = tiny_config("recursive_macro")
    model = RecursiveModel(cfg.model, cfg.output)
    out = model.forward_macro(*sample_batch(cfg), return_loss_per_sample=True)
    assert out.loss_per_sample is not None
    assert out.loss_per_sample.shape == (cfg.training.batch_size,)


def test_shortlist_includes_targets() -> None:
    cfg = tiny_config("recursive_macro_shortlist")
    head = ShortlistHead(cfg.model.d_model, cfg.model.vocab_size, cfg.output)
    hidden = torch.randn(cfg.training.batch_size, cfg.training.seq_len, cfg.model.d_model)
    _, targets = sample_batch(cfg)
    shortlist, target_pos, _, duplicate_count = head.build_shortlist(hidden, targets, seed=1)
    gathered = shortlist.gather(-1, target_pos.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(gathered, targets)
    assert duplicate_count >= 0
    unique_counts = torch.tensor(
        [torch.unique(row).numel() for row in shortlist.reshape(-1, shortlist.shape[-1])]
    )
    assert torch.equal(unique_counts, torch.full_like(unique_counts, shortlist.shape[-1]))


def test_train_step_all_modes_cpu() -> None:
    for mode in ["dense_exact", "recursive_exact", "recursive_macro", "recursive_macro_shortlist"]:
        cfg = tiny_config(mode)
        engine = TrainEngine(cfg, device=torch.device("cpu"))
        try:
            result = engine.train_step(sample_batch(cfg))
            assert torch.isfinite(result.loss)
        finally:
            engine.close()
