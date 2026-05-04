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


def test_recursive_exact_uses_fixed_recipe_schedule_per_pass() -> None:
    cfg = tiny_config("recursive_exact")
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = sample_batch(cfg)
    seen: list[tuple[int, torch.Tensor]] = []

    def record_step(
        h: torch.Tensor,
        h0: torch.Tensor,
        recipe_ids: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        del h0, active_mask
        seen.append((pass_idx, recipe_ids.detach().cpu().clone()))
        return h

    model.core.forward_step = record_step
    model.forward_exact(
        tokens,
        targets,
        fixed_recipe=1,
        fixed_recipe_schedule=[1, 2, 3, 1],
        fixed_depth=4,
    )
    assert [pass_idx for pass_idx, _ in seen] == [0, 1, 2, 3]
    assert [int(recipe_ids.unique().item()) for _, recipe_ids in seen] == [1, 2, 3, 1]


def test_recursive_exact_subset_can_reuse_cached_prelude() -> None:
    cfg = tiny_config("recursive_macro")
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = sample_batch(cfg)
    hot = model.forward_macro(tokens, targets, return_loss_per_sample=True, return_states=True)
    mask = torch.tensor([True, False])
    replay = model.forward_exact_subset(
        tokens,
        targets,
        mask,
        reuse_router_decisions=hot.meta.router,
        cached_h0=hot.meta.h0,
        return_loss_per_sample=True,
    )
    fresh = model.forward_exact_subset(
        tokens,
        targets,
        mask,
        reuse_router_decisions=hot.meta.router,
        return_loss_per_sample=True,
    )
    assert replay.loss_per_sample is not None and fresh.loss_per_sample is not None
    assert torch.allclose(replay.loss_per_sample, fresh.loss_per_sample, atol=1e-5)


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


def test_macro_shortlist_does_not_call_full_logits_path() -> None:
    cfg = tiny_config("recursive_macro_shortlist")
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, targets = sample_batch(cfg)

    def forbidden_full_logits(_h: torch.Tensor):
        raise AssertionError("shortlist path called full-vocab logits")

    model._coda_logits = forbidden_full_logits
    out = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        shortlist=True,
    )
    assert out.loss_per_sample is not None
    assert out.meta.logits is None
    assert out.meta.shortlist is not None
    assert out.logits is not None
    assert out.logits.shape[-1] == cfg.output.shortlist_max_tokens


def test_train_step_all_modes_cpu() -> None:
    for mode in ["dense_exact", "recursive_exact", "recursive_macro", "recursive_macro_shortlist"]:
        cfg = tiny_config(mode)
        engine = TrainEngine(cfg, device=torch.device("cpu"))
        try:
            result = engine.train_step(sample_batch(cfg))
            assert torch.isfinite(result.loss)
        finally:
            engine.close()
