from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.cli import (
    _build_svd_factor_cache,
    _ffn_neuron_svd_cluster_pool_output,
)
from recursive_training_engine.layers import DenseSwiGLU, SVDFactorSparseFFN
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


def test_svd_factor_sparse_ffn_recovers_dense_when_all_neurons_active() -> None:
    cfg = tiny_config("dense_exact")
    mlp = DenseSwiGLU(cfg.model)
    sparse = SVDFactorSparseFFN.from_dense(
        mlp,
        rank=min(cfg.model.d_model, cfg.model.d_ff),
        top_k=cfg.model.d_ff,
        up_m=cfg.model.d_ff,
    )
    x = torch.randn(2, 5, cfg.model.d_model)
    dense_out = mlp(x)
    sparse_out, aux = sparse(x, return_aux=True)
    assert torch.allclose(sparse_out, dense_out, atol=1e-5)
    assert aux["selected_ids"].shape == (x.shape[0] * x.shape[1], cfg.model.d_ff)


def test_svd_cluster_pool_recovers_dense_ffn_when_pool_is_full() -> None:
    cfg = tiny_config("dense_exact")
    set_seed(123)
    dense = DenseModel(cfg.model)
    factors = _build_svd_factor_cache(
        dense,
        max_rank=min(cfg.model.d_model, cfg.model.d_ff),
        device=torch.device("cpu"),
    )
    tokens, _ = sample_batch(cfg)
    x = dense.embed(tokens)
    block = dense.blocks[0]
    u = x + block.attn(block.norm1(x))
    normed = block.norm2(u)
    clustered, aux = _ffn_neuron_svd_cluster_pool_output(
        block,
        normed,
        factors[0],
        rank=min(cfg.model.d_model, cfg.model.d_ff),
        cluster_count=2,
        candidate_m=cfg.model.d_ff,
        reference_k=min(4, cfg.model.d_ff),
        score_mode="sum",
        aggregation="mean",
        cluster_iters=2,
    )
    assert torch.allclose(clustered, block.mlp(normed), atol=1e-5)
    assert aux["selection_count"] == tokens.numel()


def test_mlx_svd_sparse_ffn_compiled_matches_eager_when_available() -> None:
    mx = pytest.importorskip("mlx.core")
    from recursive_training_engine.mlx_svd_ffn import (
        build_metal_candidate_swiglu_downsum,
        build_metal_fused_svd_sparse_ffn,
        build_mlx_svd_candidate_slots,
        build_mlx_svd_sparse_ffn,
    )

    rng = np.random.default_rng(123)
    d_model = 16
    d_ff = 32
    rank = 8
    tokens = 4
    x = mx.array(rng.standard_normal((tokens, d_model)).astype(np.float32))
    w_up = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32))
    w_gate = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32))
    w_down = mx.array(rng.standard_normal((d_ff, d_model)).astype(np.float32))
    up_a = mx.array(rng.standard_normal((d_model, rank)).astype(np.float32))
    up_b = mx.array(rng.standard_normal((rank, d_ff)).astype(np.float32))
    gate_a = mx.array(rng.standard_normal((d_model, rank)).astype(np.float32))
    gate_b = mx.array(rng.standard_normal((rank, d_ff)).astype(np.float32))
    wd_norm = mx.sqrt(mx.sum(w_down * w_down, axis=-1))
    eager = build_mlx_svd_sparse_ffn(top_k=8, up_m=8, product_m=8, compile_fn=False)
    compiled = build_mlx_svd_sparse_ffn(top_k=8, up_m=8, product_m=8, compile_fn=True)
    eager_out = eager(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
    compiled_out = compiled(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
    mx.eval(eager_out, compiled_out)
    assert np.allclose(np.array(eager_out), np.array(compiled_out), atol=1e-5)
    metal = build_metal_fused_svd_sparse_ffn(
        d_model=d_model,
        d_ff=d_ff,
        rank=rank,
        top_k=8,
        up_m=8,
        product_m=8,
    )
    metal_out = metal(x, w_up, w_gate, w_down, up_a, up_b, gate_a, gate_b, wd_norm)
    mx.eval(metal_out)
    assert np.allclose(np.array(eager_out), np.array(metal_out), atol=1e-4)

    candidate_slots = build_mlx_svd_candidate_slots(up_m=8, product_m=8, compile_fn=True)
    candidate_ids = candidate_slots(x, up_a, up_b, gate_a, gate_b, wd_norm)
    hybrid = build_metal_candidate_swiglu_downsum(
        d_model=d_model,
        candidate_slots=24,
        top_k=8,
        threads=64,
    )
    hybrid_out = hybrid(x, candidate_ids, w_up, w_gate, w_down, wd_norm)
    mx.eval(candidate_ids, hybrid_out)
    assert np.allclose(np.array(eager_out), np.array(hybrid_out), atol=1e-4)


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


def test_recursive_exact_accepts_per_sample_recipe_schedule() -> None:
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

    schedule = torch.tensor([[1, 2, 3, 1], [3, 2, 1, 3]])
    model.core.forward_step = record_step
    out = model.forward_exact(
        tokens,
        targets,
        fixed_depth=4,
        recipe_schedule=schedule,
    )
    assert [pass_idx for pass_idx, _ in seen] == [0, 1, 2, 3]
    assert [recipe_ids.tolist() for _, recipe_ids in seen] == [
        [1, 3],
        [2, 2],
        [3, 1],
        [1, 3],
    ]
    assert out.meta.recipe_schedule is not None
    assert torch.equal(out.meta.recipe_schedule.cpu(), schedule)


def test_router_emits_per_pass_recipe_logits() -> None:
    cfg = tiny_config("recursive_exact")
    model = RecursiveModel(cfg.model, cfg.output)
    tokens, _ = sample_batch(cfg)
    h0 = model._prelude(tokens)
    route = model.router(h0)
    assert route.recipe_logits_by_pass is not None
    assert route.recipe_probs_by_pass is not None
    assert route.recipe_id_by_pass is not None
    assert route.recipe_logits_by_pass.shape == (
        cfg.training.batch_size,
        cfg.model.t_max,
        cfg.model.recipe_count,
    )
    assert route.recipe_id_by_pass.shape == (cfg.training.batch_size, cfg.model.t_max)
    assert route.ffn_bank_logits_by_pass is not None
    assert route.head_slot_logits_by_pass is not None
    assert route.ffn_slot_logits_by_pass is not None
    assert route.ffn_bank_logits_by_pass.shape == (
        cfg.training.batch_size,
        cfg.model.t_max,
        cfg.model.ffn_banks,
    )
    assert route.head_slot_logits_by_pass.shape[:2] == (
        cfg.training.batch_size,
        cfg.model.t_max,
    )
    assert route.ffn_slot_logits_by_pass.shape[:2] == (
        cfg.training.batch_size,
        cfg.model.t_max,
    )


def test_recursive_factorized_soft_returns_route_metadata() -> None:
    cfg = tiny_config("recursive_exact_factorized_soft")
    model = RecursiveModel(cfg.model, cfg.output)
    out = model.forward_exact_factorized_soft(
        *sample_batch(cfg),
        return_loss_per_sample=True,
        top_k=3,
        fixed_depth=4,
    )
    assert out.loss_per_sample is not None
    assert out.meta.factor_route_tuples is not None
    assert out.meta.factor_route_weights is not None
    assert out.meta.factor_route_tuples.shape[:3] == (
        cfg.training.batch_size,
        cfg.model.t_max,
        3,
    )
    assert out.meta.factor_route_weights.shape == (
        cfg.training.batch_size,
        cfg.model.t_max,
        3,
    )
    assert torch.allclose(out.meta.factor_route_weights.sum(dim=-1), torch.ones_like(out.meta.factor_route_weights[..., 0]))


def test_recursive_dense_hidden_distill_train_step_cpu(tmp_path) -> None:
    cfg = tiny_config("recursive_exact_dense_hidden_distill")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            depth_choices=[1, 2, 3, 4],
            use_global_lowrank_corrector=True,
            global_corrector_rank=4,
        ),
        training=dataclasses.replace(
            cfg.training,
            fixed_recipe=1,
            fixed_recipe_schedule=[1, 2, 3, 1],
            fixed_depth=4,
            disable_router_aux=True,
            lambda_dense_hidden=0.1,
            lambda_dense_delta=0.1,
            lambda_dense_kl=0.1,
        ),
    )
    teacher = DenseModel(dataclasses.replace(cfg.model, topology="dense"))
    checkpoint = tmp_path / "dense_teacher.pt"
    torch.save({"model": teacher.state_dict()}, checkpoint)
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training,
            aligned_lm_teacher_checkpoint=str(checkpoint),
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        result = engine.train_step(sample_batch(cfg))
        assert torch.isfinite(result.loss)
        assert "dense_hidden_loss" in result.metrics
        assert "grad_norm_global_corrector" in result.metrics
    finally:
        engine.close()


def test_recursive_sandwich_supernet_train_step_cpu() -> None:
    cfg = tiny_config("recursive_sandwich_supernet")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, depth_choices=[1, 2, 3, 4]),
        training=dataclasses.replace(
            cfg.training,
            fixed_recipe=1,
            fixed_recipe_schedule=[1, 2, 3, 1],
            fixed_depth=4,
            disable_router_aux=True,
            sandwich_random_paths=1,
            lambda_sandwich_rand_ce=0.25,
            lambda_sandwich_kd=0.5,
            lambda_sandwich_rand_kd=0.25,
            lambda_sandwich_hidden=0.05,
        ),
    )
    engine = TrainEngine(cfg, device=torch.device("cpu"))
    try:
        result = engine.train_step(sample_batch(cfg))
        assert torch.isfinite(result.loss)
        assert result.metrics["mode"] == "recursive_sandwich_supernet"
        assert "full_nll_per_token" in result.metrics
        assert "thin_nll_per_token" in result.metrics
        assert "thin_kd_loss" in result.metrics
        assert result.metrics["full_recipe_schedule"] == [0, 0, 0, 0]
        assert result.metrics["thin_recipe_schedule"] == [1, 2, 3, 1]
    finally:
        engine.close()


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
    for mode in [
        "dense_exact",
        "recursive_exact",
        "recursive_exact_factorized_soft",
        "recursive_sandwich_supernet",
        "recursive_macro",
        "recursive_macro_shortlist",
    ]:
        cfg = tiny_config(mode)
        engine = TrainEngine(cfg, device=torch.device("cpu"))
        try:
            result = engine.train_step(sample_batch(cfg))
            assert torch.isfinite(result.loss)
        finally:
            engine.close()
