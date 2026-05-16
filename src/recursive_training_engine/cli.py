from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from recursive_training_engine.artifacts import build_manifest, summarize_metrics, write_json
from recursive_training_engine.config import ExperimentConfig, ModelConfig, load_config, save_config
from recursive_training_engine.ablations import build_ablation_configs
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.kernels import optimized, reference
from recursive_training_engine.kernels.active_swiglu_triton import (
    available as triton_swiglu_available,
    triton_packed_swiglu_ffn,
)
from recursive_training_engine.kernels.cluster_pool_ffn import (
    balanced_synthetic_assignments,
    build_static_pack_gather_indices,
    cluster_pool_ffn_forward_from_assignments,
    cluster_pool_ffn_forward_preindexed,
    cluster_pool_ffn_forward_static,
    prepare_cluster_pool_weights,
    route_to_static_centers,
    scatter_cluster_pool_grads,
    suggested_cluster_capacity,
    synthetic_cluster_centers,
)
from recursive_training_engine.layers import (
    ActiveUnionSwiGLU,
    PackedDenseSwiGLU,
    PackedActiveUnionSwiGLU,
    SVDFactorSparseFFN,
    StaticClusterPoolSwiGLU,
)
from recursive_training_engine.macro import (
    apply_macro_rms_clamp,
    macro_alignment_metrics,
    macro_distill_loss,
)
from recursive_training_engine.metrics import (
    build_fairness_report,
    dense_param_count,
    hidden_cosine,
    logit_kl,
    recursive_param_count,
    solve_banks_for_fairness,
)
from recursive_training_engine.models import (
    DenseModel,
    RecursiveModel,
    lm_loss_per_sample,
    load_compatible_state_dict,
)
from recursive_training_engine.training import TrainEngine
from recursive_training_engine.utils import default_device, set_seed


def _model_for_mode(config: ExperimentConfig, mode: str | None) -> ExperimentConfig:
    if mode is None:
        return config
    cfg = dataclasses.replace(config.training, mode=mode)
    topology = "dense" if mode == "dense_exact" else "recursive"
    model = dataclasses.replace(config.model, topology=topology)
    return dataclasses.replace(config, training=cfg, model=model)


def _with_run_dir(config: ExperimentConfig, run_dir: str | None) -> ExperimentConfig:
    if run_dir is None:
        return config
    path = Path(run_dir)
    return dataclasses.replace(
        config,
        output_dir=str(path.parent if str(path.parent) != "" else Path(".")),
        run_name=path.name,
    )


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


TRAIN_MODES = [
    "dense_exact",
    "recursive_exact",
    "recursive_deferred_grouped_exact",
    "recursive_exact_route_em",
    "recursive_exact_factorized_soft",
    "recursive_exact_dense_hidden_distill",
    "recursive_sandwich_supernet",
    "recursive_macro",
    "recursive_macro_shortlist",
    "recursive_macro_distill_only",
    "recursive_macro_lm_aligned",
    "recursive_macro_shadow_coda",
]


def cmd_fairness(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dense_cfg = dataclasses.replace(config.model, topology="dense")
    rec_cfg = dataclasses.replace(config.model, topology="recursive")
    report = build_fairness_report(
        dense_cfg,
        rec_cfg,
        tolerance=config.model.fairness_tolerance,
    )
    solved = solve_banks_for_fairness(rec_cfg, max_banks=args.max_banks)
    _print_json(
        {
            **dataclasses.asdict(report),
            "passed": report.passed,
            "suggested_attn_banks": solved[0],
            "suggested_ffn_banks": solved[1],
            "suggested_relative_delta": solved[2],
        }
    )
    if args.strict and not report.passed:
        raise SystemExit(2)


def cmd_train(args: argparse.Namespace) -> None:
    config = _with_run_dir(_model_for_mode(load_config(args.config), args.mode), args.run_dir)
    if getattr(args, "teacher_checkpoint", None):
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                aligned_lm_teacher_checkpoint=args.teacher_checkpoint,
                aligned_lm_freeze_teacher=True,
            ),
        )
    print(json.dumps({"event": "resolved_config", "config": dataclasses.asdict(config)}, sort_keys=True))
    set_seed(config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    engine = TrainEngine(config)
    engine.write_run_manifest(
        {
            "data_fingerprint": streams.data_fingerprint,
            "tokenizer": streams.tokenizer_name,
            "projection_lane": config.data.vocab_projection,
            "vocab_size": config.model.vocab_size,
            "seq_len": config.training.seq_len,
            "batch_size": config.training.batch_size,
            "mode": config.training.mode,
            "semantic_depth": config.training.fixed_depth or config.model.t_max,
            "audit_mode": config.training.audit_mode,
            "audit_probability": config.training.audit_p_min
            if config.training.audit_p_min == config.training.audit_p_max
            else [config.training.audit_p_min, config.training.audit_p_max],
            "audit_cap": config.training.audit_cap,
            "audit_sampler": config.training.audit_sampler,
            "audit_fixed_count": config.training.audit_fixed_count_per_batch,
            "macro_checkpoint_source": args.resume,
            "teacher_checkpoint_source": config.training.aligned_lm_teacher_checkpoint,
            "backend_status": optimized.backend_status(),
            "train_tokens": int(streams.train.numel()),
            "eval_tokens": int(streams.eval.numel()),
            "command": sys.argv,
        }
    )
    if args.resume:
        engine.load_checkpoint(args.resume)
    batches = streams.train_batches(config.training)
    try:
        for step in range(args.steps):
            result = engine.train_step(next(batches))
            if step % config.training.log_every == 0:
                printable = {
                    key: (float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else value)
                    for key, value in result.metrics.items()
                }
                printable["step"] = step + 1
                print(json.dumps(printable, sort_keys=True))
        save_checkpoint = args.save_checkpoint
        if save_checkpoint is None and args.run_dir is not None:
            save_checkpoint = str(engine.run_dir / "checkpoint.pt")
        if save_checkpoint:
            engine.save_checkpoint(save_checkpoint)
    finally:
        engine.close()


def cmd_evaluate(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), args.mode)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, targets = next(streams.eval_batches(config.training))
    device = default_device()
    tokens = tokens.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        if config.training.mode == "dense_exact":
            model = DenseModel(dataclasses.replace(config.model, topology="dense")).to(device)
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"], strict=True)
            out = model(tokens, targets, return_loss_per_sample=True)
        else:
            model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
            if args.checkpoint:
                payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["model"], strict=True)
            if config.training.mode in {
                "recursive_exact",
                "recursive_deferred_grouped_exact",
                "recursive_exact_route_em",
                "recursive_exact_dense_hidden_distill",
            }:
                use_fixed_route = config.training.mode in {
                    "recursive_exact",
                    "recursive_deferred_grouped_exact",
                    "recursive_exact_dense_hidden_distill",
                }
                if config.training.mode == "recursive_deferred_grouped_exact":
                    out = model.forward_deferred_grouped_exact(
                        tokens,
                        targets,
                        return_loss_per_sample=True,
                        fixed_recipe=config.training.fixed_recipe,
                        fixed_recipe_schedule=config.training.fixed_recipe_schedule,
                        fixed_depth=config.training.fixed_depth,
                    )
                else:
                    out = model.forward_exact(
                        tokens,
                        targets,
                        return_loss_per_sample=True,
                        fixed_recipe=config.training.fixed_recipe if use_fixed_route else None,
                        fixed_recipe_schedule=config.training.fixed_recipe_schedule
                        if use_fixed_route
                        else None,
                        fixed_depth=config.training.fixed_depth,
                    )
            elif config.training.mode == "recursive_sandwich_supernet":
                depth = config.training.fixed_depth or config.model.t_max
                full_schedule = [0 for _ in range(config.model.t_max)]
                target_schedule = list(
                    config.training.fixed_recipe_schedule
                    or [config.training.fixed_recipe if config.training.fixed_recipe is not None else 1]
                )
                while len(target_schedule) < config.model.t_max:
                    target_schedule.append(target_schedule[-1])
                full = model.forward_exact(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=0,
                    fixed_recipe_schedule=full_schedule,
                    fixed_depth=depth,
                )
                thin = model.forward_exact(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=target_schedule[0],
                    fixed_recipe_schedule=target_schedule,
                    fixed_depth=depth,
                )
                assert full.loss_per_sample is not None and thin.loss_per_sample is not None
                temperature = config.training.sandwich_temperature
                full_logp = F.log_softmax(full.meta.logits.detach() / temperature, dim=-1)
                thin_logp = F.log_softmax(thin.meta.logits / temperature, dim=-1)
                full_thin_kl = (
                    full_logp.exp() * (full_logp - thin_logp)
                ).sum(dim=-1).mean() * (temperature * temperature)
                _print_json(
                    {
                        "mode": config.training.mode,
                        "eval_full_loss_per_sample": float(full.loss_per_sample.mean().cpu()),
                        "eval_thin_loss_per_sample": float(thin.loss_per_sample.mean().cpu()),
                        "eval_full_nll_per_token": float(
                            (full.loss_per_sample.mean() / targets.shape[1]).cpu()
                        ),
                        "eval_thin_nll_per_token": float(
                            (thin.loss_per_sample.mean() / targets.shape[1]).cpu()
                        ),
                        "eval_full_thin_kl": float(full_thin_kl.cpu()),
                        "full_recipe_schedule": full_schedule,
                        "thin_recipe_schedule": target_schedule[: config.model.t_max],
                    }
                )
                return
            elif config.training.mode == "recursive_exact_factorized_soft":
                out = model.forward_exact_factorized_soft(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    top_k=config.training.factorized_route_top_k,
                    temperature=config.training.factorized_route_temperature,
                    fixed_depth=config.training.fixed_depth,
                )
            elif config.training.mode in {"recursive_macro", "recursive_macro_lm_aligned", "recursive_macro_distill_only"}:
                out = model.forward_macro(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=config.training.fixed_recipe,
                    fixed_depth=config.training.fixed_depth,
                )
            else:
                out = model.forward_macro(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    shortlist=True,
                    fixed_recipe=config.training.fixed_recipe,
                    fixed_depth=config.training.fixed_depth,
                )
    assert out.loss_per_sample is not None
    _print_json({"mode": config.training.mode, "eval_loss_per_sample": float(out.loss_per_sample.mean().cpu())})


def _route_oracle_static_schedule(config: ExperimentConfig, depth: int) -> list[int]:
    t_max = config.model.t_max
    if config.training.fixed_recipe_schedule:
        schedule = list(config.training.fixed_recipe_schedule)
    else:
        schedule = [config.training.fixed_recipe if config.training.fixed_recipe is not None else 1]
    if not schedule:
        schedule = [1]
    while len(schedule) < t_max:
        schedule.append(schedule[-1])
    return schedule[:t_max]


def _route_oracle_random_schedules(
    *,
    batch: int,
    candidates: int,
    t_max: int,
    recipe_count: int,
    device: torch.device,
    allow_dense_fallback: bool,
) -> torch.Tensor:
    start = 0 if allow_dense_fallback else 1
    if start >= recipe_count:
        raise ValueError("route oracle needs at least one selectable sparse recipe")
    return torch.randint(start, recipe_count, (batch, candidates, t_max), device=device)


def _route_oracle_router_schedules(
    model: RecursiveModel,
    h0: torch.Tensor,
    *,
    candidates: int,
    fixed_depth: int,
    allow_dense_fallback: bool,
) -> torch.Tensor:
    route = model.router(h0, fixed_depth=fixed_depth)
    if route.recipe_logits_by_pass is None:
        return _route_oracle_random_schedules(
            batch=h0.shape[0],
            candidates=candidates,
            t_max=model.config.t_max,
            recipe_count=model.config.recipe_count,
            device=h0.device,
            allow_dense_fallback=allow_dense_fallback,
        )
    logits = route.recipe_logits_by_pass.clone()
    if not allow_dense_fallback and logits.shape[-1] > 1:
        logits[..., 0] = torch.finfo(logits.dtype).min
    probs = torch.softmax(logits, dim=-1)
    schedules = torch.empty(
        h0.shape[0],
        candidates,
        model.config.t_max,
        device=h0.device,
        dtype=torch.long,
    )
    schedules[:, 0, :] = probs.argmax(dim=-1)
    if candidates > 1:
        draws = torch.multinomial(
            probs.reshape(h0.shape[0] * model.config.t_max, model.config.recipe_count),
            num_samples=candidates - 1,
            replacement=True,
        )
        schedules[:, 1:, :] = draws.view(h0.shape[0], model.config.t_max, candidates - 1).permute(
            0,
            2,
            1,
        )
    return schedules


@torch.no_grad()
def _route_oracle_loss_per_sample(
    model: RecursiveModel,
    h0: torch.Tensor,
    targets: torch.Tensor,
    schedules: torch.Tensor,
    *,
    fixed_depth: int,
) -> torch.Tensor:
    batch, candidates, t_max = schedules.shape
    flat_h0 = h0.unsqueeze(1).expand(-1, candidates, -1, -1).reshape(
        batch * candidates,
        h0.shape[1],
        h0.shape[2],
    )
    flat_targets = targets.unsqueeze(1).expand(-1, candidates, -1).reshape(
        batch * candidates,
        targets.shape[1],
    )
    flat_schedules = schedules.reshape(batch * candidates, t_max)
    out = model.forward_exact_from_h0(
        flat_h0,
        flat_targets,
        return_loss_per_sample=True,
        fixed_depth=fixed_depth,
        recipe_schedule=flat_schedules,
    )
    if out.loss_per_sample is None:
        raise RuntimeError("route oracle requires per-sample losses")
    return out.loss_per_sample.view(batch, candidates)


@torch.no_grad()
def cmd_route_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "recursive_exact")
    fixed_depth = args.depth if args.depth is not None else config.training.fixed_depth
    fixed_depth = fixed_depth if fixed_depth is not None else config.model.t_max
    training = config.training
    if args.batch_size is not None:
        training = dataclasses.replace(training, batch_size=args.batch_size)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, topology="recursive"),
        training=dataclasses.replace(training, fixed_depth=fixed_depth),
    )
    set_seed(args.seed if args.seed is not None else config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    device = default_device()
    model = RecursiveModel(config.model, config.output).to(device)
    load_info = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if payload.get("model") is None:
            raise SystemExit(f"checkpoint has no model state: {args.checkpoint}")
        load_info = load_compatible_state_dict(model, payload["model"])
    model.eval()
    counts = sorted(set(args.candidate_counts))
    totals = {count: [] for count in counts}
    static_losses = []
    static_schedule = torch.tensor(
        _route_oracle_static_schedule(config, fixed_depth),
        dtype=torch.long,
        device=device,
    )
    batches = streams.eval_batches(config.training)
    for _ in range(args.num_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        h0 = model._prelude(tokens)
        batch_static = static_schedule.unsqueeze(0).expand(tokens.shape[0], -1)
        static = _route_oracle_loss_per_sample(
            model,
            h0,
            targets,
            batch_static.unsqueeze(1),
            fixed_depth=fixed_depth,
        )
        static_losses.append(static[:, 0])
        for count in counts:
            if args.proposal == "router":
                schedules = _route_oracle_router_schedules(
                    model,
                    h0,
                    candidates=count,
                    fixed_depth=fixed_depth,
                    allow_dense_fallback=args.allow_dense_fallback,
                )
            else:
                schedules = _route_oracle_random_schedules(
                    batch=tokens.shape[0],
                    candidates=count,
                    t_max=model.config.t_max,
                    recipe_count=model.config.recipe_count,
                    device=device,
                    allow_dense_fallback=args.allow_dense_fallback,
                )
            if args.include_static:
                schedules[:, 0, :] = batch_static
            losses = _route_oracle_loss_per_sample(
                model,
                h0,
                targets,
                schedules,
                fixed_depth=fixed_depth,
            )
            totals[count].append(losses.min(dim=1).values)
    static_all = torch.cat(static_losses)
    rows = []
    static_nll = static_all.mean() / config.training.seq_len
    for count in counts:
        best = torch.cat(totals[count])
        rows.append(
            {
                "candidate_count": count,
                "oracle_nll_per_token": float((best.mean() / config.training.seq_len).cpu()),
                "improvement_vs_static": float(((static_all - best).mean() / config.training.seq_len).cpu()),
            }
        )
    _print_json(
        {
            "mode": "route_oracle",
            "checkpoint": args.checkpoint,
            "load_info": load_info,
            "proposal": args.proposal,
            "fixed_depth": fixed_depth,
            "num_batches": args.num_batches,
            "batch_size": config.training.batch_size,
            "static_schedule": static_schedule.cpu().tolist(),
            "static_nll_per_token": float(static_nll.cpu()),
            "rows": rows,
        }
    )


def _topk_list(raw: list[int] | None, default: list[int]) -> list[int]:
    values = default if raw is None else raw
    topks = sorted({int(value) for value in values})
    if not topks or topks[0] < 1:
        raise ValueError("top-k values must be positive")
    return topks


def _contiguous_group_ids(item_count: int, group_count: int) -> torch.Tensor:
    if item_count % group_count != 0:
        raise ValueError(f"item_count={item_count} must be divisible by group_count={group_count}")
    group_size = item_count // group_count
    return torch.arange(item_count, dtype=torch.long) // group_size


def _balanced_kmeans_labels(
    features: torch.Tensor,
    *,
    group_count: int,
    group_size: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    if features.shape[0] != group_count * group_size:
        raise ValueError("balanced clustering requires item_count == group_count * group_size")
    if group_size == 1:
        return torch.arange(group_count, dtype=torch.long)
    features = F.normalize(features.float().cpu(), dim=-1)
    item_count = features.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    first = int(torch.argmax(features.norm(dim=-1)).item())
    seeds = [first]
    while len(seeds) < group_count:
        sims = features @ features[torch.tensor(seeds)].t()
        best_existing = sims.max(dim=1).values
        best_existing[torch.tensor(seeds)] = float("inf")
        seeds.append(int(torch.argmin(best_existing).item()))
    centers = features[torch.tensor(seeds)].clone()
    labels = torch.zeros(item_count, dtype=torch.long)
    jitter = torch.rand(item_count, generator=generator) * 1e-6
    for _ in range(max(1, iters)):
        sims = features @ centers.t()
        top2 = torch.topk(sims, k=min(2, group_count), dim=-1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
        order = torch.argsort(margin + jitter, descending=True)
        prefs = torch.argsort(sims, dim=-1, descending=True)
        counts = torch.zeros(group_count, dtype=torch.long)
        labels.fill_(-1)
        for item in order.tolist():
            for group in prefs[item].tolist():
                if counts[group] < group_size:
                    labels[item] = group
                    counts[group] += 1
                    break
        for group in range(group_count):
            members = features[labels == group]
            if members.numel() == 0:
                centers[group] = features[torch.randint(0, item_count, (1,), generator=generator)]
            else:
                centers[group] = F.normalize(members.mean(dim=0), dim=0)
    return labels


def _labels_from_profile_gram(
    gram: torch.Tensor,
    sum_sq: torch.Tensor,
    *,
    group_count: int,
    iters: int,
    seed: int,
) -> torch.Tensor:
    item_count = int(gram.shape[0])
    group_size = item_count // group_count
    denom = torch.sqrt(torch.clamp(sum_sq.cpu().float(), min=1e-12))
    corr = gram.cpu().float() / torch.clamp(denom[:, None] * denom[None, :], min=1e-12)
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr.fill_diagonal_(1.0)
    try:
        eigvals, eigvecs = torch.linalg.eigh(corr)
        dim = min(max(2, group_count), item_count)
        vals = eigvals[-dim:].clamp_min(0.0).sqrt()
        features = eigvecs[:, -dim:] * vals
    except RuntimeError:
        features = corr
    return _balanced_kmeans_labels(
        features,
        group_count=group_count,
        group_size=group_size,
        iters=iters,
        seed=seed,
    )


def _perm_from_labels(labels: torch.Tensor, group_count: int) -> torch.Tensor:
    parts = []
    for group in range(group_count):
        members = torch.nonzero(labels == group, as_tuple=False).flatten()
        parts.append(members)
    return torch.cat(parts).long()


@torch.no_grad()
def _nll_dense_model(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    batches: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    iterator = streams.eval_batches(config.training)
    model.eval()
    for _ in range(batches):
        tokens, targets = next(iterator)
        tokens = tokens.to(device)
        targets = targets.to(device)
        out = model(tokens, targets, return_loss_per_sample=True)
        assert out.loss_per_sample is not None
        total_loss += float(out.loss_per_sample.sum().detach().cpu())
        total_tokens += int(targets.numel())
    return total_loss / max(total_tokens, 1)


def _ffn_group_scores(scores: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    one_hot = F.one_hot(group_ids.to(scores.device), num_classes=group_count).to(scores.dtype)
    return scores.reshape(-1, scores.shape[-1]) @ one_hot


def _ffn_topk_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_count: int,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = up * F.silu(gate)
    full = block.mlp.wd(z)
    top_k = min(max(1, int(top_k)), group_count)
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    scores = z.detach().abs() * wd_norm
    group_scores = _ffn_group_scores(scores, group_ids, group_count).view(*z.shape[:-1], group_count)
    top_ids = torch.topk(group_scores, k=top_k, dim=-1).indices
    selected_groups = F.one_hot(top_ids, num_classes=group_count).sum(dim=-2).bool()
    selected_flat = selected_groups.reshape(-1, group_count)
    neuron_mask = selected_flat[:, group_ids.to(z.device)].view_as(z)
    approx = block.mlp.wd(z * neuron_mask.to(z.dtype))
    selected_score = group_scores.gather(dim=-1, index=top_ids).sum(dim=-1)
    total_score = group_scores.sum(dim=-1).clamp_min(1e-12)
    return approx, {
        "full": full,
        "selected_score": selected_score,
        "total_score": total_score,
    }


def _selected_neuron_output(
    z_flat: torch.Tensor,
    wd_rows: torch.Tensor,
    selected_ids: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    out = z_flat.new_zeros(z_flat.shape[0], wd_rows.shape[1])
    for start in range(0, z_flat.shape[0], chunk_size):
        end = min(start + chunk_size, z_flat.shape[0])
        ids = selected_ids[start:end]
        coeff = torch.gather(z_flat[start:end], dim=-1, index=ids)
        weights = wd_rows[ids]
        out[start:end] = (coeff.unsqueeze(-1) * weights).sum(dim=1)
    return out


def _solve_neuron_ls_output(
    z_flat: torch.Tensor,
    wd_rows: torch.Tensor,
    target: torch.Tensor,
    selected_ids: torch.Tensor,
    *,
    ridge: float,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = target.new_zeros(target.shape)
    coeffs = z_flat.new_zeros(selected_ids.shape)
    d_model = wd_rows.shape[1]
    for start in range(0, z_flat.shape[0], chunk_size):
        end = min(start + chunk_size, z_flat.shape[0])
        ids = selected_ids[start:end]
        z_sel = torch.gather(z_flat[start:end], dim=-1, index=ids)
        selected = z_sel.unsqueeze(-1) * wd_rows[ids]
        target_chunk = target[start:end]
        k = selected.shape[1]
        if k <= d_model:
            gram = torch.einsum("nkd,nld->nkl", selected.float(), selected.float())
            rhs = torch.einsum("nkd,nd->nk", selected.float(), target_chunk.float())
            size = k
            eye = torch.eye(size, device=gram.device, dtype=gram.dtype)
            diag_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
            gram = gram + eye.view(1, size, size) * (float(ridge) * diag_mean).view(-1, 1, 1)
            if gram.device.type == "mps":
                coeff = torch.linalg.solve(gram.cpu(), rhs.cpu().unsqueeze(-1)).squeeze(-1).to(
                    device=selected.device
                )
            else:
                coeff = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
            approx = (selected.float() * coeff.unsqueeze(-1)).sum(dim=1)
        else:
            gram_d = torch.einsum("nkd,nke->nde", selected.float(), selected.float())
            size = d_model
            eye = torch.eye(size, device=gram_d.device, dtype=gram_d.dtype)
            diag_mean = gram_d.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
            gram_d = gram_d + eye.view(1, size, size) * (float(ridge) * diag_mean).view(-1, 1, 1)
            if gram_d.device.type == "mps":
                alpha = torch.linalg.solve(
                    gram_d.cpu(),
                    target_chunk.float().cpu().unsqueeze(-1),
                ).squeeze(-1).to(device=selected.device)
            else:
                alpha = torch.linalg.solve(gram_d, target_chunk.float().unsqueeze(-1)).squeeze(-1)
            coeff = torch.einsum("nkd,nd->nk", selected.float(), alpha)
            approx = torch.einsum("nkd,nk->nd", selected.float(), coeff)
        coeffs[start:end] = coeff.to(coeffs.dtype)
        out[start:end] = approx.to(out.dtype)
    return out, coeffs


def _solve_neuron_coefficients_prior(
    selected: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
    prior: float,
    clamp: float | None,
) -> torch.Tensor:
    selected_f = selected.float()
    target_f = target.float()
    gram = torch.einsum("nkd,nld->nkl", selected_f, selected_f)
    rhs = torch.einsum("nkd,nd->nk", selected_f, target_f)
    size = gram.shape[-1]
    eye = torch.eye(size, device=gram.device, dtype=gram.dtype)
    diag_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
    lam = float(ridge) * diag_mean
    prior_vec = torch.full_like(rhs, float(prior))
    gram = gram + eye.view(1, size, size) * lam.view(-1, 1, 1)
    rhs = rhs + lam.unsqueeze(-1) * prior_vec
    if gram.device.type == "mps":
        coeffs = torch.linalg.solve(gram.cpu(), rhs.cpu().unsqueeze(-1)).squeeze(-1).to(
            device=selected.device
        )
    else:
        coeffs = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    if clamp is not None:
        coeffs = coeffs.clamp(0.0, float(clamp))
    return coeffs.to(selected.dtype)


def _selection_jaccard(prev_ids: torch.Tensor, ids: torch.Tensor, *, universe: int) -> tuple[float, float, int]:
    prev_flat = prev_ids.reshape(-1, prev_ids.shape[-1])
    ids_flat = ids.reshape(-1, ids.shape[-1])
    prev_mask = F.one_hot(prev_flat, num_classes=universe).sum(dim=1).bool()
    mask = F.one_hot(ids_flat, num_classes=universe).sum(dim=1).bool()
    intersection = (prev_mask & mask).sum(dim=-1).float()
    union = (prev_mask | mask).sum(dim=-1).float().clamp_min(1.0)
    count = int(ids_flat.shape[0])
    return (
        float((intersection / union).sum().detach().cpu()),
        float((intersection / max(1, ids.shape[-1])).sum().detach().cpu()),
        count,
    )


@torch.no_grad()
def _ffn_neuron_sparse_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    *,
    top_k: int,
    selector: str,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    original_shape = normed.shape[:-1]
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = (up * F.silu(gate)).reshape(-1, block.mlp.wd.in_features)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    hidden = z.shape[-1]
    top_k = min(max(1, int(top_k)), hidden)
    full = block.mlp.wd(z).view(*original_shape, -1)
    if top_k == hidden:
        ids = torch.arange(hidden, device=z.device).view(1, hidden).expand(z.shape[0], -1)
        coeffs = z.new_ones(ids.shape)
        return full, _coeff_aux(coeffs), ids.view(*original_shape, hidden)
    wd_norm = wd_rows.detach().float().norm(dim=-1).to(z.device)
    target = full.reshape(z.shape[0], -1)
    if selector == "norm":
        scores = z.detach().abs() * wd_norm
        ids = torch.topk(scores, k=top_k, dim=-1).indices
        approx = _selected_neuron_output(z, wd_rows, ids)
        coeffs = z.new_ones(ids.shape)
        return approx.view(*original_shape, -1), _coeff_aux(coeffs), ids.view(*original_shape, top_k)
    selected_mask = torch.zeros(z.shape[0], hidden, device=z.device, dtype=torch.bool)
    residual = target
    selected: list[torch.Tensor] = []
    wd_norm_sq = wd_rows.detach().float().square().sum(dim=-1).to(z.device).clamp_min(1e-12)
    approx = target.new_zeros(target.shape)
    for _ in range(top_k):
        dot = (residual @ wd_rows.t()) * z
        norm_sq = z.detach().float().square().to(dot.dtype) * wd_norm_sq.to(dot.dtype)
        if selector == "omp_unit":
            scores = 2.0 * dot - norm_sq
        elif selector == "omp_ls":
            scores = dot.square() / norm_sq.clamp_min(1e-12)
        else:
            raise ValueError(f"unsupported neuron selector: {selector}")
        scores = scores.masked_fill(selected_mask, -torch.inf)
        chosen = scores.argmax(dim=-1)
        selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
        selected.append(chosen)
        piece = z.gather(dim=-1, index=chosen.unsqueeze(-1)) * wd_rows[chosen]
        if selector == "omp_unit":
            approx = approx + piece
            residual = residual - piece
        else:
            chosen_dot = dot.gather(dim=-1, index=chosen.unsqueeze(-1)).squeeze(-1)
            chosen_norm = norm_sq.gather(dim=-1, index=chosen.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            step_coeff = (chosen_dot / chosen_norm).to(piece.dtype)
            residual = residual - step_coeff.unsqueeze(-1) * piece
    ids = torch.stack(selected, dim=-1)
    if selector == "omp_unit":
        coeffs = z.new_ones(ids.shape)
    else:
        approx, coeffs = _solve_neuron_ls_output(z, wd_rows, target, ids, ridge=ridge)
    return approx.view(*original_shape, -1), _coeff_aux(coeffs), ids.view(*original_shape, top_k)


@torch.no_grad()
def _dense_forward_with_ffn_topk(
    model: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_ids_by_layer: list[torch.Tensor],
    top_k: int,
) -> torch.Tensor:
    x = model.embed(tokens)
    group_count = int(max(int(group_ids.max().item()) for group_ids in group_ids_by_layer) + 1)
    for layer_idx, block in enumerate(model.blocks):
        u = x + block.attn(block.norm1(x))
        mlp, _ = _ffn_topk_output(
            block,
            block.norm2(u),
            group_ids_by_layer[layer_idx],
            group_count=group_count,
            top_k=top_k,
        )
        x = u + mlp
    hidden = model.final_norm(x)
    logits = hidden @ model.vocab_weight.t()
    return lm_loss_per_sample(logits, targets)


@torch.no_grad()
def _nll_ffn_topk(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    group_ids_by_layer: list[torch.Tensor],
    top_k: int,
    batches: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    iterator = streams.eval_batches(config.training)
    model.eval()
    for _ in range(batches):
        tokens, targets = next(iterator)
        tokens = tokens.to(device)
        targets = targets.to(device)
        losses = _dense_forward_with_ffn_topk(
            model,
            tokens,
            targets,
            group_ids_by_layer=group_ids_by_layer,
            top_k=top_k,
        )
        total_loss += float(losses.sum().detach().cpu())
        total_tokens += int(targets.numel())
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _collect_ffn_profile_grams(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    batches: int,
    device: torch.device,
    progress: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    d_ff = config.model.d_ff
    grams = [torch.zeros(d_ff, d_ff, dtype=torch.float32, device=device) for _ in model.blocks]
    sum_sqs = [torch.zeros(d_ff, dtype=torch.float32, device=device) for _ in model.blocks]
    iterator = streams.train_batches(config.training)
    tokens_seen = 0
    model.eval()
    for batch_idx in range(batches):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        tokens_seen += int(tokens.numel())
        x = model.embed(tokens)
        for layer_idx, block in enumerate(model.blocks):
            u = x + block.attn(block.norm1(x))
            up, gate = block.mlp.wug(block.norm2(u)).chunk(2, dim=-1)
            z = up * F.silu(gate)
            wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
            scores = (z.detach().abs() * wd_norm).reshape(-1, d_ff).float()
            grams[layer_idx].add_(scores.t() @ scores)
            sum_sqs[layer_idx].add_(scores.square().sum(dim=0))
            x = u + block.mlp.wd(z)
        if progress and (batch_idx + 1) % progress == 0:
            print(json.dumps({"event": "ffn_regroup_profile_batch", "batch": batch_idx + 1}))
    return [g.cpu() for g in grams], [s.cpu() for s in sum_sqs], tokens_seen


@torch.no_grad()
def _ffn_retention_rows(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    groupings: dict[str, list[torch.Tensor]],
    topks: list[int],
    batches: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int], dict[str, float]] = {}
    group_count = config.model.ffn_groups
    iterator = streams.eval_batches(config.training)
    model.eval()
    for _ in range(batches):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        x = model.embed(tokens)
        for layer_idx, block in enumerate(model.blocks):
            u = x + block.attn(block.norm1(x))
            normed = block.norm2(u)
            dense_mlp = block.mlp(normed)
            for name, group_ids_by_layer in groupings.items():
                group_ids = group_ids_by_layer[layer_idx]
                for top_k in topks:
                    approx, aux = _ffn_topk_output(
                        block,
                        normed,
                        group_ids,
                        group_count=group_count,
                        top_k=top_k,
                    )
                    full = aux["full"]
                    key = (name, layer_idx + 1, top_k)
                    bucket = totals.setdefault(
                        key,
                        {
                            "selected_score": 0.0,
                            "total_score": 0.0,
                            "cos_sum": 0.0,
                            "cos_count": 0.0,
                            "err_sq": 0.0,
                            "full_sq": 0.0,
                        },
                    )
                    bucket["selected_score"] += float(aux["selected_score"].sum().detach().cpu())
                    bucket["total_score"] += float(aux["total_score"].sum().detach().cpu())
                    flat_full = full.reshape(-1, full.shape[-1]).float()
                    flat_approx = approx.reshape(-1, approx.shape[-1]).float()
                    cos = F.cosine_similarity(flat_approx, flat_full, dim=-1)
                    bucket["cos_sum"] += float(cos.sum().detach().cpu())
                    bucket["cos_count"] += float(cos.numel())
                    bucket["err_sq"] += float((flat_approx - flat_full).square().sum().detach().cpu())
                    bucket["full_sq"] += float(flat_full.square().sum().detach().cpu())
            x = u + dense_mlp
    rows = []
    for (grouping, layer, top_k), values in sorted(totals.items()):
        rows.append(
            {
                "grouping": grouping,
                "layer": layer,
                "top_k": top_k,
                "score_retention": values["selected_score"] / max(values["total_score"], 1e-12),
                "output_cosine": values["cos_sum"] / max(values["cos_count"], 1.0),
                "relative_error": math.sqrt(values["err_sq"] / max(values["full_sq"], 1e-12)),
            }
        )
    return rows


def _apply_ffn_permutations(model: DenseModel, permutations: list[torch.Tensor]) -> None:
    with torch.no_grad():
        d_ff = model.config.d_ff
        for block, perm_cpu in zip(model.blocks, permutations, strict=True):
            perm = perm_cpu.to(block.mlp.wug.weight.device)
            wug = block.mlp.wug.weight.data
            wd = block.mlp.wd.weight.data
            wug_up = wug[:d_ff].index_select(0, perm)
            wug_gate = wug[d_ff:].index_select(0, perm)
            wug.copy_(torch.cat([wug_up, wug_gate], dim=0))
            wd.copy_(wd.index_select(1, perm))


@torch.no_grad()
def _max_logit_diff(
    left: DenseModel,
    right: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    device: torch.device,
) -> float:
    tokens, _ = next(streams.eval_batches(config.training))
    tokens = tokens.to(device)
    a = left(tokens).meta.logits
    b = right(tokens).meta.logits
    assert a is not None and b is not None
    return float((a - b).abs().max().detach().cpu())


def cmd_ffn_regroup_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    topks = _topk_list(args.topk, [2, 4, 8])
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    grams, sum_sqs, profile_tokens = _collect_ffn_profile_grams(
        dense,
        streams,
        config,
        batches=args.profile_batches,
        device=device,
        progress=args.progress,
    )
    regrouped_ids = [
        _labels_from_profile_gram(
            gram,
            sum_sq,
            group_count=config.model.ffn_groups,
            iters=args.cluster_iters,
            seed=(args.seed if args.seed is not None else config.training.seed) + layer_idx,
        )
        for layer_idx, (gram, sum_sq) in enumerate(zip(grams, sum_sqs, strict=True))
    ]
    old_ids = [
        _contiguous_group_ids(config.model.d_ff, config.model.ffn_groups)
        for _ in dense.blocks
    ]
    permutations = [_perm_from_labels(labels, config.model.ffn_groups) for labels in regrouped_ids]
    retention_rows = _ffn_retention_rows(
        dense,
        streams,
        config,
        groupings={"old_contiguous": old_ids, "regrouped": regrouped_ids},
        topks=topks,
        batches=args.eval_batches,
        device=device,
    )
    dense_nll_before = _nll_dense_model(
        dense,
        streams,
        config,
        batches=args.eval_batches,
        device=device,
    )
    nll_rows = []
    for top_k in topks:
        nll_rows.append(
            {
                "grouping": "old_contiguous",
                "top_k": top_k,
                "nll_per_token": _nll_ffn_topk(
                    dense,
                    streams,
                    config,
                    group_ids_by_layer=old_ids,
                    top_k=top_k,
                    batches=args.eval_batches,
                    device=device,
                ),
            }
        )
        nll_rows.append(
            {
                "grouping": "regrouped",
                "top_k": top_k,
                "nll_per_token": _nll_ffn_topk(
                    dense,
                    streams,
                    config,
                    group_ids_by_layer=regrouped_ids,
                    top_k=top_k,
                    batches=args.eval_batches,
                    device=device,
                ),
            }
        )
    permuted = copy.deepcopy(dense).to(device)
    _apply_ffn_permutations(permuted, permutations)
    dense_nll_after = _nll_dense_model(
        permuted,
        streams,
        config,
        batches=args.eval_batches,
        device=device,
    )
    contiguous_after = [
        _contiguous_group_ids(config.model.d_ff, config.model.ffn_groups)
        for _ in permuted.blocks
    ]
    permuted_topk_rows = [
        {
            "grouping": "permuted_contiguous",
            "top_k": top_k,
            "nll_per_token": _nll_ffn_topk(
                permuted,
                streams,
                config,
                group_ids_by_layer=contiguous_after,
                top_k=top_k,
                batches=args.eval_batches,
                device=device,
            ),
        }
        for top_k in topks
    ]
    report = {
        "mode": "ffn_regroup_oracle",
        "checkpoint": args.dense_checkpoint,
        "profile_batches": args.profile_batches,
        "profile_tokens": profile_tokens,
        "eval_batches": args.eval_batches,
        "eval_tokens": args.eval_batches * config.training.batch_size * config.training.seq_len,
        "topk": topks,
        "ffn_groups": config.model.ffn_groups,
        "group_size": config.model.d_ff // config.model.ffn_groups,
        "dense_nll_before_permutation": dense_nll_before,
        "dense_nll_after_permutation": dense_nll_after,
        "dense_nll_permutation_delta": dense_nll_after - dense_nll_before,
        "permutation_max_logit_abs_diff_one_batch": _max_logit_diff(
            dense,
            permuted,
            streams,
            config,
            device=device,
        ),
        "topk_nll": nll_rows,
        "permuted_contiguous_topk_nll": permuted_topk_rows,
        "retention_by_layer": retention_rows,
    }
    if args.include_permutations:
        report["permutations"] = [perm.tolist() for perm in permutations]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _head_group_scores(scores: torch.Tensor, group_ids: torch.Tensor, group_count: int) -> torch.Tensor:
    one_hot = F.one_hot(group_ids.to(scores.device), num_classes=group_count).to(scores.dtype)
    return scores.reshape(-1, scores.shape[-1]) @ one_hot


def _dense_attention_heads(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    b, s, d = x.shape
    n_heads = block.attn.n_heads
    head_dim = block.attn.head_dim
    qkv = block.attn.wqkv(x).view(b, s, 3, n_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if block.attn.rope is not None:
        q, k = block.attn.rope(q, k)
    y = optimized.k_flash_causal_dense(q, k, v)
    return y.transpose(1, 2).contiguous()


def _attention_head_contribs(block: torch.nn.Module, heads: torch.Tensor) -> torch.Tensor:
    cols_per_head = block.attn.head_dim
    pieces = []
    for head in range(block.attn.n_heads):
        cols = slice(head * cols_per_head, (head + 1) * cols_per_head)
        pieces.append(F.linear(heads[:, :, head, :], block.attn.wo.weight[:, cols]))
    return torch.stack(pieces, dim=2)


def _attention_topk_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_count: int,
    top_k: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    heads = _dense_attention_heads(block, normed)
    full = block.attn.wo(heads.reshape(normed.shape[0], normed.shape[1], -1))
    contribs = _attention_head_contribs(block, heads)
    scores = contribs.detach().float().norm(dim=-1)
    top_k = min(max(1, int(top_k)), group_count)
    group_scores = _head_group_scores(scores, group_ids, group_count).view(*scores.shape[:-1], group_count)
    top_ids = torch.topk(group_scores, k=top_k, dim=-1).indices
    selected_groups = F.one_hot(top_ids, num_classes=group_count).sum(dim=-2).bool()
    selected_flat = selected_groups.reshape(-1, group_count)
    head_mask = selected_flat[:, group_ids.to(normed.device)].view(*scores.shape)
    masked = heads * head_mask.unsqueeze(-1).to(heads.dtype)
    approx = block.attn.wo(masked.reshape(normed.shape[0], normed.shape[1], -1))
    selected_score = group_scores.gather(dim=-1, index=top_ids).sum(dim=-1)
    total_score = group_scores.sum(dim=-1).clamp_min(1e-12)
    return approx, {
        "full": full,
        "selected_score": selected_score,
        "total_score": total_score,
    }


@torch.no_grad()
def _dense_forward_with_head_topk(
    model: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_ids_by_layer: list[torch.Tensor],
    top_k: int,
) -> torch.Tensor:
    x = model.embed(tokens)
    group_count = int(max(int(group_ids.max().item()) for group_ids in group_ids_by_layer) + 1)
    for layer_idx, block in enumerate(model.blocks):
        attn, _ = _attention_topk_output(
            block,
            block.norm1(x),
            group_ids_by_layer[layer_idx],
            group_count=group_count,
            top_k=top_k,
        )
        u = x + attn
        x = u + block.mlp(block.norm2(u))
    hidden = model.final_norm(x)
    logits = hidden @ model.vocab_weight.t()
    return lm_loss_per_sample(logits, targets)


@torch.no_grad()
def _nll_head_topk(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    group_ids_by_layer: list[torch.Tensor],
    top_k: int,
    batches: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    iterator = streams.eval_batches(config.training)
    model.eval()
    for _ in range(batches):
        tokens, targets = next(iterator)
        tokens = tokens.to(device)
        targets = targets.to(device)
        losses = _dense_forward_with_head_topk(
            model,
            tokens,
            targets,
            group_ids_by_layer=group_ids_by_layer,
            top_k=top_k,
        )
        total_loss += float(losses.sum().detach().cpu())
        total_tokens += int(targets.numel())
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _collect_head_profile_grams(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    batches: int,
    device: torch.device,
    progress: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
    n_heads = config.model.n_heads
    grams = [torch.zeros(n_heads, n_heads, dtype=torch.float32, device=device) for _ in model.blocks]
    sum_sqs = [torch.zeros(n_heads, dtype=torch.float32, device=device) for _ in model.blocks]
    iterator = streams.train_batches(config.training)
    tokens_seen = 0
    model.eval()
    for batch_idx in range(batches):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        tokens_seen += int(tokens.numel())
        x = model.embed(tokens)
        for layer_idx, block in enumerate(model.blocks):
            normed = block.norm1(x)
            heads = _dense_attention_heads(block, normed)
            contribs = _attention_head_contribs(block, heads)
            scores = contribs.detach().float().norm(dim=-1).reshape(-1, n_heads)
            grams[layer_idx].add_(scores.t() @ scores)
            sum_sqs[layer_idx].add_(scores.square().sum(dim=0))
            u = x + block.attn.wo(heads.reshape(x.shape[0], x.shape[1], -1))
            x = u + block.mlp(block.norm2(u))
        if progress and (batch_idx + 1) % progress == 0:
            print(json.dumps({"event": "head_regroup_profile_batch", "batch": batch_idx + 1}))
    return [g.cpu() for g in grams], [s.cpu() for s in sum_sqs], tokens_seen


@torch.no_grad()
def _head_retention_rows(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    groupings: dict[str, list[torch.Tensor]],
    topks: list[int],
    batches: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int, int], dict[str, float]] = {}
    group_count = config.model.head_groups
    iterator = streams.eval_batches(config.training)
    model.eval()
    for _ in range(batches):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        x = model.embed(tokens)
        for layer_idx, block in enumerate(model.blocks):
            normed = block.norm1(x)
            dense_attn = block.attn(normed)
            for name, group_ids_by_layer in groupings.items():
                group_ids = group_ids_by_layer[layer_idx]
                for top_k in topks:
                    approx, aux = _attention_topk_output(
                        block,
                        normed,
                        group_ids,
                        group_count=group_count,
                        top_k=top_k,
                    )
                    full = aux["full"]
                    key = (name, layer_idx + 1, top_k)
                    bucket = totals.setdefault(
                        key,
                        {
                            "selected_score": 0.0,
                            "total_score": 0.0,
                            "cos_sum": 0.0,
                            "cos_count": 0.0,
                            "err_sq": 0.0,
                            "full_sq": 0.0,
                        },
                    )
                    bucket["selected_score"] += float(aux["selected_score"].sum().detach().cpu())
                    bucket["total_score"] += float(aux["total_score"].sum().detach().cpu())
                    flat_full = full.reshape(-1, full.shape[-1]).float()
                    flat_approx = approx.reshape(-1, approx.shape[-1]).float()
                    cos = F.cosine_similarity(flat_approx, flat_full, dim=-1)
                    bucket["cos_sum"] += float(cos.sum().detach().cpu())
                    bucket["cos_count"] += float(cos.numel())
                    bucket["err_sq"] += float((flat_approx - flat_full).square().sum().detach().cpu())
                    bucket["full_sq"] += float(flat_full.square().sum().detach().cpu())
            u = x + dense_attn
            x = u + block.mlp(block.norm2(u))
    rows = []
    for (grouping, layer, top_k), values in sorted(totals.items()):
        rows.append(
            {
                "grouping": grouping,
                "layer": layer,
                "top_k": top_k,
                "score_retention": values["selected_score"] / max(values["total_score"], 1e-12),
                "output_cosine": values["cos_sum"] / max(values["cos_count"], 1.0),
                "relative_error": math.sqrt(values["err_sq"] / max(values["full_sq"], 1e-12)),
            }
        )
    return rows


def _apply_head_permutations(model: DenseModel, permutations: list[torch.Tensor]) -> None:
    with torch.no_grad():
        d = model.config.d_model
        head_dim = d // model.config.n_heads
        for block, perm_cpu in zip(model.blocks, permutations, strict=True):
            perm = perm_cpu.to(block.attn.wqkv.weight.device)
            old_head_rows = torch.cat(
                [
                    torch.arange(int(head) * head_dim, (int(head) + 1) * head_dim, device=perm.device)
                    for head in perm.tolist()
                ]
            )
            qkv_rows = torch.cat([old_head_rows + offset * d for offset in range(3)])
            block.attn.wqkv.weight.data.copy_(block.attn.wqkv.weight.data.index_select(0, qkv_rows))
            block.attn.wo.weight.data.copy_(block.attn.wo.weight.data.index_select(1, old_head_rows))


def cmd_head_regroup_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    topks = _topk_list(args.topk, [config.model.active_head_groups])
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    grams, sum_sqs, profile_tokens = _collect_head_profile_grams(
        dense,
        streams,
        config,
        batches=args.profile_batches,
        device=device,
        progress=args.progress,
    )
    regrouped_ids = [
        _labels_from_profile_gram(
            gram,
            sum_sq,
            group_count=config.model.head_groups,
            iters=args.cluster_iters,
            seed=(args.seed if args.seed is not None else config.training.seed) + layer_idx,
        )
        for layer_idx, (gram, sum_sq) in enumerate(zip(grams, sum_sqs, strict=True))
    ]
    old_ids = [
        _contiguous_group_ids(config.model.n_heads, config.model.head_groups)
        for _ in dense.blocks
    ]
    permutations = [_perm_from_labels(labels, config.model.head_groups) for labels in regrouped_ids]
    retention_rows = _head_retention_rows(
        dense,
        streams,
        config,
        groupings={"old_contiguous": old_ids, "regrouped": regrouped_ids},
        topks=topks,
        batches=args.eval_batches,
        device=device,
    )
    dense_nll_before = _nll_dense_model(dense, streams, config, batches=args.eval_batches, device=device)
    nll_rows = []
    for top_k in topks:
        nll_rows.append(
            {
                "grouping": "old_contiguous",
                "top_k": top_k,
                "nll_per_token": _nll_head_topk(
                    dense,
                    streams,
                    config,
                    group_ids_by_layer=old_ids,
                    top_k=top_k,
                    batches=args.eval_batches,
                    device=device,
                ),
            }
        )
        nll_rows.append(
            {
                "grouping": "regrouped",
                "top_k": top_k,
                "nll_per_token": _nll_head_topk(
                    dense,
                    streams,
                    config,
                    group_ids_by_layer=regrouped_ids,
                    top_k=top_k,
                    batches=args.eval_batches,
                    device=device,
                ),
            }
        )
    permuted = copy.deepcopy(dense).to(device)
    _apply_head_permutations(permuted, permutations)
    dense_nll_after = _nll_dense_model(permuted, streams, config, batches=args.eval_batches, device=device)
    contiguous_after = [
        _contiguous_group_ids(config.model.n_heads, config.model.head_groups)
        for _ in permuted.blocks
    ]
    permuted_topk_rows = [
        {
            "grouping": "permuted_contiguous",
            "top_k": top_k,
            "nll_per_token": _nll_head_topk(
                permuted,
                streams,
                config,
                group_ids_by_layer=contiguous_after,
                top_k=top_k,
                batches=args.eval_batches,
                device=device,
            ),
        }
        for top_k in topks
    ]
    report = {
        "mode": "head_regroup_oracle",
        "checkpoint": args.dense_checkpoint,
        "profile_batches": args.profile_batches,
        "profile_tokens": profile_tokens,
        "eval_batches": args.eval_batches,
        "eval_tokens": args.eval_batches * config.training.batch_size * config.training.seq_len,
        "topk": topks,
        "head_groups": config.model.head_groups,
        "heads_per_group": config.model.n_heads // config.model.head_groups,
        "dense_nll_before_permutation": dense_nll_before,
        "dense_nll_after_permutation": dense_nll_after,
        "dense_nll_permutation_delta": dense_nll_after - dense_nll_before,
        "permutation_max_logit_abs_diff_one_batch": _max_logit_diff(
            dense,
            permuted,
            streams,
            config,
            device=device,
        ),
        "topk_nll": nll_rows,
        "permuted_contiguous_topk_nll": permuted_topk_rows,
        "retention_by_layer": retention_rows,
    }
    if args.include_permutations:
        report["permutations"] = [perm.tolist() for perm in permutations]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _dense_states_including_embed(dense: DenseModel, tokens: torch.Tensor) -> dict[int, torch.Tensor]:
    states: dict[int, torch.Tensor] = {0: dense.embed(tokens)}
    x = states[0]
    for idx, block in enumerate(dense.blocks, start=1):
        x = block(x)
        states[idx] = x
    return states


def _group_mask_from_ids(
    group_ids: torch.Tensor,
    selected_groups: torch.Tensor | list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.as_tensor(selected_groups, dtype=torch.long, device=device)
    return (group_ids.to(device).view(-1, 1) == selected.view(1, -1)).any(dim=1)


def _attention_selected_group_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    selected_groups: torch.Tensor | list[int],
) -> torch.Tensor:
    heads = _dense_attention_heads(block, normed)
    mask = _group_mask_from_ids(group_ids, selected_groups, device=normed.device)
    heads = heads * mask.view(1, 1, -1, 1).to(heads.dtype)
    return block.attn.wo(heads.reshape(normed.shape[0], normed.shape[1], -1))


def _ffn_selected_group_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    selected_groups: torch.Tensor | list[int],
) -> torch.Tensor:
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = up * F.silu(gate)
    mask = _group_mask_from_ids(group_ids, selected_groups, device=normed.device)
    return block.mlp.wd(z * mask.view(1, 1, -1).to(z.dtype))


def _deferred_grouped_block_all(
    block: torch.nn.Module,
    h: torch.Tensor,
    *,
    attn_group_ids: torch.Tensor,
    ffn_group_ids: torch.Tensor,
    head_groups: int,
    ffn_groups: int,
) -> torch.Tensor:
    attn = _attention_selected_group_output(
        block,
        block.norm1(h),
        attn_group_ids,
        list(range(head_groups)),
    )
    u = h + attn
    ffn = _ffn_selected_group_output(
        block,
        block.norm2(u),
        ffn_group_ids,
        list(range(ffn_groups)),
    )
    return u + ffn


def _sequential_grouped_block_all(
    block: torch.nn.Module,
    h: torch.Tensor,
    *,
    attn_group_ids: torch.Tensor,
    ffn_group_ids: torch.Tensor,
    head_groups: int,
    ffn_groups: int,
) -> torch.Tensor:
    out = h
    for group in range(head_groups):
        out = out + _attention_selected_group_output(
            block,
            block.norm1(out),
            attn_group_ids,
            [group],
        )
    for group in range(ffn_groups):
        out = out + _ffn_selected_group_output(
            block,
            block.norm2(out),
            ffn_group_ids,
            [group],
        )
    return out


def _deferred_grouped_block_topk(
    block: torch.nn.Module,
    h: torch.Tensor,
    *,
    attn_group_ids: torch.Tensor,
    ffn_group_ids: torch.Tensor,
    head_groups: int,
    ffn_groups: int,
    attn_top_k: int,
    ffn_top_k: int,
) -> torch.Tensor:
    attn, _ = _attention_topk_output(
        block,
        block.norm1(h),
        attn_group_ids,
        group_count=head_groups,
        top_k=attn_top_k,
    )
    u = h + attn
    ffn, _ = _ffn_topk_output(
        block,
        block.norm2(u),
        ffn_group_ids,
        group_count=ffn_groups,
        top_k=ffn_top_k,
    )
    return u + ffn


def _attention_group_contribs(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    heads = _dense_attention_heads(block, normed)
    head_contribs = _attention_head_contribs(block, heads)
    pieces = []
    for group in range(group_count):
        mask = group_ids.to(normed.device) == group
        pieces.append(head_contribs[:, :, mask, :].sum(dim=2))
    return torch.stack(pieces, dim=2)


def _ffn_group_contribs(
    block: torch.nn.Module,
    normed: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    group_count: int,
) -> torch.Tensor:
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = up * F.silu(gate)
    pieces = []
    for group in range(group_count):
        mask = group_ids.to(normed.device) == group
        pieces.append(F.linear(z[:, :, mask], block.mlp.wd.weight[:, mask]))
    return torch.stack(pieces, dim=2)


def _gather_contribs(contribs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.unsqueeze(-1).expand(*indices.shape, contribs.shape[-1])
    return torch.gather(contribs, dim=2, index=gather_idx)


def _solve_group_coefficients(
    selected: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float,
) -> torch.Tensor:
    selected_f = selected.float()
    target_f = target.float()
    gram = torch.einsum("...kd,...ld->...kl", selected_f, selected_f)
    rhs = torch.einsum("...kd,...d->...k", selected_f, target_f)
    size = gram.shape[-1]
    eye = torch.eye(size, device=gram.device, dtype=gram.dtype)
    diag_mean = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
    gram = gram + eye.view(*((1,) * (gram.ndim - 2)), size, size) * (
        float(ridge) * diag_mean
    ).unsqueeze(-1).unsqueeze(-1)
    if gram.device.type == "mps":
        coeffs = torch.linalg.solve(
            gram.cpu(),
            rhs.cpu().unsqueeze(-1),
        ).squeeze(-1).to(device=selected.device)
    else:
        coeffs = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    return coeffs.to(selected.dtype)


def _coeff_aux(coeffs: torch.Tensor) -> dict[str, float]:
    coeffs_f = coeffs.detach().float()
    count = int(coeffs_f.numel())
    return {
        "coeff_count": float(count),
        "coeff_sum": float(coeffs_f.sum().cpu()),
        "coeff_abs_sum": float(coeffs_f.abs().sum().cpu()),
        "coeff_sq_sum": float(coeffs_f.square().sum().cpu()),
    }


def _add_coeff_aux(dst: dict[str, float], src: dict[str, float]) -> None:
    for key in ("coeff_count", "coeff_sum", "coeff_abs_sum", "coeff_sq_sum"):
        dst[key] = dst.get(key, 0.0) + float(src.get(key, 0.0))


def _group_sparse_approx(
    contribs: torch.Tensor,
    *,
    top_k: int,
    selector: str,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    group_count = contribs.shape[2]
    top_k = min(max(1, int(top_k)), group_count)
    target = contribs.sum(dim=2)
    if top_k == group_count:
        coeffs = contribs.new_ones(*contribs.shape[:2], top_k)
        return target, _coeff_aux(coeffs)
    if selector == "norm":
        scores = contribs.detach().float().norm(dim=-1)
        top_ids = torch.topk(scores, k=top_k, dim=-1).indices
        selected = _gather_contribs(contribs, top_ids)
        coeffs = contribs.new_ones(*top_ids.shape)
        return selected.sum(dim=2), _coeff_aux(coeffs)
    selected_mask = torch.zeros(*contribs.shape[:2], group_count, device=contribs.device, dtype=torch.bool)
    residual = target
    selected_pieces: list[torch.Tensor] = []
    if selector == "omp_unit":
        for _ in range(top_k):
            dot = (contribs * residual.unsqueeze(2)).sum(dim=-1)
            norm_sq = contribs.detach().float().square().sum(dim=-1).to(dot.dtype)
            scores = (2.0 * dot - norm_sq).masked_fill(selected_mask, -torch.inf)
            chosen = scores.argmax(dim=-1)
            selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
            piece = _gather_contribs(contribs, chosen.unsqueeze(-1)).squeeze(2)
            selected_pieces.append(piece)
            residual = residual - piece
        coeffs = contribs.new_ones(*contribs.shape[:2], top_k)
        return torch.stack(selected_pieces, dim=2).sum(dim=2), _coeff_aux(coeffs)
    if selector != "omp_ls":
        raise ValueError(f"unsupported full-stack oracle selector: {selector}")
    approx = torch.zeros_like(target)
    coeffs = contribs.new_zeros(*contribs.shape[:2], 0)
    for _ in range(top_k):
        dot = (contribs * residual.unsqueeze(2)).sum(dim=-1)
        norm_sq = contribs.detach().float().square().sum(dim=-1).to(dot.dtype).clamp_min(1e-12)
        scores = (dot.square() / norm_sq).masked_fill(selected_mask, -torch.inf)
        chosen = scores.argmax(dim=-1)
        selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
        selected_pieces.append(_gather_contribs(contribs, chosen.unsqueeze(-1)).squeeze(2))
        selected = torch.stack(selected_pieces, dim=2)
        coeffs = _solve_group_coefficients(selected, target, ridge=ridge)
        approx = (selected * coeffs.unsqueeze(-1)).sum(dim=2)
        residual = target - approx
    return approx, _coeff_aux(coeffs)


def _deferred_grouped_block_sparse_oracle(
    block: torch.nn.Module,
    h: torch.Tensor,
    *,
    attn_group_ids: torch.Tensor,
    ffn_group_ids: torch.Tensor,
    head_groups: int,
    ffn_groups: int,
    top_k: int,
    selector: str,
    ridge: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if selector == "norm":
        out = _deferred_grouped_block_topk(
            block,
            h,
            attn_group_ids=attn_group_ids,
            ffn_group_ids=ffn_group_ids,
            head_groups=head_groups,
            ffn_groups=ffn_groups,
            attn_top_k=min(top_k, head_groups),
            ffn_top_k=min(top_k, ffn_groups),
        )
        coeffs = h.new_ones(
            h.shape[0],
            h.shape[1],
            min(top_k, head_groups) + min(top_k, ffn_groups),
        )
        return out, _coeff_aux(coeffs)
    aux: dict[str, float] = {}
    attn_contribs = _attention_group_contribs(
        block,
        block.norm1(h),
        attn_group_ids,
        group_count=head_groups,
    )
    attn, attn_aux = _group_sparse_approx(
        attn_contribs,
        top_k=min(top_k, head_groups),
        selector=selector,
        ridge=ridge,
    )
    _add_coeff_aux(aux, attn_aux)
    u = h + attn
    ffn_contribs = _ffn_group_contribs(
        block,
        block.norm2(u),
        ffn_group_ids,
        group_count=ffn_groups,
    )
    ffn, ffn_aux = _group_sparse_approx(
        ffn_contribs,
        top_k=min(top_k, ffn_groups),
        selector=selector,
        ridge=ridge,
    )
    _add_coeff_aux(aux, ffn_aux)
    return u + ffn, aux


def _full_stack_deferred_logits_and_states(
    dense: DenseModel,
    tokens: torch.Tensor,
    *,
    attn_group_ids: torch.Tensor,
    ffn_group_ids: torch.Tensor,
    head_groups: int,
    ffn_groups: int,
    top_k: int | None,
    selector: str = "norm",
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, float]]:
    x = dense.embed(tokens)
    states: dict[int, torch.Tensor] = {}
    aux: dict[str, float] = {}
    for layer_idx, block in enumerate(dense.blocks, start=1):
        if top_k is None:
            x = _deferred_grouped_block_all(
                block,
                x,
                attn_group_ids=attn_group_ids,
                ffn_group_ids=ffn_group_ids,
                head_groups=head_groups,
                ffn_groups=ffn_groups,
            )
        else:
            x, layer_aux = _deferred_grouped_block_sparse_oracle(
                block,
                x,
                attn_group_ids=attn_group_ids,
                ffn_group_ids=ffn_group_ids,
                head_groups=head_groups,
                ffn_groups=ffn_groups,
                top_k=top_k,
                selector=selector,
                ridge=ridge,
            )
            _add_coeff_aux(aux, layer_aux)
        states[layer_idx] = x
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), states, aux


@torch.no_grad()
def _block_oracle_metrics(
    dense: DenseModel,
    candidate: torch.Tensor,
    target: torch.Tensor,
    targets: torch.Tensor,
    *,
    state_layer: int,
    temperature: float,
) -> dict[str, float]:
    cand = candidate.float()
    tgt = target.float()
    cos = F.cosine_similarity(cand.flatten(1), tgt.flatten(1), dim=-1).mean()
    mse = (cand - tgt).square().mean()
    logits = _dense_suffix_logits_from_state(dense, candidate, state_layer=state_layer)
    target_logits = _dense_suffix_logits_from_state(dense, target, state_layer=state_layer)
    nll = F.cross_entropy(logits.flatten(0, -2), targets.flatten())
    target_nll = F.cross_entropy(target_logits.flatten(0, -2), targets.flatten())
    target_logp = F.log_softmax(target_logits.detach() / temperature, dim=-1)
    cand_logp = F.log_softmax(logits / temperature, dim=-1)
    kl = (target_logp.exp() * (target_logp - cand_logp)).sum(dim=-1).mean() * (
        temperature * temperature
    )
    return {
        "hidden_cosine": float(cos.detach().cpu()),
        "hidden_mse": float(mse.detach().cpu()),
        "suffix_kl": float(kl.detach().cpu()),
        "nll_per_token": float(nll.detach().cpu()),
        "target_nll_per_token": float(target_nll.detach().cpu()),
    }


def cmd_deferred_grouped_block_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    if args.layer < 1 or args.layer > config.model.n_dense_layers:
        raise SystemExit(f"--layer must be in [1, {config.model.n_dense_layers}]")
    topks = _topk_list(args.topk, [2, 4, 8])
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    block = dense.blocks[args.layer - 1]
    attn_group_ids = _contiguous_group_ids(config.model.n_heads, config.model.head_groups)
    ffn_group_ids = _contiguous_group_ids(config.model.d_ff, config.model.ffn_groups)
    rows: list[dict[str, Any]] = []
    max_abs_diffs: dict[str, float] = {}
    batches = streams.eval_batches(config.training)
    for batch_idx in range(args.num_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        states = _dense_states_including_embed(dense, tokens)
        input_state = states[args.layer - 1]
        target = states[args.layer]
        dense_block = block(input_state)
        deferred_all = _deferred_grouped_block_all(
            block,
            input_state,
            attn_group_ids=attn_group_ids,
            ffn_group_ids=ffn_group_ids,
            head_groups=config.model.head_groups,
            ffn_groups=config.model.ffn_groups,
        )
        sequential_all = _sequential_grouped_block_all(
            block,
            input_state,
            attn_group_ids=attn_group_ids,
            ffn_group_ids=ffn_group_ids,
            head_groups=config.model.head_groups,
            ffn_groups=config.model.ffn_groups,
        )
        variants: list[tuple[str, torch.Tensor, dict[str, Any]]] = [
            ("dense_block", dense_block, {}),
            ("deferred_all_groups", deferred_all, {}),
            ("sequential_all_groups", sequential_all, {}),
        ]
        for top_k in topks:
            variants.append(
                (
                    f"deferred_top{top_k}_both",
                    _deferred_grouped_block_topk(
                        block,
                        input_state,
                        attn_group_ids=attn_group_ids,
                        ffn_group_ids=ffn_group_ids,
                        head_groups=config.model.head_groups,
                        ffn_groups=config.model.ffn_groups,
                        attn_top_k=min(top_k, config.model.head_groups),
                        ffn_top_k=min(top_k, config.model.ffn_groups),
                    ),
                    {
                        "attn_top_k": min(top_k, config.model.head_groups),
                        "ffn_top_k": min(top_k, config.model.ffn_groups),
                    },
                )
            )
            variants.append(
                (
                    f"deferred_dense_attn_top{top_k}_ffn",
                    (
                        lambda u, tk: u
                        + _ffn_topk_output(
                            block,
                            block.norm2(u),
                            ffn_group_ids,
                            group_count=config.model.ffn_groups,
                            top_k=min(tk, config.model.ffn_groups),
                        )[0]
                    )(
                        input_state + block.attn(block.norm1(input_state)),
                        top_k,
                    ),
                    {
                        "attn_top_k": config.model.head_groups,
                        "ffn_top_k": min(top_k, config.model.ffn_groups),
                    },
                )
            )
        for name, value, extra in variants:
            metrics = _block_oracle_metrics(
                dense,
                value,
                target,
                targets,
                state_layer=args.layer,
                temperature=args.temperature,
            )
            rows.append(
                {
                    "batch": batch_idx + 1,
                    "variant": name,
                    **extra,
                    **metrics,
                }
            )
        max_abs_diffs["dense_block_vs_target"] = max(
            max_abs_diffs.get("dense_block_vs_target", 0.0),
            float((dense_block - target).abs().max().detach().cpu()),
        )
        max_abs_diffs["deferred_all_vs_target"] = max(
            max_abs_diffs.get("deferred_all_vs_target", 0.0),
            float((deferred_all - target).abs().max().detach().cpu()),
        )
    averaged: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = averaged.setdefault(
            row["variant"],
            {
                "variant": row["variant"],
                "count": 0,
                "hidden_cosine": 0.0,
                "hidden_mse": 0.0,
                "suffix_kl": 0.0,
                "nll_per_token": 0.0,
                "target_nll_per_token": 0.0,
            },
        )
        for key in ("attn_top_k", "ffn_top_k"):
            if key in row:
                bucket[key] = row[key]
        bucket["count"] += 1
        for key in ("hidden_cosine", "hidden_mse", "suffix_kl", "nll_per_token", "target_nll_per_token"):
            bucket[key] += float(row[key])
    summary_rows = []
    for bucket in averaged.values():
        count = max(int(bucket.pop("count")), 1)
        for key in ("hidden_cosine", "hidden_mse", "suffix_kl", "nll_per_token", "target_nll_per_token"):
            bucket[key] /= count
        summary_rows.append(bucket)
    order = {
        "dense_block": 0,
        "deferred_all_groups": 1,
        "sequential_all_groups": 2,
    }
    summary_rows.sort(key=lambda item: (order.get(item["variant"], 10), item.get("ffn_top_k", 0), item["variant"]))
    report = {
        "mode": "deferred_grouped_block_oracle",
        "checkpoint": args.dense_checkpoint,
        "layer": args.layer,
        "input_layer": args.layer - 1,
        "num_batches": args.num_batches,
        "eval_tokens": args.num_batches * config.training.batch_size * config.training.seq_len,
        "head_groups": config.model.head_groups,
        "ffn_groups": config.model.ffn_groups,
        "topk": topks,
        "max_abs_diffs": max_abs_diffs,
        "rows": summary_rows,
        "per_batch_rows": rows if args.include_per_batch else None,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _init_full_stack_oracle_bucket(
    name: str,
    *,
    top_k: int | None,
    selector: str | None,
    layer_count: int,
) -> dict[str, Any]:
    bucket: dict[str, Any] = {
        "variant": name,
        "top_k": top_k,
        "selector": selector,
        "tokens": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "kl_sum": 0.0,
        "final_hidden_cosine_sum": 0.0,
        "coeff_count": 0.0,
        "coeff_sum": 0.0,
        "coeff_abs_sum": 0.0,
        "coeff_sq_sum": 0.0,
    }
    for layer in range(1, layer_count + 1):
        bucket[f"layer_{layer}_hidden_cosine_sum"] = 0.0
    return bucket


def _finish_full_stack_oracle_bucket(bucket: dict[str, Any], *, layer_count: int) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    if bucket["top_k"] is not None:
        row["top_k"] = bucket["top_k"]
        row["attn_top_k"] = bucket["top_k"]
        row["ffn_top_k"] = bucket["top_k"]
    if bucket["selector"] is not None:
        row["selector"] = bucket["selector"]
    coeff_count = float(bucket.get("coeff_count", 0.0))
    if coeff_count > 0:
        coeff_mean = float(bucket["coeff_sum"]) / coeff_count
        coeff_sq_mean = float(bucket["coeff_sq_sum"]) / coeff_count
        row["coeff_abs_mean"] = float(bucket["coeff_abs_sum"]) / coeff_count
        row["coeff_mean"] = coeff_mean
        row["coeff_variance"] = max(0.0, coeff_sq_mean - coeff_mean * coeff_mean)
    row["layer_hidden_cosine"] = {
        str(layer): bucket[f"layer_{layer}_hidden_cosine_sum"] / samples
        for layer in range(1, layer_count + 1)
    }
    return row


def cmd_deferred_grouped_full_stack_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    topks = _topk_list(args.topk, [2, 4, 8])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    selectors = list(args.selectors or ["norm", "omp_unit", "omp_ls"])
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    attn_group_ids = _contiguous_group_ids(config.model.n_heads, config.model.head_groups)
    ffn_group_ids = _contiguous_group_ids(config.model.d_ff, config.model.ffn_groups)
    variants: list[tuple[str, int | None, str | None]] = [("deferred_all", None, None)]
    for selector in selectors:
        for top_k in topks:
            variants.append((f"{selector}_top{top_k}", top_k, selector))
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_full_stack_oracle_bucket(
            "dense", top_k=None, selector=None, layer_count=config.model.n_dense_layers
        )
    }
    for name, top_k, selector in variants:
        buckets[name] = _init_full_stack_oracle_bucket(
            name, top_k=top_k, selector=selector, layer_count=config.model.n_dense_layers
        )
    max_abs_diffs: dict[str, float] = {"deferred_all_logits_vs_dense": 0.0}
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True, return_states=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.states is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for layer in range(1, config.model.n_dense_layers + 1):
            dense_bucket[f"layer_{layer}_hidden_cosine_sum"] += float(sample_count)
        for name, top_k, selector in variants:
            logits, states, aux = _full_stack_deferred_logits_and_states(
                dense,
                tokens,
                attn_group_ids=attn_group_ids,
                ffn_group_ids=ffn_group_ids,
                head_groups=config.model.head_groups,
                ffn_groups=config.model.ffn_groups,
                top_k=top_k,
                selector=selector or "norm",
                ridge=args.ridge,
            )
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            hidden = dense.final_norm(states[config.model.n_dense_layers])
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket = buckets[name]
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            _add_coeff_aux(bucket, aux)
            for layer in range(1, config.model.n_dense_layers + 1):
                layer_cos = F.cosine_similarity(
                    states[layer].float().flatten(1),
                    dense_out.meta.states[layer].float().flatten(1),
                    dim=-1,
                ).sum()
                bucket[f"layer_{layer}_hidden_cosine_sum"] += float(layer_cos.detach().cpu())
            if top_k is None:
                max_abs_diffs["deferred_all_logits_vs_dense"] = max(
                    max_abs_diffs["deferred_all_logits_vs_dense"],
                    float((logits - dense_logits).abs().max().detach().cpu()),
                )
                for layer in range(1, config.model.n_dense_layers + 1):
                    key = f"deferred_all_layer_{layer}_vs_dense"
                    max_abs_diffs[key] = max(
                        max_abs_diffs.get(key, 0.0),
                        float((states[layer] - dense_out.meta.states[layer]).abs().max().detach().cpu()),
                    )
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "full_stack_oracle_batch", "batch": batch_idx + 1}))
    rows = [
        _finish_full_stack_oracle_bucket(
            buckets[name], layer_count=config.model.n_dense_layers
        )
        for name in ["dense", *[name for name, _, _ in variants]]
    ]
    report = {
        "mode": "deferred_grouped_full_stack_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "head_groups": config.model.head_groups,
        "ffn_groups": config.model.ffn_groups,
        "topk": topks,
        "selectors": selectors,
        "temperature": args.temperature,
        "ridge": args.ridge,
        "max_abs_diffs": max_abs_diffs,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _init_neuron_oracle_bucket(
    name: str,
    *,
    selector: str | None,
    requested_k: int | None,
    active_neurons: int | None,
) -> dict[str, Any]:
    return {
        "variant": name,
        "selector": selector,
        "requested_k": requested_k,
        "active_neurons_per_token": active_neurons,
        "tokens": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "kl_sum": 0.0,
        "final_hidden_cosine_sum": 0.0,
        "coeff_count": 0.0,
        "coeff_sum": 0.0,
        "coeff_abs_sum": 0.0,
        "coeff_sq_sum": 0.0,
        "overlap_jaccard_sum": 0.0,
        "overlap_recall_sum": 0.0,
        "overlap_count": 0,
    }


def _finish_neuron_oracle_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    if bucket["selector"] is not None:
        row["selector"] = bucket["selector"]
    if bucket["requested_k"] is not None:
        row["requested_k"] = bucket["requested_k"]
    if bucket["active_neurons_per_token"] is not None:
        row["active_neurons_per_token"] = bucket["active_neurons_per_token"]
    coeff_count = float(bucket.get("coeff_count", 0.0))
    if coeff_count > 0:
        coeff_mean = float(bucket["coeff_sum"]) / coeff_count
        coeff_sq_mean = float(bucket["coeff_sq_sum"]) / coeff_count
        row["coeff_abs_mean"] = float(bucket["coeff_abs_sum"]) / coeff_count
        row["coeff_mean"] = coeff_mean
        row["coeff_variance"] = max(0.0, coeff_sq_mean - coeff_mean * coeff_mean)
    overlap_count = max(int(bucket.get("overlap_count", 0)), 1)
    if bucket.get("overlap_count", 0):
        row["selection_overlap_jaccard"] = bucket["overlap_jaccard_sum"] / overlap_count
        row["selection_overlap_recall"] = bucket["overlap_recall_sum"] / overlap_count
    return row


class LowRankNeuronSelector(torch.nn.Module):
    def __init__(self, *, n_layers: int, d_model: int, d_ff: int, rank: int):
        super().__init__()
        scale_q = 1.0 / math.sqrt(d_model)
        scale_k = 1.0 / math.sqrt(max(rank, 1))
        self.q = torch.nn.Parameter(torch.randn(n_layers, d_model, rank) * scale_q)
        self.keys = torch.nn.Parameter(torch.randn(n_layers, d_ff, rank) * scale_k)
        self.bias = torch.nn.Parameter(torch.zeros(n_layers, d_ff))

    def forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        q = torch.matmul(x, self.q[layer_idx])
        return torch.matmul(q, self.keys[layer_idx].t()) + self.bias[layer_idx]


class FactorUnionNeuronSelector(torch.nn.Module):
    def __init__(self, *, n_layers: int, d_model: int, d_ff: int, rank: int):
        super().__init__()
        self.up = LowRankNeuronSelector(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            rank=rank,
        )
        self.gate = LowRankNeuronSelector(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            rank=rank,
        )
        self.product = LowRankNeuronSelector(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            rank=rank,
        )

    def forward(self, kind: str, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        if kind == "up":
            return self.up(layer_idx, x)
        if kind == "gate":
            return self.gate(layer_idx, x)
        if kind == "product":
            return self.product(layer_idx, x)
        raise ValueError(f"unsupported factor selector kind: {kind}")


def _ffn_neuron_scores(block: torch.nn.Module, normed: torch.Tensor) -> torch.Tensor:
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = up * F.silu(gate)
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    return z.detach().abs() * wd_norm


def _selector_loss_for_scores(
    logits: torch.Tensor,
    scores: torch.Tensor,
    *,
    label_k: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    label_k = min(max(1, int(label_k)), scores.shape[-1])
    top_ids = torch.topk(scores, k=label_k, dim=-1).indices
    labels = torch.zeros_like(scores)
    labels.scatter_(-1, top_ids, 1.0)
    pos_weight = torch.tensor(
        (scores.shape[-1] - label_k) / max(label_k, 1),
        dtype=logits.dtype,
        device=logits.device,
    )
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    target = torch.log1p(scores)
    target = (target - target.mean(dim=-1, keepdim=True)) / target.std(dim=-1, keepdim=True).clamp_min(
        1e-4
    )
    pred = (logits - logits.mean(dim=-1, keepdim=True)) / logits.std(dim=-1, keepdim=True).clamp_min(
        1e-4
    )
    mse = F.mse_loss(pred, target)
    loss = bce + 0.1 * mse
    return loss, {
        "selector_bce": float(bce.detach().cpu()),
        "selector_score_mse": float(mse.detach().cpu()),
    }


@torch.no_grad()
def _ffn_neuron_candidate_rerank_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    selector: LowRankNeuronSelector,
    *,
    layer_idx: int,
    candidate_m: int,
    top_k: int,
    reranker: str = "norm",
    ridge: float = 1e-4,
    coeff_prior: float = 1.0,
    coeff_clamp: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    logits = selector(layer_idx, normed)
    hidden = logits.shape[-1]
    candidate_m = min(max(1, int(candidate_m)), hidden)
    top_k = min(max(1, int(top_k)), candidate_m)
    candidate_ids = torch.topk(logits, k=candidate_m, dim=-1).indices
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    z = (up * F.silu(gate)).reshape(-1, hidden)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    exact_scores = z.detach().abs() * wd_norm
    cand_flat = candidate_ids.reshape(-1, candidate_m)
    cand_scores = torch.gather(exact_scores, dim=-1, index=cand_flat)
    if reranker == "norm":
        local = torch.topk(cand_scores, k=top_k, dim=-1).indices
        selected_ids = torch.gather(cand_flat, dim=-1, index=local)
        approx_flat = _selected_neuron_output(z, wd_rows, selected_ids)
        coeffs = z.new_ones(selected_ids.shape)
    else:
        target = block.mlp.wd(z)
        selected_mask = torch.zeros(z.shape[0], candidate_m, device=z.device, dtype=torch.bool)
        residual = target
        selected_local: list[torch.Tensor] = []
        cand_z = torch.gather(z, dim=-1, index=cand_flat)
        cand_w = wd_rows[cand_flat]
        cand_norm = (cand_z.detach().float().square() * cand_w.detach().float().square().sum(dim=-1)).clamp_min(
            1e-12
        )
        approx_flat = target.new_zeros(target.shape)
        coeffs = z.new_zeros(z.shape[0], 0)
        for _ in range(top_k):
            dot = torch.einsum("nd,nmd->nm", residual.float(), cand_w.float()).to(z.dtype) * cand_z
            if reranker == "omp_unit":
                scores = 2.0 * dot - cand_norm.to(dot.dtype)
            elif reranker in {"omp_ls", "omp_ridge1"}:
                scores = dot.square() / cand_norm.to(dot.dtype)
            else:
                raise ValueError(f"unsupported candidate reranker: {reranker}")
            scores = scores.masked_fill(selected_mask, -torch.inf)
            chosen_local = scores.argmax(dim=-1)
            selected_mask.scatter_(-1, chosen_local.unsqueeze(-1), True)
            selected_local.append(chosen_local)
            piece_ids = torch.gather(cand_flat, dim=-1, index=chosen_local.unsqueeze(-1)).squeeze(-1)
            piece = z.gather(dim=-1, index=piece_ids.unsqueeze(-1)) * wd_rows[piece_ids]
            if reranker == "omp_unit":
                approx_flat = approx_flat + piece
                residual = residual - piece
            else:
                local_ids = torch.stack(selected_local, dim=-1)
                selected_ids_step = torch.gather(cand_flat, dim=-1, index=local_ids)
                selected = (
                    z.gather(dim=-1, index=selected_ids_step).unsqueeze(-1)
                    * wd_rows[selected_ids_step]
                )
                if reranker == "omp_ls":
                    approx_flat, coeffs = _solve_neuron_ls_output(
                        z,
                        wd_rows,
                        target,
                        selected_ids_step,
                        ridge=ridge,
                    )
                else:
                    coeffs = _solve_neuron_coefficients_prior(
                        selected,
                        target,
                        ridge=ridge,
                        prior=coeff_prior,
                        clamp=coeff_clamp,
                    )
                    approx_flat = (selected.float() * coeffs.float().unsqueeze(-1)).sum(dim=1).to(
                        target.dtype
                    )
                residual = target - approx_flat
        if reranker == "omp_unit":
            selected_ids = torch.gather(cand_flat, dim=-1, index=torch.stack(selected_local, dim=-1))
            coeffs = z.new_ones(selected_ids.shape)
        else:
            selected_ids = torch.gather(cand_flat, dim=-1, index=torch.stack(selected_local, dim=-1))
    approx = approx_flat.view(*original_shape, -1)
    true_ids = torch.topk(exact_scores, k=top_k, dim=-1).indices
    cand_mask = F.one_hot(cand_flat, num_classes=hidden).sum(dim=1).bool()
    true_mask = F.one_hot(true_ids, num_classes=hidden).sum(dim=1).bool()
    selected_mask = F.one_hot(selected_ids, num_classes=hidden).sum(dim=1).bool()
    candidate_recall = (cand_mask & true_mask).sum(dim=-1).float() / max(top_k, 1)
    selected_recall = (selected_mask & true_mask).sum(dim=-1).float() / max(top_k, 1)
    selected_score = torch.gather(exact_scores, dim=-1, index=selected_ids).sum(dim=-1)
    true_score = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    return approx, {
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "selected_recall_sum": float(selected_recall.sum().detach().cpu()),
        "score_retention_sum": float((selected_score / true_score).sum().detach().cpu()),
        "selection_count": int(z.shape[0]),
        **_coeff_aux(coeffs),
    }


@torch.no_grad()
def _ffn_neuron_factor_union_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    *,
    top_k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    gate_act = F.silu(gate)
    z = (up * gate_act).reshape(-1, block.mlp.wd.in_features)
    up_flat = up.reshape_as(z)
    gate_flat = gate_act.reshape_as(z)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    hidden = z.shape[-1]
    top_k = min(max(1, int(top_k)), hidden)
    candidate_mask = torch.zeros(z.shape[0], hidden, device=z.device, dtype=torch.bool)

    def add_top(scores: torch.Tensor, count: int) -> None:
        if count <= 0:
            return
        ids = torch.topk(scores, k=min(int(count), hidden), dim=-1).indices
        candidate_mask.scatter_(-1, ids, True)

    exact_scores = z.detach().abs() * wd_norm
    add_top(up_flat.detach().abs() * wd_norm, up_m)
    add_top(gate_flat.detach().abs() * wd_norm, gate_m)
    add_top(exact_scores, product_m)
    # In pathological cases with tiny m values, ensure every token has at least k candidates.
    too_small = candidate_mask.sum(dim=-1) < top_k
    if bool(too_small.any()):
        fill_ids = torch.topk(exact_scores[too_small], k=top_k, dim=-1).indices
        candidate_mask[too_small].scatter_(-1, fill_ids, True)

    true_ids = torch.topk(exact_scores, k=top_k, dim=-1).indices
    true_mask = F.one_hot(true_ids, num_classes=hidden).sum(dim=1).bool()
    if reranker == "norm":
        masked_scores = exact_scores.masked_fill(~candidate_mask, -torch.inf)
        selected_ids = torch.topk(masked_scores, k=top_k, dim=-1).indices
        approx = _selected_neuron_output(z, wd_rows, selected_ids)
    elif reranker == "omp_unit":
        target = block.mlp.wd(z)
        residual = target
        selected_mask = torch.zeros_like(candidate_mask)
        selected: list[torch.Tensor] = []
        wd_norm_sq = wd_rows.detach().float().square().sum(dim=-1).to(z.device).clamp_min(1e-12)
        approx = target.new_zeros(target.shape)
        for _ in range(top_k):
            dot = (residual @ wd_rows.t()) * z
            norm_sq = z.detach().float().square().to(dot.dtype) * wd_norm_sq.to(dot.dtype)
            scores = (2.0 * dot - norm_sq).masked_fill(~candidate_mask | selected_mask, -torch.inf)
            chosen = scores.argmax(dim=-1)
            selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
            selected.append(chosen)
            piece = z.gather(dim=-1, index=chosen.unsqueeze(-1)) * wd_rows[chosen]
            approx = approx + piece
            residual = residual - piece
        selected_ids = torch.stack(selected, dim=-1)
    else:
        raise ValueError(f"unsupported factor-union reranker: {reranker}")
    selected_mask_final = F.one_hot(selected_ids, num_classes=hidden).sum(dim=1).bool()
    selected_score = torch.gather(exact_scores, dim=-1, index=selected_ids).sum(dim=-1)
    true_score = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    candidate_size = candidate_mask.sum(dim=-1).float()
    candidate_recall = (candidate_mask & true_mask).sum(dim=-1).float() / max(top_k, 1)
    selected_recall = (selected_mask_final & true_mask).sum(dim=-1).float() / max(top_k, 1)
    return approx.view(*original_shape, -1), {
        "candidate_size_sum": float(candidate_size.sum().detach().cpu()),
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "selected_recall_sum": float(selected_recall.sum().detach().cpu()),
        "score_retention_sum": float((selected_score / true_score).sum().detach().cpu()),
        "selection_count": int(z.shape[0]),
    }


@torch.no_grad()
def _ffn_neuron_learned_factor_union_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    selector: FactorUnionNeuronSelector,
    *,
    layer_idx: int,
    top_k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    gate_act = F.silu(gate)
    z = (up * gate_act).reshape(-1, block.mlp.wd.in_features)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    hidden = z.shape[-1]
    top_k = min(max(1, int(top_k)), hidden)
    flat_normed = normed.reshape(-1, normed.shape[-1])
    candidate_mask = torch.zeros(z.shape[0], hidden, device=z.device, dtype=torch.bool)

    def add_top(kind: str, count: int) -> None:
        if count <= 0:
            return
        logits = selector(kind, layer_idx, flat_normed)
        ids = torch.topk(logits, k=min(int(count), hidden), dim=-1).indices
        candidate_mask.scatter_(-1, ids, True)

    exact_scores = z.detach().abs() * wd_norm
    add_top("up", up_m)
    add_top("gate", gate_m)
    add_top("product", product_m)
    too_small = candidate_mask.sum(dim=-1) < top_k
    if bool(too_small.any()):
        fill_ids = torch.topk(exact_scores[too_small], k=top_k, dim=-1).indices
        candidate_mask[too_small].scatter_(-1, fill_ids, True)
    true_ids = torch.topk(exact_scores, k=top_k, dim=-1).indices
    true_mask = F.one_hot(true_ids, num_classes=hidden).sum(dim=1).bool()
    if reranker == "norm":
        masked_scores = exact_scores.masked_fill(~candidate_mask, -torch.inf)
        selected_ids = torch.topk(masked_scores, k=top_k, dim=-1).indices
        approx = _selected_neuron_output(z, wd_rows, selected_ids)
    elif reranker == "omp_unit":
        target = block.mlp.wd(z)
        residual = target
        selected_mask = torch.zeros_like(candidate_mask)
        selected: list[torch.Tensor] = []
        wd_norm_sq = wd_rows.detach().float().square().sum(dim=-1).to(z.device).clamp_min(1e-12)
        approx = target.new_zeros(target.shape)
        for _ in range(top_k):
            dot = (residual @ wd_rows.t()) * z
            norm_sq = z.detach().float().square().to(dot.dtype) * wd_norm_sq.to(dot.dtype)
            scores = (2.0 * dot - norm_sq).masked_fill(~candidate_mask | selected_mask, -torch.inf)
            chosen = scores.argmax(dim=-1)
            selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
            selected.append(chosen)
            piece = z.gather(dim=-1, index=chosen.unsqueeze(-1)) * wd_rows[chosen]
            approx = approx + piece
            residual = residual - piece
        selected_ids = torch.stack(selected, dim=-1)
    else:
        raise ValueError(f"unsupported learned factor-union reranker: {reranker}")
    selected_mask_final = F.one_hot(selected_ids, num_classes=hidden).sum(dim=1).bool()
    selected_score = torch.gather(exact_scores, dim=-1, index=selected_ids).sum(dim=-1)
    true_score = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    candidate_size = candidate_mask.sum(dim=-1).float()
    candidate_recall = (candidate_mask & true_mask).sum(dim=-1).float() / max(top_k, 1)
    selected_recall = (selected_mask_final & true_mask).sum(dim=-1).float() / max(top_k, 1)
    return approx.view(*original_shape, -1), {
        "candidate_size_sum": float(candidate_size.sum().detach().cpu()),
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "selected_recall_sum": float(selected_recall.sum().detach().cpu()),
        "score_retention_sum": float((selected_score / true_score).sum().detach().cpu()),
        "selection_count": int(z.shape[0]),
    }


def _build_svd_factor_cache(
    dense: DenseModel,
    *,
    max_rank: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    factors: list[dict[str, torch.Tensor]] = []
    for block in dense.blocks:
        weight = block.mlp.wug.weight.detach().float().cpu()
        up_w, gate_w = weight.chunk(2, dim=0)
        layer: dict[str, torch.Tensor] = {}
        for name, mat in (("up", up_w.t().contiguous()), ("gate", gate_w.t().contiguous())):
            u, s, vh = torch.linalg.svd(mat, full_matrices=False)
            rank = min(max_rank, s.numel())
            layer[f"{name}_a"] = (u[:, :rank] * s[:rank]).to(device=device)
            layer[f"{name}_b"] = vh[:rank, :].to(device=device)
        factors.append(layer)
    return factors


def _cluster_assignments_from_features(
    features: torch.Tensor,
    *,
    cluster_count: int,
    iters: int,
) -> torch.Tensor:
    """Small cosine k-means used only by the cluster-pool oracle."""

    token_count = features.shape[0]
    cluster_count = min(max(1, int(cluster_count)), max(1, token_count))
    if cluster_count == 1 or token_count == 1:
        return torch.zeros(token_count, device=features.device, dtype=torch.long)
    x = F.normalize(features.float(), dim=-1)
    init_idx = torch.linspace(
        0,
        token_count - 1,
        steps=cluster_count,
        device=features.device,
    ).round().long()
    centers = x.index_select(0, init_idx).contiguous()
    assignments = torch.zeros(token_count, device=features.device, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        assignments = (x @ centers.t()).argmax(dim=-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assignments, minlength=cluster_count).to(new_centers.dtype)
        new_centers.index_add_(0, assignments, x)
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty].unsqueeze(-1).clamp_min(1.0)
        new_centers[~nonempty] = centers[~nonempty]
        centers = F.normalize(new_centers, dim=-1)
    return assignments


def _fit_cluster_centers_from_features(
    features: torch.Tensor,
    *,
    cluster_count: int,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit small cosine k-means centers for static cluster-pool codebooks."""

    token_count = features.shape[0]
    cluster_count = min(max(1, int(cluster_count)), max(1, token_count))
    x = F.normalize(features.float(), dim=-1)
    if cluster_count == 1 or token_count == 1:
        assignments = torch.zeros(token_count, device=features.device, dtype=torch.long)
        centers = F.normalize(x.mean(dim=0, keepdim=True), dim=-1)
        return assignments, centers
    init_idx = torch.linspace(
        0,
        token_count - 1,
        steps=cluster_count,
        device=features.device,
    ).round().long()
    centers = x.index_select(0, init_idx).contiguous()
    assignments = torch.zeros(token_count, device=features.device, dtype=torch.long)
    for _ in range(max(1, int(iters))):
        assignments = (x @ centers.t()).argmax(dim=-1)
        new_centers = torch.zeros_like(centers)
        counts = torch.bincount(assignments, minlength=cluster_count).to(new_centers.dtype)
        new_centers.index_add_(0, assignments, x)
        nonempty = counts > 0
        new_centers[nonempty] = new_centers[nonempty] / counts[nonempty].unsqueeze(-1).clamp_min(1.0)
        new_centers[~nonempty] = centers[~nonempty]
        centers = F.normalize(new_centers, dim=-1)
    return assignments, centers


def _aggregate_cluster_scores(
    scores: torch.Tensor,
    assignments: torch.Tensor,
    *,
    cluster_count: int,
    aggregation: str,
) -> torch.Tensor:
    cluster_count = max(1, int(cluster_count))
    if aggregation == "mean":
        agg = scores.new_zeros(cluster_count, scores.shape[-1])
        agg.index_add_(0, assignments, scores)
        counts = torch.bincount(assignments, minlength=cluster_count).to(scores.dtype).clamp_min(1.0)
        return agg / counts.unsqueeze(-1)
    if aggregation == "max":
        rows = []
        for cluster_idx in range(cluster_count):
            mask = assignments == cluster_idx
            if bool(mask.any()):
                rows.append(scores[mask].amax(dim=0))
            else:
                rows.append(scores.new_zeros(scores.shape[-1]))
        return torch.stack(rows, dim=0)
    raise ValueError(f"unsupported cluster score aggregation: {aggregation}")


@torch.no_grad()
def _svd_cluster_features_and_scores(
    block: torch.nn.Module,
    normed: torch.Tensor,
    factors: dict[str, torch.Tensor],
    *,
    rank: int,
    score_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = normed.reshape(-1, normed.shape[-1])
    hidden = block.mlp.wd.in_features
    rank = min(max(1, int(rank)), factors["up_a"].shape[1], factors["gate_a"].shape[1])
    q_up = flat @ factors["up_a"][:, :rank]
    q_gate = flat @ factors["gate_a"][:, :rank]
    up_hat = q_up @ factors["up_b"][:rank, :]
    gate_hat = q_gate @ factors["gate_b"][:rank, :]
    gate_hat_act = F.silu(gate_hat)
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(device=flat.device, dtype=flat.dtype)
    up_scores = up_hat.detach().abs() * wd_norm
    gate_scores = gate_hat_act.detach().abs() * wd_norm
    product_scores = (up_hat * gate_hat_act).detach().abs() * wd_norm
    if score_mode == "sum":
        pool_scores = up_scores + gate_scores + product_scores
    elif score_mode == "upgate":
        pool_scores = up_scores + gate_scores
    elif score_mode == "product":
        pool_scores = product_scores
    else:
        raise ValueError(f"unsupported cluster pool score mode: {score_mode}")
    if pool_scores.shape[-1] != hidden:
        raise RuntimeError("cluster score width does not match FFN hidden width")
    return torch.cat([q_up, q_gate], dim=-1), pool_scores


@torch.no_grad()
def _ffn_neuron_svd_cluster_pool_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    factors: dict[str, torch.Tensor],
    *,
    rank: int,
    cluster_count: int,
    candidate_m: int,
    reference_k: int,
    score_mode: str,
    aggregation: str,
    cluster_iters: int,
    profile: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    flat = normed.reshape(-1, normed.shape[-1])
    hidden = block.mlp.wd.in_features
    rank = min(max(1, int(rank)), factors["up_a"].shape[1], factors["gate_a"].shape[1])
    candidate_m = min(max(1, int(candidate_m)), hidden)
    reference_k = min(max(1, int(reference_k)), hidden)
    cluster_count = min(max(1, int(cluster_count)), flat.shape[0])

    start = time.perf_counter()
    features, pool_scores = _svd_cluster_features_and_scores(
        block,
        normed,
        factors,
        rank=rank,
        score_mode=score_mode,
    )
    wd_rows = block.mlp.wd.weight.t().contiguous()
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(device=flat.device, dtype=flat.dtype)

    assignments = _cluster_assignments_from_features(
        features,
        cluster_count=cluster_count,
        iters=cluster_iters,
    )
    actual_cluster_count = int(assignments.max().detach().cpu()) + 1 if assignments.numel() else 1
    cluster_count = max(cluster_count, actual_cluster_count)
    aggregate_scores = _aggregate_cluster_scores(
        pool_scores.float(),
        assignments,
        cluster_count=cluster_count,
        aggregation=aggregation,
    )
    candidate_ids = torch.topk(
        aggregate_scores,
        k=min(candidate_m, aggregate_scores.shape[-1]),
        dim=-1,
    ).indices

    out = flat.new_zeros(flat.shape[0], flat.shape[-1])
    up_weight, gate_weight = block.mlp.wug.weight.detach().chunk(2, dim=0)
    cluster_sizes = torch.bincount(assignments, minlength=cluster_count)
    for cluster_idx in range(cluster_count):
        token_idx = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten()
        if token_idx.numel() == 0:
            continue
        ids = candidate_ids[cluster_idx]
        x_cluster = flat.index_select(0, token_idx)
        up = x_cluster @ up_weight.index_select(0, ids).t().contiguous()
        gate = x_cluster @ gate_weight.index_select(0, ids).t().contiguous()
        z = up * F.silu(gate)
        y_cluster = z @ wd_rows.index_select(0, ids).contiguous()
        if out.device.type == "mps":
            # MPS does not implement aten::index_copy.out. This path is only
            # used by local oracle diagnostics; CUDA training kernels use the
            # packed/preindexed execution path instead.
            out[token_idx] = y_cluster
        else:
            out.index_copy_(0, token_idx, y_cluster)

    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    exact_scores = (up * F.silu(gate)).reshape(flat.shape[0], hidden).detach().abs() * wd_norm
    true_ids = torch.topk(exact_scores, k=reference_k, dim=-1).indices
    token_candidates = candidate_ids.index_select(0, assignments)
    candidate_hits = true_ids.unsqueeze(-1).eq(token_candidates.unsqueeze(1)).any(dim=-1)
    candidate_recall = candidate_hits.float().sum(dim=-1) / max(reference_k, 1)
    candidate_scores = torch.gather(exact_scores, dim=-1, index=token_candidates).sum(dim=-1)
    true_scores = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    nonempty = cluster_sizes[cluster_sizes > 0].float()
    elapsed = time.perf_counter() - start
    return out.view(*original_shape, -1), {
        "candidate_size_sum": float(candidate_m * flat.shape[0]),
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "score_retention_sum": float((candidate_scores / true_scores).sum().detach().cpu()),
        "selection_count": int(flat.shape[0]),
        "cluster_count_sum": float(cluster_count),
        "nonempty_cluster_count_sum": float(nonempty.numel()),
        "empty_cluster_count_sum": float(cluster_count - nonempty.numel()),
        "max_cluster_size_sum": float(nonempty.max().detach().cpu()) if nonempty.numel() else 0.0,
        "min_cluster_size_sum": float(nonempty.min().detach().cpu()) if nonempty.numel() else 0.0,
        "mean_cluster_size_sum": float(nonempty.mean().detach().cpu()) if nonempty.numel() else 0.0,
        "cluster_imbalance_sum": float((nonempty.max() / nonempty.mean().clamp_min(1.0)).detach().cpu())
        if nonempty.numel()
        else 0.0,
        "cluster_metric_count": 1.0,
        "cluster_pool_ffn_flop_ratio_sum": float(candidate_m / hidden),
        "cluster_pool_exec_seconds": elapsed if profile else 0.0,
    }


@torch.no_grad()
def _ffn_neuron_svd_factor_union_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    factors: dict[str, torch.Tensor],
    *,
    rank: int,
    top_k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    gate_act = F.silu(gate)
    z = (up * gate_act).reshape(-1, block.mlp.wd.in_features)
    flat_normed = normed.reshape(-1, normed.shape[-1])
    wd_rows = block.mlp.wd.weight.t().contiguous()
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
    hidden = z.shape[-1]
    rank = min(max(1, int(rank)), factors["up_a"].shape[1], factors["gate_a"].shape[1])
    top_k = min(max(1, int(top_k)), hidden)
    up_hat = (flat_normed @ factors["up_a"][:, :rank]) @ factors["up_b"][:rank, :]
    gate_hat = (flat_normed @ factors["gate_a"][:, :rank]) @ factors["gate_b"][:rank, :]
    candidate_mask = torch.zeros(z.shape[0], hidden, device=z.device, dtype=torch.bool)

    def add_top(scores: torch.Tensor, count: int) -> None:
        if count <= 0:
            return
        ids = torch.topk(scores, k=min(int(count), hidden), dim=-1).indices
        candidate_mask.scatter_(-1, ids, True)

    exact_scores = z.detach().abs() * wd_norm
    add_top(up_hat.detach().abs() * wd_norm, up_m)
    add_top(F.silu(gate_hat).detach().abs() * wd_norm, gate_m)
    add_top((up_hat * F.silu(gate_hat)).detach().abs() * wd_norm, product_m)
    too_small = candidate_mask.sum(dim=-1) < top_k
    if bool(too_small.any()):
        fill_ids = torch.topk(exact_scores[too_small], k=top_k, dim=-1).indices
        candidate_mask[too_small].scatter_(-1, fill_ids, True)
    true_ids = torch.topk(exact_scores, k=top_k, dim=-1).indices
    true_mask = F.one_hot(true_ids, num_classes=hidden).sum(dim=1).bool()
    if reranker == "norm":
        masked_scores = exact_scores.masked_fill(~candidate_mask, -torch.inf)
        selected_ids = torch.topk(masked_scores, k=top_k, dim=-1).indices
        approx = _selected_neuron_output(z, wd_rows, selected_ids)
    elif reranker == "omp_unit":
        target = block.mlp.wd(z)
        residual = target
        selected_mask = torch.zeros_like(candidate_mask)
        selected: list[torch.Tensor] = []
        wd_norm_sq = wd_rows.detach().float().square().sum(dim=-1).to(z.device).clamp_min(1e-12)
        approx = target.new_zeros(target.shape)
        for _ in range(top_k):
            dot = (residual @ wd_rows.t()) * z
            norm_sq = z.detach().float().square().to(dot.dtype) * wd_norm_sq.to(dot.dtype)
            scores = (2.0 * dot - norm_sq).masked_fill(~candidate_mask | selected_mask, -torch.inf)
            chosen = scores.argmax(dim=-1)
            selected_mask.scatter_(-1, chosen.unsqueeze(-1), True)
            selected.append(chosen)
            piece = z.gather(dim=-1, index=chosen.unsqueeze(-1)) * wd_rows[chosen]
            approx = approx + piece
            residual = residual - piece
        selected_ids = torch.stack(selected, dim=-1)
    else:
        raise ValueError(f"unsupported SVD factor-union reranker: {reranker}")
    selected_mask_final = F.one_hot(selected_ids, num_classes=hidden).sum(dim=1).bool()
    selected_score = torch.gather(exact_scores, dim=-1, index=selected_ids).sum(dim=-1)
    true_score = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    candidate_size = candidate_mask.sum(dim=-1).float()
    candidate_recall = (candidate_mask & true_mask).sum(dim=-1).float() / max(top_k, 1)
    selected_recall = (selected_mask_final & true_mask).sum(dim=-1).float() / max(top_k, 1)
    return approx.view(*original_shape, -1), {
        "candidate_size_sum": float(candidate_size.sum().detach().cpu()),
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "selected_recall_sum": float(selected_recall.sum().detach().cpu()),
        "score_retention_sum": float((selected_score / true_score).sum().detach().cpu()),
        "selection_count": int(z.shape[0]),
    }


@torch.no_grad()
def _full_stack_sparse_ffn_logits_and_states(
    dense: DenseModel,
    tokens: torch.Tensor,
    *,
    mode: str,
    requested_k: int,
    ffn_group_ids: torch.Tensor,
    ffn_groups: int,
    selector: str = "norm",
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux: dict[str, float] = {}
    prev_ids: torch.Tensor | None = None
    ffn_group_size = dense.config.d_ff // ffn_groups
    for block in dense.blocks:
        u = x + block.attn(block.norm1(x))
        normed = block.norm2(u)
        if mode == "group":
            group_top_k = min(ffn_groups, max(1, math.ceil(requested_k / ffn_group_size)))
            ffn, _ = _ffn_topk_output(
                block,
                normed,
                ffn_group_ids,
                group_count=ffn_groups,
                top_k=group_top_k,
            )
            ids = None
        elif mode == "neuron":
            ffn, coeff_aux, ids = _ffn_neuron_sparse_output(
                block,
                normed,
                top_k=requested_k,
                selector=selector,
                ridge=ridge,
            )
            _add_coeff_aux(aux, coeff_aux)
        else:
            raise ValueError(f"unsupported sparse FFN oracle mode: {mode}")
        x = u + ffn
        if mode == "neuron" and ids is not None:
            if prev_ids is not None:
                jaccard, recall, count = _selection_jaccard(
                    prev_ids,
                    ids,
                    universe=dense.config.d_ff,
                )
                aux["overlap_jaccard_sum"] = aux.get("overlap_jaccard_sum", 0.0) + jaccard
                aux["overlap_recall_sum"] = aux.get("overlap_recall_sum", 0.0) + recall
                aux["overlap_count"] = aux.get("overlap_count", 0.0) + count
            prev_ids = ids
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


def cmd_deferred_neuron_full_stack_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ks = _topk_list(args.k, [16, 32, 64, 128])
    selectors = list(args.selectors or ["norm", "omp_unit", "omp_ls"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    ffn_group_ids = _contiguous_group_ids(config.model.d_ff, config.model.ffn_groups)
    group_size = config.model.d_ff // config.model.ffn_groups
    variants: list[tuple[str, str | None, str, int, int]] = []
    for k in ks:
        group_top_k = min(config.model.ffn_groups, max(1, math.ceil(k / group_size)))
        variants.append((f"group_norm_k{k}", "group_norm", "group", k, group_top_k * group_size))
    for selector in selectors:
        for k in ks:
            variants.append((f"neuron_{selector}_k{k}", selector, "neuron", k, min(k, config.model.d_ff)))
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_neuron_oracle_bucket(
            "dense",
            selector=None,
            requested_k=None,
            active_neurons=None,
        )
    }
    for name, selector, _, k, active_neurons in variants:
        buckets[name] = _init_neuron_oracle_bucket(
            name,
            selector=selector,
            requested_k=k,
            active_neurons=active_neurons,
        )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for name, selector, mode, k, _ in variants:
            logits, hidden, aux = _full_stack_sparse_ffn_logits_and_states(
                dense,
                tokens,
                mode=mode,
                requested_k=k,
                ffn_group_ids=ffn_group_ids,
                ffn_groups=config.model.ffn_groups,
                selector=selector if selector is not None and selector != "group_norm" else "norm",
                ridge=args.ridge,
            )
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket = buckets[name]
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            _add_coeff_aux(bucket, aux)
            for key in ("overlap_jaccard_sum", "overlap_recall_sum", "overlap_count"):
                bucket[key] = bucket.get(key, 0.0) + float(aux.get(key, 0.0))
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "neuron_full_stack_oracle_batch", "batch": batch_idx + 1}))
    rows = [_finish_neuron_oracle_bucket(buckets["dense"])]
    rows.extend(_finish_neuron_oracle_bucket(buckets[name]) for name, *_ in variants)
    report = {
        "mode": "deferred_neuron_full_stack_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "ffn_groups": config.model.ffn_groups,
        "ffn_group_size": group_size,
        "k": ks,
        "selectors": selectors,
        "temperature": args.temperature,
        "ridge": args.ridge,
        "attention": "dense",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _init_selector_oracle_bucket(
    name: str,
    *,
    variant_type: str,
    k: int | None,
    candidate_m: int | None,
    reranker: str | None = None,
    ridge: float | None = None,
    coeff_clamp: float | None = None,
) -> dict[str, Any]:
    return {
        "variant": name,
        "variant_type": variant_type,
        "k": k,
        "candidate_m": candidate_m,
        "reranker": reranker,
        "ridge": ridge,
        "coeff_clamp": coeff_clamp,
        "tokens": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "kl_sum": 0.0,
        "final_hidden_cosine_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0,
        "coeff_count": 0.0,
        "coeff_sum": 0.0,
        "coeff_abs_sum": 0.0,
        "coeff_sq_sum": 0.0,
    }


def _finish_selector_oracle_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "variant_type": bucket["variant_type"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    if bucket["k"] is not None:
        row["k"] = bucket["k"]
    if bucket["candidate_m"] is not None:
        row["candidate_m"] = bucket["candidate_m"]
    if bucket["reranker"] is not None:
        row["reranker"] = bucket["reranker"]
    if bucket["ridge"] is not None:
        row["ridge"] = bucket["ridge"]
    if bucket["coeff_clamp"] is not None:
        row["coeff_clamp"] = bucket["coeff_clamp"]
    count = int(bucket.get("selection_count", 0))
    if count > 0:
        row["candidate_recall"] = bucket["candidate_recall_sum"] / count
        row["selected_recall"] = bucket["selected_recall_sum"] / count
        row["score_retention"] = bucket["score_retention_sum"] / count
    coeff_count = float(bucket.get("coeff_count", 0.0))
    if coeff_count > 0:
        coeff_mean = float(bucket["coeff_sum"]) / coeff_count
        coeff_sq_mean = float(bucket["coeff_sq_sum"]) / coeff_count
        row["coeff_abs_mean"] = float(bucket["coeff_abs_sum"]) / coeff_count
        row["coeff_mean"] = coeff_mean
        row["coeff_variance"] = max(0.0, coeff_sq_mean - coeff_mean * coeff_mean)
    return row


def _train_lowrank_neuron_selector(
    dense: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    rank: int,
    label_k: int,
    train_batches: int,
    lr: float,
    weight_decay: float,
    train_tokens_per_batch: int | None,
    device: torch.device,
    progress: int,
) -> tuple[LowRankNeuronSelector, list[dict[str, float]]]:
    selector = LowRankNeuronSelector(
        n_layers=config.model.n_dense_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        rank=rank,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=weight_decay)
    iterator = streams.train_batches(config.training)
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(config.training.seed + 17)
    logs: list[dict[str, float]] = []
    dense.eval()
    selector.train()
    for batch_idx in range(max(0, int(train_batches))):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            x = dense.embed(tokens)
        total_loss_value = 0.0
        aux_sums = {"selector_bce": 0.0, "selector_score_mse": 0.0}
        for layer_idx, block in enumerate(dense.blocks):
            with torch.no_grad():
                u = x + block.attn(block.norm1(x))
                normed = block.norm2(u)
                up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
                z = up * F.silu(gate)
                wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(z.device)
                scores = z.detach().abs() * wd_norm
                next_x = u + block.mlp.wd(z)
            flat_normed = normed.detach().reshape(-1, normed.shape[-1])
            flat_scores = scores.reshape(-1, scores.shape[-1])
            if train_tokens_per_batch is not None and train_tokens_per_batch > 0:
                sample_count = min(int(train_tokens_per_batch), flat_normed.shape[0])
                if sample_count < flat_normed.shape[0]:
                    if device.type == "mps":
                        sample_idx = torch.randperm(
                            flat_normed.shape[0],
                            generator=generator,
                            device="cpu",
                        )[:sample_count].to(device)
                    else:
                        sample_idx = torch.randperm(
                            flat_normed.shape[0],
                            generator=generator,
                            device=device,
                        )[:sample_count]
                    flat_normed = flat_normed.index_select(0, sample_idx)
                    flat_scores = flat_scores.index_select(0, sample_idx)
            logits = selector(layer_idx, flat_normed)
            loss, aux = _selector_loss_for_scores(logits, flat_scores, label_k=label_k)
            (loss / config.model.n_dense_layers).backward()
            total_loss_value += float(loss.detach().cpu())
            aux_sums["selector_bce"] += aux["selector_bce"]
            aux_sums["selector_score_mse"] += aux["selector_score_mse"]
            x = next_x.detach()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.empty_cache()
        if progress and (batch_idx + 1) % progress == 0:
            log = {
                "batch": float(batch_idx + 1),
                "selector_loss": total_loss_value / config.model.n_dense_layers,
                "selector_bce": aux_sums["selector_bce"] / config.model.n_dense_layers,
                "selector_score_mse": aux_sums["selector_score_mse"] / config.model.n_dense_layers,
            }
            logs.append(log)
            print(json.dumps({"event": "selector_train_batch", **log}))
    selector.eval()
    return selector, logs


@torch.no_grad()
def _selector_candidate_full_stack_logits(
    dense: DenseModel,
    selector: LowRankNeuronSelector,
    tokens: torch.Tensor,
    *,
    k: int,
    candidate_m: int,
    reranker: str = "norm",
    ridge: float = 1e-4,
    coeff_prior: float = 1.0,
    coeff_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_candidate_rerank_output(
            block,
            block.norm2(u),
            selector,
            layer_idx=layer_idx,
            candidate_m=candidate_m,
            top_k=k,
            reranker=reranker,
            ridge=ridge,
            coeff_prior=coeff_prior,
            coeff_clamp=coeff_clamp,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


def cmd_deferred_neuron_selector_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ks = _topk_list(args.k, [64, 128])
    candidate_ms = _topk_list(args.candidate_m, [128, 256])
    rerankers = list(args.rerankers or ["norm"])
    ridge_values = [float(value) for value in (args.rerank_ridge or [args.ridge])]
    clamp_values: list[float | None] = []
    for value in args.rerank_clamp or ["none"]:
        clamp_values.append(None if str(value).lower() == "none" else float(value))
    label_k = args.label_k if args.label_k is not None else max(ks)
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    selector = LowRankNeuronSelector(
        n_layers=config.model.n_dense_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        rank=args.selector_rank,
    ).to(device)
    train_logs: list[dict[str, float]] = []
    loaded_selector = False
    if args.selector_checkpoint:
        payload = torch.load(args.selector_checkpoint, map_location=device, weights_only=False)
        selector.load_state_dict(payload["selector"], strict=True)
        loaded_selector = True
    elif args.train_batches > 0:
        selector, train_logs = _train_lowrank_neuron_selector(
            dense,
            streams,
            config,
            rank=args.selector_rank,
            label_k=label_k,
            train_batches=args.train_batches,
            lr=args.selector_lr,
            weight_decay=args.selector_weight_decay,
            train_tokens_per_batch=args.selector_train_tokens,
            device=device,
            progress=args.train_progress,
        )
        if args.save_selector:
            output = Path(args.save_selector)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "selector": selector.state_dict(),
                    "selector_rank": args.selector_rank,
                    "label_k": label_k,
                    "train_batches": args.train_batches,
                    "selector_train_tokens": args.selector_train_tokens,
                    "config": {
                        "n_dense_layers": config.model.n_dense_layers,
                        "d_model": config.model.d_model,
                        "d_ff": config.model.d_ff,
                    },
                },
                output,
            )
    else:
        raise SystemExit("--selector-checkpoint is required when --train-batches is 0")
    selector.eval()
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_selector_oracle_bucket(
            "dense",
            variant_type="dense",
            k=None,
            candidate_m=None,
        )
    }
    for k in ks:
        name = f"oracle_neuron_norm_k{k}"
        buckets[name] = _init_selector_oracle_bucket(
            name,
            variant_type="oracle_norm",
            k=k,
            candidate_m=None,
        )
    for candidate_m in candidate_ms:
        for k in ks:
            if k > candidate_m:
                continue
            for reranker in rerankers:
                if reranker == "omp_ridge1":
                    combos = [(ridge, clamp) for ridge in ridge_values for clamp in clamp_values]
                elif reranker == "omp_ls":
                    combos = [(args.ridge, None)]
                else:
                    combos = [(None, None)]
                for ridge_value, clamp in combos:
                    suffix = reranker
                    if reranker == "omp_ridge1":
                        suffix = f"{suffix}_r{ridge_value:g}"
                        if clamp is not None:
                            suffix = f"{suffix}_clamp{clamp:g}"
                    name = f"selector_m{candidate_m}_{suffix}_k{k}"
                    buckets[name] = _init_selector_oracle_bucket(
                        name,
                        variant_type="selector_candidate_rerank",
                        k=k,
                        candidate_m=candidate_m,
                        reranker=reranker,
                        ridge=ridge_value,
                        coeff_clamp=clamp,
                    )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for k in ks:
            logits, hidden, _ = _full_stack_sparse_ffn_logits_and_states(
                dense,
                tokens,
                mode="neuron",
                requested_k=k,
                ffn_group_ids=torch.empty(0, dtype=torch.long, device=device),
                ffn_groups=config.model.ffn_groups,
                selector="norm",
                ridge=args.ridge,
            )
            name = f"oracle_neuron_norm_k{k}"
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
        for candidate_m in candidate_ms:
            for k in ks:
                if k > candidate_m:
                    continue
                for reranker in rerankers:
                    if reranker == "omp_ridge1":
                        combos = [(ridge, clamp) for ridge in ridge_values for clamp in clamp_values]
                    elif reranker == "omp_ls":
                        combos = [(args.ridge, None)]
                    else:
                        combos = [(None, None)]
                    for ridge_value, clamp in combos:
                        logits, hidden, aux = _selector_candidate_full_stack_logits(
                            dense,
                            selector,
                            tokens,
                            k=k,
                            candidate_m=candidate_m,
                            reranker=reranker,
                            ridge=args.ridge if ridge_value is None else ridge_value,
                            coeff_prior=1.0,
                            coeff_clamp=clamp,
                        )
                        suffix = reranker
                        if reranker == "omp_ridge1":
                            suffix = f"{suffix}_r{ridge_value:g}"
                            if clamp is not None:
                                suffix = f"{suffix}_clamp{clamp:g}"
                        name = f"selector_m{candidate_m}_{suffix}_k{k}"
                        bucket = buckets[name]
                        loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
                        cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
                        kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                            args.temperature * args.temperature
                        )
                        final_cos = F.cosine_similarity(
                            hidden.float().flatten(1),
                            dense_hidden.float().flatten(1),
                            dim=-1,
                        ).sum()
                        bucket["tokens"] += token_count
                        bucket["samples"] += sample_count
                        bucket["loss_sum"] += float(loss.detach().cpu())
                        bucket["kl_sum"] += float(kl.detach().cpu())
                        bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
                        for key in (
                            "candidate_recall_sum",
                            "selected_recall_sum",
                            "score_retention_sum",
                            "selection_count",
                            "coeff_count",
                            "coeff_sum",
                            "coeff_abs_sum",
                            "coeff_sq_sum",
                        ):
                            bucket[key] += float(aux.get(key, 0.0))
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "selector_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_selector_oracle_bucket(bucket) for bucket in buckets.values()]
    report = {
        "mode": "deferred_neuron_selector_oracle",
        "checkpoint": args.dense_checkpoint,
        "selector_rank": args.selector_rank,
        "label_k": label_k,
        "train_batches": args.train_batches,
        "selector_train_tokens": args.selector_train_tokens,
        "selector_checkpoint": args.selector_checkpoint,
        "save_selector": args.save_selector,
        "loaded_selector": loaded_selector,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "k": ks,
        "candidate_m": candidate_ms,
        "rerankers": rerankers,
        "rerank_ridge": ridge_values,
        "rerank_clamp": clamp_values,
        "temperature": args.temperature,
        "ridge": args.ridge,
        "attention": "dense",
        "train_logs": train_logs,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


@torch.no_grad()
def _factor_union_full_stack_logits(
    dense: DenseModel,
    tokens: torch.Tensor,
    *,
    k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
    }
    for block in dense.blocks:
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_factor_union_output(
            block,
            block.norm2(u),
            top_k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


def _init_factor_union_bucket(
    name: str,
    *,
    variant_type: str,
    k: int | None = None,
    rank: int | None = None,
    up_m: int | None = None,
    gate_m: int | None = None,
    product_m: int | None = None,
    reranker: str | None = None,
) -> dict[str, Any]:
    return {
        "variant": name,
        "variant_type": variant_type,
        "k": k,
        "rank": rank,
        "up_m": up_m,
        "gate_m": gate_m,
        "product_m": product_m,
        "reranker": reranker,
        "tokens": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "kl_sum": 0.0,
        "final_hidden_cosine_sum": 0.0,
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
    }


def _finish_factor_union_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "variant_type": bucket["variant_type"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    for key in ("k", "rank", "up_m", "gate_m", "product_m", "reranker"):
        if bucket.get(key) is not None:
            row[key] = bucket[key]
    count = max(float(bucket.get("selection_count", 0.0)), 1.0)
    if bucket.get("selection_count", 0.0):
        row["avg_candidate_size"] = bucket["candidate_size_sum"] / count
        row["candidate_recall"] = bucket["candidate_recall_sum"] / count
        row["selected_recall"] = bucket["selected_recall_sum"] / count
        row["score_retention"] = bucket["score_retention_sum"] / count
    return row


def cmd_deferred_neuron_factor_union_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ks = _topk_list(args.k, [64])
    rerankers = list(args.rerankers or ["norm", "omp_unit"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_factor_union_bucket("dense", variant_type="dense")
    }
    for k in ks:
        name = f"oracle_neuron_norm_k{k}"
        buckets[name] = _init_factor_union_bucket(name, variant_type="oracle_norm", k=k)
    variants: list[tuple[str, int, int, int, int, str]] = []
    for k in ks:
        for reranker in rerankers:
            variants.append((f"factor_upgate_m{args.factor_m}_{reranker}_k{k}", k, args.factor_m, args.factor_m, 0, reranker))
            variants.append(
                (
                    f"factor_upgateprod_m{args.product_factor_m}_{reranker}_k{k}",
                    k,
                    args.product_factor_m,
                    args.product_factor_m,
                    args.product_factor_m,
                    reranker,
                )
            )
    for name, k, up_m, gate_m, product_m, reranker in variants:
        buckets[name] = _init_factor_union_bucket(
            name,
            variant_type="factor_union",
            k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for k in ks:
            logits, hidden, _ = _full_stack_sparse_ffn_logits_and_states(
                dense,
                tokens,
                mode="neuron",
                requested_k=k,
                ffn_group_ids=torch.empty(0, dtype=torch.long, device=device),
                ffn_groups=config.model.ffn_groups,
                selector="norm",
                ridge=args.ridge,
            )
            name = f"oracle_neuron_norm_k{k}"
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
        for name, k, up_m, gate_m, product_m, reranker in variants:
            logits, hidden, aux = _factor_union_full_stack_logits(
                dense,
                tokens,
                k=k,
                up_m=up_m,
                gate_m=gate_m,
                product_m=product_m,
                reranker=reranker,
            )
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "selected_recall_sum",
                "score_retention_sum",
                "selection_count",
            ):
                bucket[key] += float(aux.get(key, 0.0))
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "factor_union_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_factor_union_bucket(bucket) for bucket in buckets.values()]
    report = {
        "mode": "deferred_neuron_factor_union_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "k": ks,
        "factor_m": args.factor_m,
        "product_factor_m": args.product_factor_m,
        "rerankers": rerankers,
        "temperature": args.temperature,
        "attention": "dense",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _train_factor_union_selector(
    dense: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    rank: int,
    up_label_k: int,
    gate_label_k: int,
    product_label_k: int,
    train_batches: int,
    lr: float,
    weight_decay: float,
    train_tokens_per_batch: int | None,
    device: torch.device,
    progress: int,
) -> tuple[FactorUnionNeuronSelector, list[dict[str, float]]]:
    selector = FactorUnionNeuronSelector(
        n_layers=config.model.n_dense_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        rank=rank,
    ).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=weight_decay)
    iterator = streams.train_batches(config.training)
    generator = torch.Generator(device=device if device.type != "mps" else "cpu")
    generator.manual_seed(config.training.seed + 29)
    logs: list[dict[str, float]] = []
    dense.eval()
    selector.train()
    active_heads = [
        ("up", up_label_k),
        ("gate", gate_label_k),
        ("product", product_label_k),
    ]
    active_heads = [(name, k) for name, k in active_heads if k > 0]
    loss_divisor = max(1, config.model.n_dense_layers * len(active_heads))
    for batch_idx in range(max(0, int(train_batches))):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            x = dense.embed(tokens)
        total_loss_value = 0.0
        aux_sums: dict[str, float] = {
            "up_bce": 0.0,
            "gate_bce": 0.0,
            "product_bce": 0.0,
            "up_score_mse": 0.0,
            "gate_score_mse": 0.0,
            "product_score_mse": 0.0,
        }
        for layer_idx, block in enumerate(dense.blocks):
            with torch.no_grad():
                u = x + block.attn(block.norm1(x))
                normed = block.norm2(u)
                up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
                gate_act = F.silu(gate)
                z = up * gate_act
                wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(normed.device)
                scores = {
                    "up": up.detach().abs() * wd_norm,
                    "gate": gate_act.detach().abs() * wd_norm,
                    "product": z.detach().abs() * wd_norm,
                }
                next_x = u + block.mlp.wd(z)
            flat_normed = normed.detach().reshape(-1, normed.shape[-1])
            flat_scores = {name: value.reshape(-1, value.shape[-1]) for name, value in scores.items()}
            if train_tokens_per_batch is not None and train_tokens_per_batch > 0:
                sample_count = min(int(train_tokens_per_batch), flat_normed.shape[0])
                if sample_count < flat_normed.shape[0]:
                    if device.type == "mps":
                        sample_idx = torch.randperm(
                            flat_normed.shape[0],
                            generator=generator,
                            device="cpu",
                        )[:sample_count].to(device)
                    else:
                        sample_idx = torch.randperm(
                            flat_normed.shape[0],
                            generator=generator,
                            device=device,
                        )[:sample_count]
                    flat_normed = flat_normed.index_select(0, sample_idx)
                    flat_scores = {
                        name: value.index_select(0, sample_idx)
                        for name, value in flat_scores.items()
                    }
            for name, label_k in active_heads:
                logits = selector(name, layer_idx, flat_normed)
                loss, aux = _selector_loss_for_scores(
                    logits,
                    flat_scores[name],
                    label_k=label_k,
                )
                (loss / loss_divisor).backward()
                total_loss_value += float(loss.detach().cpu())
                aux_sums[f"{name}_bce"] += aux["selector_bce"]
                aux_sums[f"{name}_score_mse"] += aux["selector_score_mse"]
            x = next_x.detach()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.empty_cache()
        if progress and (batch_idx + 1) % progress == 0:
            denom = max(1, config.model.n_dense_layers)
            log = {
                "batch": float(batch_idx + 1),
                "selector_loss": total_loss_value / loss_divisor,
                "up_bce": aux_sums["up_bce"] / denom,
                "gate_bce": aux_sums["gate_bce"] / denom,
                "product_bce": aux_sums["product_bce"] / denom,
                "up_score_mse": aux_sums["up_score_mse"] / denom,
                "gate_score_mse": aux_sums["gate_score_mse"] / denom,
                "product_score_mse": aux_sums["product_score_mse"] / denom,
            }
            logs.append(log)
            print(json.dumps({"event": "factor_selector_train_batch", **log}))
    selector.eval()
    return selector, logs


@torch.no_grad()
def _learned_factor_union_full_stack_logits(
    dense: DenseModel,
    selector: FactorUnionNeuronSelector,
    tokens: torch.Tensor,
    *,
    k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_learned_factor_union_output(
            block,
            block.norm2(u),
            selector,
            layer_idx=layer_idx,
            top_k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


@torch.no_grad()
def _svd_factor_union_full_stack_logits(
    dense: DenseModel,
    svd_factors: list[dict[str, torch.Tensor]],
    tokens: torch.Tensor,
    *,
    rank: int,
    k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    reranker: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "selected_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_svd_factor_union_output(
            block,
            block.norm2(u),
            svd_factors[layer_idx],
            rank=rank,
            top_k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


@torch.no_grad()
def _svd_cluster_pool_full_stack_logits(
    dense: DenseModel,
    svd_factors: list[dict[str, torch.Tensor]],
    tokens: torch.Tensor,
    *,
    rank: int,
    cluster_count: int,
    candidate_m: int,
    reference_k: int,
    score_mode: str,
    aggregation: str,
    cluster_iters: int,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
        "cluster_count_sum": 0.0,
        "nonempty_cluster_count_sum": 0.0,
        "empty_cluster_count_sum": 0.0,
        "max_cluster_size_sum": 0.0,
        "min_cluster_size_sum": 0.0,
        "mean_cluster_size_sum": 0.0,
        "cluster_imbalance_sum": 0.0,
        "cluster_metric_count": 0.0,
        "cluster_pool_ffn_flop_ratio_sum": 0.0,
        "cluster_pool_exec_seconds": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_svd_cluster_pool_output(
            block,
            block.norm2(u),
            svd_factors[layer_idx],
            rank=rank,
            cluster_count=cluster_count,
            candidate_m=candidate_m,
            reference_k=reference_k,
            score_mode=score_mode,
            aggregation=aggregation,
            cluster_iters=cluster_iters,
            profile=profile,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


@torch.no_grad()
def _build_static_svd_cluster_pool_codebook(
    dense: DenseModel,
    svd_factors: list[dict[str, torch.Tensor]],
    batches,
    *,
    calibration_batches: int,
    device: torch.device,
    rank: int,
    cluster_count: int,
    candidate_m: int,
    score_mode: str,
    aggregation: str,
    cluster_iters: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, float]]:
    layer_features: list[list[torch.Tensor]] = [[] for _ in dense.blocks]
    layer_scores: list[list[torch.Tensor]] = [[] for _ in dense.blocks]
    tokens_seen = 0
    for _ in range(calibration_batches):
        tokens, _ = next(batches)
        tokens = tokens.to(device)
        x = dense.embed(tokens)
        tokens_seen += int(tokens.numel())
        for layer_idx, block in enumerate(dense.blocks):
            u = x + block.attn(block.norm1(x))
            normed = block.norm2(u)
            features, scores = _svd_cluster_features_and_scores(
                block,
                normed,
                svd_factors[layer_idx],
                rank=rank,
                score_mode=score_mode,
            )
            # Calibration is an offline/codebook build step. Keep its tensors
            # on CPU in fp16 so large calibration windows do not sit in GPU/MPS
            # memory while later layers are collected.
            layer_features[layer_idx].append(features.detach().to("cpu", dtype=torch.float16))
            layer_scores[layer_idx].append(scores.detach().to("cpu", dtype=torch.float16))
            x = u + block.mlp(normed)
        if device.type == "mps":
            torch.mps.empty_cache()

    codebook: list[dict[str, torch.Tensor]] = []
    nonempty_sum = 0.0
    imbalance_sum = 0.0
    for layer_idx in range(len(dense.blocks)):
        features = torch.cat(layer_features[layer_idx], dim=0).float()
        scores = torch.cat(layer_scores[layer_idx], dim=0).float()
        assignments, centers = _fit_cluster_centers_from_features(
            features,
            cluster_count=cluster_count,
            iters=cluster_iters,
        )
        actual_cluster_count = min(max(1, int(cluster_count)), features.shape[0])
        aggregate_scores = _aggregate_cluster_scores(
            scores,
            assignments,
            cluster_count=actual_cluster_count,
            aggregation=aggregation,
        )
        candidate_ids = torch.topk(
            aggregate_scores,
            k=min(candidate_m, aggregate_scores.shape[-1]),
            dim=-1,
        ).indices
        cluster_sizes = torch.bincount(assignments, minlength=actual_cluster_count).float()
        nonempty = cluster_sizes[cluster_sizes > 0]
        nonempty_sum += float(nonempty.numel())
        imbalance_sum += float((nonempty.max() / nonempty.mean().clamp_min(1.0)).item()) if nonempty.numel() else 0.0
        codebook.append(
            {
                "centers": centers.to(device=device),
                "candidate_ids": candidate_ids.to(device=device),
            }
        )
    metric_count = max(1, len(dense.blocks))
    return codebook, {
        "calibration_tokens_seen": float(tokens_seen),
        "calibration_avg_nonempty_clusters": nonempty_sum / metric_count,
        "calibration_avg_cluster_imbalance": imbalance_sum / metric_count,
    }


@torch.no_grad()
def _ffn_neuron_static_svd_cluster_pool_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    factors: dict[str, torch.Tensor],
    codebook: dict[str, torch.Tensor],
    *,
    rank: int,
    reference_k: int,
    profile: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    original_shape = normed.shape[:-1]
    flat = normed.reshape(-1, normed.shape[-1])
    hidden = block.mlp.wd.in_features
    reference_k = min(max(1, int(reference_k)), hidden)
    centers = codebook["centers"].to(device=flat.device)
    candidate_ids = codebook["candidate_ids"].to(device=flat.device)
    cluster_count = candidate_ids.shape[0]
    candidate_m = candidate_ids.shape[1]

    start = time.perf_counter()
    features, _ = _svd_cluster_features_and_scores(
        block,
        normed,
        factors,
        rank=rank,
        score_mode="sum",
    )
    assignments = (F.normalize(features.float(), dim=-1) @ centers.float().t()).argmax(dim=-1)

    out = flat.new_zeros(flat.shape[0], flat.shape[-1])
    up_weight, gate_weight = block.mlp.wug.weight.detach().chunk(2, dim=0)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    cluster_sizes = torch.bincount(assignments, minlength=cluster_count)
    for cluster_idx in range(cluster_count):
        token_idx = torch.nonzero(assignments == cluster_idx, as_tuple=False).flatten()
        if token_idx.numel() == 0:
            continue
        ids = candidate_ids[cluster_idx]
        x_cluster = flat.index_select(0, token_idx)
        up = x_cluster @ up_weight.index_select(0, ids).t().contiguous()
        gate = x_cluster @ gate_weight.index_select(0, ids).t().contiguous()
        z = up * F.silu(gate)
        y_cluster = z @ wd_rows.index_select(0, ids).contiguous()
        if out.device.type == "mps":
            # MPS does not implement aten::index_copy.out. This path is only
            # used by local oracle diagnostics; CUDA training kernels use the
            # packed/preindexed execution path instead.
            out[token_idx] = y_cluster
        else:
            out.index_copy_(0, token_idx, y_cluster)

    up, gate = block.mlp.wug(normed).chunk(2, dim=-1)
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(device=flat.device, dtype=flat.dtype)
    exact_scores = (up * F.silu(gate)).reshape(flat.shape[0], hidden).detach().abs() * wd_norm
    true_ids = torch.topk(exact_scores, k=reference_k, dim=-1).indices
    token_candidates = candidate_ids.index_select(0, assignments)
    candidate_hits = true_ids.unsqueeze(-1).eq(token_candidates.unsqueeze(1)).any(dim=-1)
    candidate_recall = candidate_hits.float().sum(dim=-1) / max(reference_k, 1)
    candidate_scores = torch.gather(exact_scores, dim=-1, index=token_candidates).sum(dim=-1)
    true_scores = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    nonempty = cluster_sizes[cluster_sizes > 0].float()
    elapsed = time.perf_counter() - start
    return out.view(*original_shape, -1), {
        "candidate_size_sum": float(candidate_m * flat.shape[0]),
        "candidate_recall_sum": float(candidate_recall.sum().detach().cpu()),
        "score_retention_sum": float((candidate_scores / true_scores).sum().detach().cpu()),
        "selection_count": int(flat.shape[0]),
        "cluster_count_sum": float(cluster_count),
        "nonempty_cluster_count_sum": float(nonempty.numel()),
        "empty_cluster_count_sum": float(cluster_count - nonempty.numel()),
        "max_cluster_size_sum": float(nonempty.max().detach().cpu()) if nonempty.numel() else 0.0,
        "min_cluster_size_sum": float(nonempty.min().detach().cpu()) if nonempty.numel() else 0.0,
        "mean_cluster_size_sum": float(nonempty.mean().detach().cpu()) if nonempty.numel() else 0.0,
        "cluster_imbalance_sum": float((nonempty.max() / nonempty.mean().clamp_min(1.0)).detach().cpu())
        if nonempty.numel()
        else 0.0,
        "cluster_metric_count": 1.0,
        "cluster_pool_ffn_flop_ratio_sum": float(candidate_m / hidden),
        "cluster_pool_exec_seconds": elapsed if profile else 0.0,
    }


@torch.no_grad()
def _svd_static_cluster_pool_full_stack_logits(
    dense: DenseModel,
    svd_factors: list[dict[str, torch.Tensor]],
    codebook: list[dict[str, torch.Tensor]],
    tokens: torch.Tensor,
    *,
    rank: int,
    reference_k: int,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
        "cluster_count_sum": 0.0,
        "nonempty_cluster_count_sum": 0.0,
        "empty_cluster_count_sum": 0.0,
        "max_cluster_size_sum": 0.0,
        "min_cluster_size_sum": 0.0,
        "mean_cluster_size_sum": 0.0,
        "cluster_imbalance_sum": 0.0,
        "cluster_metric_count": 0.0,
        "cluster_pool_ffn_flop_ratio_sum": 0.0,
        "cluster_pool_exec_seconds": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_static_svd_cluster_pool_output(
            block,
            block.norm2(u),
            svd_factors[layer_idx],
            codebook[layer_idx],
            rank=rank,
            reference_k=reference_k,
            profile=profile,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


@torch.no_grad()
def _active_union_ids_from_codebook(
    candidate_ids: torch.Tensor,
    *,
    hidden: int,
    cap: int | None = None,
) -> torch.Tensor:
    candidate_ids = candidate_ids.to(dtype=torch.long)
    full_union = torch.unique(candidate_ids.reshape(-1), sorted=True)
    if cap is None or int(cap) <= 0 or full_union.numel() <= int(cap):
        return full_union
    cap = min(int(cap), hidden)
    cluster_count, candidate_m = candidate_ids.shape
    # Candidate ids are sorted within each cluster by aggregate score. Use a
    # simple frequency-plus-rank score so capped union remains tied to the
    # original cluster codebook and does not peek at eval labels.
    rank_weight = torch.linspace(
        1.0,
        1.0 / max(candidate_m, 1),
        steps=candidate_m,
        device=candidate_ids.device,
        dtype=torch.float32,
    )
    weights = rank_weight.unsqueeze(0).expand(cluster_count, candidate_m).reshape(-1)
    scores = torch.zeros(hidden, device=candidate_ids.device, dtype=torch.float32)
    scores.index_add_(0, candidate_ids.reshape(-1), weights)
    active = torch.topk(scores, k=cap, dim=0).indices
    return torch.sort(active).values


@torch.no_grad()
def _ffn_neuron_static_union_output(
    block: torch.nn.Module,
    normed: torch.Tensor,
    codebook: dict[str, torch.Tensor],
    *,
    reference_k: int,
    cap: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate one FFN using the union of all static cluster candidates."""

    original_shape = normed.shape[:-1]
    flat = normed.reshape(-1, normed.shape[-1])
    hidden = block.mlp.wd.in_features
    reference_k = min(max(1, int(reference_k)), hidden)
    active_ids = _active_union_ids_from_codebook(
        codebook["candidate_ids"].to(device=flat.device),
        hidden=hidden,
        cap=cap,
    )
    up_weight, gate_weight = block.mlp.wug.weight.detach().chunk(2, dim=0)
    wd_rows = block.mlp.wd.weight.t().contiguous()
    up = flat @ up_weight.index_select(0, active_ids).t().contiguous()
    gate = flat @ gate_weight.index_select(0, active_ids).t().contiguous()
    z = up * F.silu(gate)
    out = z @ wd_rows.index_select(0, active_ids).contiguous()

    full_up, full_gate = block.mlp.wug(normed).chunk(2, dim=-1)
    wd_norm = block.mlp.wd.weight.detach().norm(dim=0).to(device=flat.device, dtype=flat.dtype)
    exact_scores = (full_up * F.silu(full_gate)).reshape(flat.shape[0], hidden).detach().abs() * wd_norm
    true_ids = torch.topk(exact_scores, k=reference_k, dim=-1).indices
    union_hits = true_ids.unsqueeze(-1).eq(active_ids.view(1, 1, -1)).any(dim=-1)
    union_recall = union_hits.float().sum(dim=-1) / max(reference_k, 1)
    true_score = torch.gather(exact_scores, dim=-1, index=true_ids).sum(dim=-1).clamp_min(1e-12)
    union_score = exact_scores.index_select(-1, active_ids).sum(dim=-1)
    return out.view(*original_shape, -1), {
        "active_size_sum": float(active_ids.numel() * flat.shape[0]),
        "active_fraction_sum": float(active_ids.numel() / max(hidden, 1)),
        "union_recall_sum": float(union_recall.sum().detach().cpu()),
        "union_score_ratio_sum": float((union_score / true_score).sum().detach().cpu()),
        "selection_count": float(flat.shape[0]),
        "union_metric_count": 1.0,
        "active_cap": float(cap or 0),
    }


@torch.no_grad()
def _svd_static_union_full_stack_logits(
    dense: DenseModel,
    codebook: list[dict[str, torch.Tensor]],
    tokens: torch.Tensor,
    *,
    reference_k: int,
    cap: int | None = None,
    layer_caps: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    aux = {
        "active_size_sum": 0.0,
        "active_fraction_sum": 0.0,
        "union_recall_sum": 0.0,
        "union_score_ratio_sum": 0.0,
        "selection_count": 0.0,
        "union_metric_count": 0.0,
    }
    for layer_idx, block in enumerate(dense.blocks):
        layer_cap = cap if layer_caps is None else int(layer_caps[layer_idx])
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = _ffn_neuron_static_union_output(
            block,
            block.norm2(u),
            codebook[layer_idx],
            reference_k=reference_k,
            cap=layer_cap,
        )
        for key in aux:
            aux[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t(), hidden, aux


def _iter_static_cluster_pool_mlps(model: DenseModel):
    for block in model.blocks:
        if isinstance(block.mlp, StaticClusterPoolSwiGLU):
            yield block.mlp


def _iter_active_union_mlps(model: DenseModel):
    for block in model.blocks:
        if isinstance(block.mlp, ActiveUnionSwiGLU):
            yield block.mlp


def _set_triton_swiglu_backward(model: DenseModel, enabled: bool) -> None:
    use_triton = bool(enabled and triton_swiglu_available())
    for module in model.modules():
        if hasattr(module, "use_triton_swiglu_backward"):
            module.use_triton_swiglu_backward = use_triton


def _pack_dense_ffns_for_benchmark(model: DenseModel) -> None:
    for block in model.blocks:
        if isinstance(block.mlp, PackedDenseSwiGLU):
            continue
        if isinstance(block.mlp, (ActiveUnionSwiGLU, PackedActiveUnionSwiGLU, StaticClusterPoolSwiGLU)):
            raise RuntimeError("dense benchmark packing expects dense SwiGLU blocks")
        device = block.mlp.wug.weight.device
        dtype = block.mlp.wug.weight.dtype
        block.mlp = PackedDenseSwiGLU(block.mlp).to(device=device, dtype=dtype)


def _set_active_union_sparse_enabled(model: DenseModel, enabled: bool) -> None:
    for mlp in _iter_active_union_mlps(model):
        mlp.sparse_enabled = bool(enabled)


def _install_active_union_ffns(
    model: DenseModel,
    codebook: list[dict[str, torch.Tensor]],
    *,
    cap: int | None = None,
    layer_caps: Sequence[int] | None = None,
    packed: bool = False,
) -> None:
    for layer_idx, block in enumerate(model.blocks):
        layer_cap = cap if layer_caps is None else int(layer_caps[layer_idx])
        hidden = int(codebook[layer_idx]["candidate_ids"].max().item()) + 1
        if hasattr(block.mlp, "wd"):
            hidden = int(block.mlp.wd.in_features)
        active_ids = _active_union_ids_from_codebook(
            codebook[layer_idx]["candidate_ids"],
            hidden=hidden,
            cap=layer_cap,
        )
        if packed:
            if isinstance(block.mlp, PackedActiveUnionSwiGLU):
                if torch.equal(block.mlp.active_ids.detach().cpu(), active_ids.detach().cpu()):
                    continue
                raise RuntimeError("cannot change PackedActiveUnionSwiGLU active ids without a dense master")
            device = block.mlp.wug.weight.device
            dtype = block.mlp.wug.weight.dtype
            block.mlp = PackedActiveUnionSwiGLU(block.mlp, active_ids).to(device=device, dtype=dtype)
            continue
        if isinstance(block.mlp, ActiveUnionSwiGLU):
            block.mlp.update_active_ids(active_ids)
            block.mlp.sparse_enabled = True
            continue
        if isinstance(block.mlp, PackedActiveUnionSwiGLU):
            raise RuntimeError("cannot install indexed ActiveUnionSwiGLU over a packed active-union module")
        device = block.mlp.wug.weight.device
        dtype = block.mlp.wug.weight.dtype
        block.mlp = ActiveUnionSwiGLU(
            block.mlp,
            active_ids,
            sparse_enabled=True,
        ).to(device=device, dtype=dtype)


def _set_static_cluster_pool_sparse_enabled(model: DenseModel, enabled: bool) -> None:
    for mlp in _iter_static_cluster_pool_mlps(model):
        mlp.sparse_enabled = bool(enabled)


def _install_static_cluster_pool_ffns(
    model: DenseModel,
    svd_factors: list[dict[str, torch.Tensor]],
    codebook: list[dict[str, torch.Tensor]],
    *,
    rank: int,
) -> None:
    for layer_idx, block in enumerate(model.blocks):
        if isinstance(block.mlp, StaticClusterPoolSwiGLU):
            block.mlp.update_codebook(svd_factors[layer_idx], codebook[layer_idx], rank=rank)
            block.mlp.sparse_enabled = True
            continue
        device = block.mlp.wug.weight.device
        dtype = block.mlp.wug.weight.dtype
        block.mlp = StaticClusterPoolSwiGLU(
            block.mlp,
            svd_factors[layer_idx],
            codebook[layer_idx],
            rank=rank,
            sparse_enabled=True,
        ).to(device=device, dtype=dtype)


def _refresh_static_cluster_pool_ffns(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    calibration_batches: int,
    device: torch.device,
    rank: int,
    cluster_count: int,
    candidate_m: int,
    score_mode: str,
    aggregation: str,
    cluster_iters: int,
) -> dict[str, float]:
    was_training = model.training
    _set_static_cluster_pool_sparse_enabled(model, False)
    model.eval()
    svd_factors = _build_svd_factor_cache(model, max_rank=rank, device=device)
    codebook, aux = _build_static_svd_cluster_pool_codebook(
        model,
        svd_factors,
        streams.train_batches(config.training),
        calibration_batches=calibration_batches,
        device=device,
        rank=rank,
        cluster_count=cluster_count,
        candidate_m=candidate_m,
        score_mode=score_mode,
        aggregation=aggregation,
        cluster_iters=cluster_iters,
    )
    _install_static_cluster_pool_ffns(model, svd_factors, codebook, rank=rank)
    _set_static_cluster_pool_sparse_enabled(model, True)
    model.train(was_training)
    return aux


def _refresh_active_union_ffns(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    calibration_batches: int,
    device: torch.device,
    rank: int,
    cluster_count: int,
    candidate_m: int,
    score_mode: str,
    aggregation: str,
    cluster_iters: int,
    cap: int | None = None,
    layer_caps: Sequence[int] | None = None,
    packed: bool = False,
) -> dict[str, float]:
    was_training = model.training
    _set_active_union_sparse_enabled(model, False)
    model.eval()
    svd_factors = _build_svd_factor_cache(model, max_rank=rank, device=device)
    codebook, aux = _build_static_svd_cluster_pool_codebook(
        model,
        svd_factors,
        streams.train_batches(config.training),
        calibration_batches=calibration_batches,
        device=device,
        rank=rank,
        cluster_count=cluster_count,
        candidate_m=candidate_m,
        score_mode=score_mode,
        aggregation=aggregation,
        cluster_iters=cluster_iters,
    )
    _install_active_union_ffns(model, codebook, cap=cap, layer_caps=layer_caps, packed=packed)
    _set_active_union_sparse_enabled(model, True)
    model.train(was_training)
    return aux


@torch.no_grad()
def _eval_lm_nll(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    batches: int,
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    iterator = streams.eval_batches(config.training)
    for _ in range(batches):
        tokens, targets = next(iterator)
        tokens = tokens.to(device)
        targets = targets.to(device)
        out = model(tokens, targets, return_loss_per_sample=True)
        assert out.loss_per_sample is not None
        total_loss += float(out.loss_per_sample.sum().detach().cpu())
        total_tokens += int(targets.numel())
    model.train(was_training)
    return total_loss / max(total_tokens, 1)


def _train_lm_continuation(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    refresh_interval: int = 0,
    refresh_callback=None,
    grad_clip_norm: float | None = None,
    optimizer_state: dict[str, Any] | None = None,
    eval_steps: set[int] | None = None,
    eval_callback=None,
    progress: int = 0,
    label: str = "model",
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_state_loaded = False
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state_loaded = True
        for group in optimizer.param_groups:
            group["lr"] = lr
            group["weight_decay"] = weight_decay
            if "base_lr" in group:
                group["base_lr"] = lr
    batches = streams.train_batches(config.training)
    losses: list[float] = []
    refresh_events: list[dict[str, Any]] = []
    eval_curve: list[dict[str, float]] = []
    eval_steps = set(eval_steps or [])

    def maybe_eval(step: int) -> None:
        if step not in eval_steps or eval_callback is None:
            return
        eval_curve.append({"step": float(step), "nll_per_token": float(eval_callback())})

    model.train()
    start = time.perf_counter()
    maybe_eval(0)
    for step in range(steps):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(tokens, targets, return_loss_per_sample=True)
        assert out.loss_per_sample is not None
        loss = out.loss_per_sample.mean() / float(targets.shape[1])
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if refresh_interval > 0 and (step + 1) % refresh_interval == 0:
            if refresh_callback is not None:
                aux = refresh_callback()
                refresh_events.append({"step": step + 1, **aux})
        maybe_eval(step + 1)
        if progress and (step + 1) % progress == 0:
            print(
                json.dumps(
                    {
                        "event": "static_cluster_pool_continuation_step",
                        "variant": label,
                        "step": step + 1,
                        "nll_per_token": losses[-1],
                    }
                )
            )
    _sync_device(device)
    elapsed = time.perf_counter() - start
    return {
        "steps": steps,
        "train_seconds": elapsed,
        "tokens_trained": steps * config.training.batch_size * config.training.seq_len,
        "tokens_per_second": (steps * config.training.batch_size * config.training.seq_len)
        / max(elapsed, 1e-9),
        "first_train_nll": losses[0] if losses else None,
        "last_train_nll": losses[-1] if losses else None,
        "mean_train_nll": sum(losses) / max(len(losses), 1),
        "optimizer_state_loaded": optimizer_state_loaded,
        "eval_curve": eval_curve,
        "refresh_events": refresh_events,
    }


def _static_cluster_pool_coverage_metrics(model: DenseModel) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for layer_idx, mlp in enumerate(_iter_static_cluster_pool_mlps(model)):
        ids = mlp.candidate_ids.detach().cpu()
        cluster_count, candidate_m = ids.shape
        masks = F.one_hot(ids, num_classes=mlp.wd.in_features).sum(dim=1).bool()
        coverage = masks.any(dim=0)
        overlaps = []
        for i in range(cluster_count):
            for j in range(i + 1, cluster_count):
                overlaps.append(float((masks[i] & masks[j]).sum().item()))
        if overlaps:
            overlap_min = min(overlaps)
            overlap_mean = sum(overlaps) / len(overlaps)
            overlap_max = max(overlaps)
        else:
            overlap_min = overlap_mean = overlap_max = 0.0
        rows.append(
            {
                "layer": float(layer_idx),
                "unique_selected_neurons": float(coverage.sum().item()),
                "coverage_fraction": float(coverage.float().mean().item()),
                "cluster_overlap_min": overlap_min,
                "cluster_overlap_mean": overlap_mean,
                "cluster_overlap_max": overlap_max,
                "cluster_overlap_mean_fraction": overlap_mean / max(float(candidate_m), 1.0),
            }
        )
    if not rows:
        return {"rows": [], "mean": {}}
    mean = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
        if key != "layer"
    }
    return {"rows": rows, "mean": mean}


def _active_union_coverage_metrics(model: DenseModel) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for layer_idx, block in enumerate(model.blocks):
        mlp = block.mlp
        if not isinstance(mlp, (ActiveUnionSwiGLU, PackedActiveUnionSwiGLU)):
            continue
        ids = mlp.active_ids.detach().cpu()
        if isinstance(mlp, ActiveUnionSwiGLU):
            hidden = int(mlp.wd.in_features)
        else:
            hidden = int(getattr(mlp, "d_ff_total", int(ids.numel())))
        rows.append(
            {
                "layer": float(layer_idx),
                "unique_selected_neurons": float(ids.numel()),
                "coverage_fraction": float(ids.numel() / max(hidden, 1)),
            }
        )
    if not rows:
        return {"rows": [], "mean": {}}
    mean = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
        if key != "layer"
    }
    return {"rows": rows, "mean": mean}


def _static_cluster_pool_grad_row_metrics(model: DenseModel) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for layer_idx, mlp in enumerate(_iter_static_cluster_pool_mlps(model)):
        ids = mlp.candidate_ids.detach().to(device=mlp.wug.weight.device)
        selected = ids.reshape(-1).unique()
        row_mask = torch.zeros(mlp.wd.in_features, device=ids.device, dtype=torch.bool)
        row_mask[selected] = True
        if mlp.wug.weight.grad is None or mlp.wd.weight.grad is None:
            continue
        up_grad, gate_grad = mlp.wug.weight.grad.detach().chunk(2, dim=0)
        down_grad = mlp.wd.weight.grad.detach().t().contiguous()

        def row_stats(name: str, grad: torch.Tensor) -> dict[str, float]:
            row_norm = grad.float().norm(dim=-1)
            selected_norm = row_norm[row_mask]
            unselected_norm = row_norm[~row_mask]
            return {
                f"{name}_selected_row_grad_mean": float(selected_norm.mean().detach().cpu())
                if selected_norm.numel()
                else 0.0,
                f"{name}_unselected_row_grad_mean": float(unselected_norm.mean().detach().cpu())
                if unselected_norm.numel()
                else 0.0,
                f"{name}_selected_row_grad_max": float(selected_norm.max().detach().cpu())
                if selected_norm.numel()
                else 0.0,
                f"{name}_unselected_row_grad_max": float(unselected_norm.max().detach().cpu())
                if unselected_norm.numel()
                else 0.0,
                f"{name}_dead_row_count": float((row_norm <= 0).sum().detach().cpu()),
            }

        row = {
            "layer": float(layer_idx),
            "selected_rows": float(selected.numel()),
            "coverage_fraction": float(row_mask.float().mean().detach().cpu()),
            **row_stats("up", up_grad),
            **row_stats("gate", gate_grad),
            **row_stats("down", down_grad),
        }
        rows.append(row)
    if not rows:
        return {"rows": [], "mean": {}}
    mean = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
        if key != "layer"
    }
    return {"rows": rows, "mean": mean}


def _static_cluster_pool_input_grad_alignment(
    model: DenseModel,
    streams,
    config: ExperimentConfig,
    *,
    device: torch.device,
    batches: int,
    seed: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    iterator = streams.eval_batches(config.training)
    for _ in range(batches):
        tokens, _ = next(iterator)
        tokens = tokens.to(device)
        with torch.no_grad():
            x = model.embed(tokens)
        for layer_idx, block in enumerate(model.blocks):
            with torch.no_grad():
                u = x + block.attn(block.norm1(x))
                normed = block.norm2(u).detach()
            if not isinstance(block.mlp, StaticClusterPoolSwiGLU):
                with torch.no_grad():
                    x = u + block.mlp(normed)
                continue
            x_dense = normed.detach().clone().requires_grad_()
            x_sparse = normed.detach().clone().requires_grad_()
            sparse_enabled = block.mlp.sparse_enabled
            block.mlp.sparse_enabled = False
            dense_out = block.mlp(x_dense)
            block.mlp.sparse_enabled = True
            sparse_out = block.mlp(x_sparse)
            block.mlp.sparse_enabled = sparse_enabled
            upstream = torch.randn(
                dense_out.shape,
                generator=gen,
                device="cpu",
                dtype=torch.float32,
            ).to(device=device, dtype=dense_out.dtype)
            dense_grad = torch.autograd.grad((dense_out * upstream).sum(), x_dense)[0]
            sparse_grad = torch.autograd.grad((sparse_out * upstream).sum(), x_sparse)[0]
            rows.append(
                {
                    "layer": float(layer_idx),
                    "ffn_output_cosine": _cosine_flat(dense_out.detach(), sparse_out.detach()),
                    "input_grad_cosine": _cosine_flat(dense_grad, sparse_grad),
                }
            )
            with torch.no_grad():
                x = u + block.mlp(normed)
    model.train(was_training)
    if not rows:
        return {}
    return {
        "ffn_output_cosine": sum(row["ffn_output_cosine"] for row in rows) / len(rows),
        "input_grad_cosine": sum(row["input_grad_cosine"] for row in rows) / len(rows),
    }


def _build_svd_sparse_ffns(
    dense: DenseModel,
    *,
    rank: int,
    k: int,
    up_m: int,
    gate_m: int,
    product_m: int,
    device: torch.device,
    candidate_mode: str = "mask",
    svd_factors: list[dict[str, torch.Tensor]] | None = None,
) -> list[SVDFactorSparseFFN]:
    modules: list[SVDFactorSparseFFN] = []
    for layer_idx, block in enumerate(dense.blocks):
        sparse = SVDFactorSparseFFN.from_dense(
            block.mlp,
            rank=rank,
            top_k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            candidate_mode=candidate_mode,
        ).to(device)
        if svd_factors is not None:
            with torch.no_grad():
                sparse.up_a.copy_(svd_factors[layer_idx]["up_a"][:, : sparse.rank])
                sparse.up_b.copy_(svd_factors[layer_idx]["up_b"][: sparse.rank, :])
                sparse.gate_a.copy_(svd_factors[layer_idx]["gate_a"][:, : sparse.rank])
                sparse.gate_b.copy_(svd_factors[layer_idx]["gate_b"][: sparse.rank, :])
        sparse.eval()
        modules.append(sparse)
    return modules


@torch.no_grad()
def _svd_hot_full_stack_logits(
    dense: DenseModel,
    sparse_ffns: list[SVDFactorSparseFFN],
    tokens: torch.Tensor,
    *,
    profile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    x = dense.embed(tokens)
    candidate_size_sum = 0.0
    selection_count = 0.0
    timing_sums = {
        "selector_score_time": 0.0,
        "candidate_union_dedup_time": 0.0,
        "exact_candidate_activation_time": 0.0,
        "rerank_topk_time": 0.0,
        "down_sum_time": 0.0,
    }
    for block, sparse_ffn in zip(dense.blocks, sparse_ffns, strict=True):
        u = x + block.attn(block.norm1(x))
        ffn, layer_aux = sparse_ffn(block.norm2(u), return_aux=True, profile=profile)
        candidate_size_sum += float(layer_aux["avg_candidate_size"]) * u.numel() / u.shape[-1]
        selection_count += float(u.numel() / u.shape[-1])
        for key in timing_sums:
            timing_sums[key] += float(layer_aux.get(key, 0.0))
        x = u + ffn
    hidden = dense.final_norm(x)
    aux = {
        "candidate_size_sum": candidate_size_sum,
        "selection_count": selection_count,
    }
    aux.update(timing_sums)
    return hidden @ dense.vocab_weight.t(), hidden, aux


def cmd_deferred_neuron_factor_selector_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ks = _topk_list(args.k, [64])
    rerankers = list(args.rerankers or ["norm"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    selector = FactorUnionNeuronSelector(
        n_layers=config.model.n_dense_layers,
        d_model=config.model.d_model,
        d_ff=config.model.d_ff,
        rank=args.selector_rank,
    ).to(device)
    train_logs: list[dict[str, float]] = []
    loaded_selector = False
    if args.selector_checkpoint:
        payload = torch.load(args.selector_checkpoint, map_location=device, weights_only=False)
        selector.load_state_dict(payload["selector"], strict=True)
        loaded_selector = True
    elif args.train_batches > 0:
        selector, train_logs = _train_factor_union_selector(
            dense,
            streams,
            config,
            rank=args.selector_rank,
            up_label_k=args.factor_m,
            gate_label_k=args.factor_m,
            product_label_k=args.product_factor_m,
            train_batches=args.train_batches,
            lr=args.selector_lr,
            weight_decay=args.selector_weight_decay,
            train_tokens_per_batch=args.selector_train_tokens,
            device=device,
            progress=args.train_progress,
        )
        if args.save_selector:
            output = Path(args.save_selector)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "selector": selector.state_dict(),
                    "selector_rank": args.selector_rank,
                    "factor_m": args.factor_m,
                    "product_factor_m": args.product_factor_m,
                    "train_batches": args.train_batches,
                    "selector_train_tokens": args.selector_train_tokens,
                },
                output,
            )
    else:
        raise SystemExit("--selector-checkpoint is required when --train-batches is 0")
    selector.eval()
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_factor_union_bucket("dense", variant_type="dense")
    }
    for k in ks:
        name = f"oracle_neuron_norm_k{k}"
        buckets[name] = _init_factor_union_bucket(name, variant_type="oracle_norm", k=k)
    variants: list[tuple[str, int, int, int, int, str]] = []
    for k in ks:
        for reranker in rerankers:
            variants.append(
                (
                    f"learned_upgate_m{args.factor_m}_{reranker}_k{k}",
                    k,
                    args.factor_m,
                    args.factor_m,
                    0,
                    reranker,
                )
            )
            variants.append(
                (
                    f"learned_upgateprod_m{args.product_factor_m}_{reranker}_k{k}",
                    k,
                    args.product_factor_m,
                    args.product_factor_m,
                    args.product_factor_m,
                    reranker,
                )
            )
    for name, k, up_m, gate_m, product_m, reranker in variants:
        buckets[name] = _init_factor_union_bucket(
            name,
            variant_type="learned_factor_union",
            k=k,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for k in ks:
            logits, hidden, _ = _full_stack_sparse_ffn_logits_and_states(
                dense,
                tokens,
                mode="neuron",
                requested_k=k,
                ffn_group_ids=torch.empty(0, dtype=torch.long, device=device),
                ffn_groups=config.model.ffn_groups,
                selector="norm",
                ridge=args.ridge,
            )
            name = f"oracle_neuron_norm_k{k}"
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
        for name, k, up_m, gate_m, product_m, reranker in variants:
            logits, hidden, aux = _learned_factor_union_full_stack_logits(
                dense,
                selector,
                tokens,
                k=k,
                up_m=up_m,
                gate_m=gate_m,
                product_m=product_m,
                reranker=reranker,
            )
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "selected_recall_sum",
                "score_retention_sum",
                "selection_count",
            ):
                bucket[key] += float(aux.get(key, 0.0))
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "factor_selector_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_factor_union_bucket(bucket) for bucket in buckets.values()]
    report = {
        "mode": "deferred_neuron_factor_selector_oracle",
        "checkpoint": args.dense_checkpoint,
        "selector_rank": args.selector_rank,
        "factor_m": args.factor_m,
        "product_factor_m": args.product_factor_m,
        "train_batches": args.train_batches,
        "selector_train_tokens": args.selector_train_tokens,
        "selector_checkpoint": args.selector_checkpoint,
        "save_selector": args.save_selector,
        "loaded_selector": loaded_selector,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "k": ks,
        "rerankers": rerankers,
        "temperature": args.temperature,
        "attention": "dense",
        "train_logs": train_logs,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_deferred_neuron_svd_factor_union_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ks = _topk_list(args.k, [64])
    ranks = [int(rank) for rank in (args.ranks or [64])]
    factor_ms = [int(value) for value in (args.factor_m or [64])]
    rerankers = list(args.rerankers or ["norm"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    svd_factors = _build_svd_factor_cache(
        dense,
        max_rank=max(ranks),
        device=device,
    )
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_factor_union_bucket("dense", variant_type="dense")
    }
    for k in ks:
        name = f"oracle_neuron_norm_k{k}"
        buckets[name] = _init_factor_union_bucket(name, variant_type="oracle_norm", k=k)
    variants: list[tuple[str, int, int, int, int, int, str]] = []
    for k in ks:
        for rank in ranks:
            for factor_m in factor_ms:
                for reranker in rerankers:
                    variants.append(
                        (
                            f"svd_rank{rank}_upgate_m{factor_m}_{reranker}_k{k}",
                            k,
                            rank,
                            factor_m,
                            factor_m,
                            0,
                            reranker,
                        )
                    )
                    if args.product_factor_m > 0:
                        variants.append(
                            (
                                f"svd_rank{rank}_upgateprod_m{args.product_factor_m}_{reranker}_k{k}",
                                k,
                                rank,
                                args.product_factor_m,
                                args.product_factor_m,
                                args.product_factor_m,
                                reranker,
                            )
                        )
    for name, k, rank, up_m, gate_m, product_m, reranker in variants:
        buckets[name] = _init_factor_union_bucket(
            name,
            variant_type="svd_factor_union",
            k=k,
            rank=rank,
            up_m=up_m,
            gate_m=gate_m,
            product_m=product_m,
            reranker=reranker,
        )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for k in ks:
            logits, hidden, _ = _full_stack_sparse_ffn_logits_and_states(
                dense,
                tokens,
                mode="neuron",
                requested_k=k,
                ffn_group_ids=torch.empty(0, dtype=torch.long, device=device),
                ffn_groups=config.model.ffn_groups,
                selector="norm",
                ridge=args.ridge,
            )
            name = f"oracle_neuron_norm_k{k}"
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
        for name, k, rank, up_m, gate_m, product_m, reranker in variants:
            logits, hidden, aux = _svd_factor_union_full_stack_logits(
                dense,
                svd_factors,
                tokens,
                rank=rank,
                k=k,
                up_m=up_m,
                gate_m=gate_m,
                product_m=product_m,
                reranker=reranker,
            )
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "selected_recall_sum",
                "score_retention_sum",
                "selection_count",
            ):
                bucket[key] += float(aux.get(key, 0.0))
        if device.type == "mps":
            torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "svd_factor_union_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_factor_union_bucket(bucket) for bucket in buckets.values()]
    report = {
        "mode": "deferred_neuron_svd_factor_union_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "k": ks,
        "ranks": ranks,
        "factor_m": factor_ms,
        "product_factor_m": args.product_factor_m,
        "rerankers": rerankers,
        "temperature": args.temperature,
        "attention": "dense",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _init_cluster_pool_bucket(
    name: str,
    *,
    variant_type: str,
    rank: int | None = None,
    cluster_count: int | None = None,
    candidate_m: int | None = None,
    reference_k: int | None = None,
    score_mode: str | None = None,
    aggregation: str | None = None,
    calibration_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "variant": name,
        "variant_type": variant_type,
        "rank": rank,
        "cluster_count": cluster_count,
        "candidate_m": candidate_m,
        "reference_k": reference_k,
        "score_mode": score_mode,
        "aggregation": aggregation,
        "calibration_tokens": calibration_tokens,
        "tokens": 0,
        "samples": 0,
        "loss_sum": 0.0,
        "kl_sum": 0.0,
        "final_hidden_cosine_sum": 0.0,
        "candidate_size_sum": 0.0,
        "candidate_recall_sum": 0.0,
        "score_retention_sum": 0.0,
        "selection_count": 0.0,
        "cluster_count_sum": 0.0,
        "nonempty_cluster_count_sum": 0.0,
        "empty_cluster_count_sum": 0.0,
        "max_cluster_size_sum": 0.0,
        "min_cluster_size_sum": 0.0,
        "mean_cluster_size_sum": 0.0,
        "cluster_imbalance_sum": 0.0,
        "cluster_metric_count": 0.0,
        "cluster_pool_ffn_flop_ratio_sum": 0.0,
        "cluster_pool_exec_seconds": 0.0,
    }


def _finish_cluster_pool_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "variant_type": bucket["variant_type"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    for key in (
        "rank",
        "cluster_count",
        "candidate_m",
        "reference_k",
        "score_mode",
        "aggregation",
        "calibration_tokens",
    ):
        if bucket.get(key) is not None:
            row[key] = bucket[key]
    selection_count = max(float(bucket.get("selection_count", 0.0)), 1.0)
    if bucket.get("selection_count", 0.0):
        row["avg_candidate_size"] = bucket["candidate_size_sum"] / selection_count
        row["candidate_recall"] = bucket["candidate_recall_sum"] / selection_count
        row["score_retention_vs_reference_topk"] = bucket["score_retention_sum"] / selection_count
    metric_count = max(float(bucket.get("cluster_metric_count", 0.0)), 1.0)
    if bucket.get("cluster_metric_count", 0.0):
        row["avg_nonempty_clusters"] = bucket["nonempty_cluster_count_sum"] / metric_count
        row["avg_empty_clusters"] = bucket["empty_cluster_count_sum"] / metric_count
        row["avg_max_cluster_size"] = bucket["max_cluster_size_sum"] / metric_count
        row["avg_min_cluster_size"] = bucket["min_cluster_size_sum"] / metric_count
        row["avg_mean_cluster_size"] = bucket["mean_cluster_size_sum"] / metric_count
        row["avg_cluster_imbalance"] = bucket["cluster_imbalance_sum"] / metric_count
        row["estimated_cluster_ffn_flop_ratio"] = bucket["cluster_pool_ffn_flop_ratio_sum"] / metric_count
    if bucket.get("cluster_pool_exec_seconds", 0.0):
        row["cluster_pool_exec_seconds"] = bucket["cluster_pool_exec_seconds"]
        row["cluster_pool_tokens_per_second"] = tokens / bucket["cluster_pool_exec_seconds"]
    return row


def cmd_deferred_neuron_cluster_pool_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ranks = [int(rank) for rank in (args.ranks or [64])]
    cluster_counts = [int(value) for value in (args.clusters or [8, 16, 32, 64])]
    candidate_ms = [int(value) for value in (args.candidate_m or [128, 192, 256])]
    score_modes = list(args.score_modes or ["sum"])
    aggregations = list(args.aggregations or ["mean"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    svd_factors = _build_svd_factor_cache(
        dense,
        max_rank=max(ranks),
        device=device,
    )
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_cluster_pool_bucket("dense", variant_type="dense")
    }
    variants: list[tuple[str, int, int, int, str, str]] = []
    for rank in ranks:
        for cluster_count in cluster_counts:
            for candidate_m in candidate_ms:
                for score_mode in score_modes:
                    for aggregation in aggregations:
                        name = (
                            f"cluster_svd_rank{rank}_c{cluster_count}_m{candidate_m}_"
                            f"{score_mode}_{aggregation}"
                        )
                        variants.append((name, rank, cluster_count, candidate_m, score_mode, aggregation))
                        buckets[name] = _init_cluster_pool_bucket(
                            name,
                            variant_type="svd_cluster_pool",
                            rank=rank,
                            cluster_count=cluster_count,
                            candidate_m=candidate_m,
                            reference_k=args.reference_k,
                            score_mode=score_mode,
                            aggregation=aggregation,
                        )
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for name, rank, cluster_count, candidate_m, score_mode, aggregation in variants:
            logits, hidden, aux = _svd_cluster_pool_full_stack_logits(
                dense,
                svd_factors,
                tokens,
                rank=rank,
                cluster_count=cluster_count,
                candidate_m=candidate_m,
                reference_k=args.reference_k,
                score_mode=score_mode,
                aggregation=aggregation,
                cluster_iters=args.cluster_iters,
                profile=args.profile,
            )
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "score_retention_sum",
                "selection_count",
                "cluster_count_sum",
                "nonempty_cluster_count_sum",
                "empty_cluster_count_sum",
                "max_cluster_size_sum",
                "min_cluster_size_sum",
                "mean_cluster_size_sum",
                "cluster_imbalance_sum",
                "cluster_metric_count",
                "cluster_pool_ffn_flop_ratio_sum",
                "cluster_pool_exec_seconds",
            ):
                bucket[key] += float(aux.get(key, 0.0))
        if device.type == "mps":
            torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "cluster_pool_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_cluster_pool_bucket(bucket) for bucket in buckets.values()]
    report = {
        "mode": "deferred_neuron_cluster_pool_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "ranks": ranks,
        "clusters": cluster_counts,
        "candidate_m": candidate_ms,
        "reference_k": args.reference_k,
        "score_modes": score_modes,
        "aggregations": aggregations,
        "cluster_iters": args.cluster_iters,
        "temperature": args.temperature,
        "attention": "dense",
        "notes": "cluster pool executes all M candidates per cluster; no per-token final top-k",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_deferred_neuron_static_cluster_pool_oracle(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    ranks = [int(rank) for rank in (args.ranks or [64])]
    cluster_counts = [int(value) for value in (args.clusters or [8, 16])]
    candidate_ms = [int(value) for value in (args.candidate_m or [192])]
    calibration_tokens_values = [int(value) for value in (args.calibration_tokens or [8192])]
    score_modes = list(args.score_modes or ["sum"])
    aggregations = list(args.aggregations or ["mean"])
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    svd_factors = _build_svd_factor_cache(
        dense,
        max_rank=max(ranks),
        device=device,
    )
    buckets: dict[str, dict[str, Any]] = {
        "dense": _init_cluster_pool_bucket("dense", variant_type="dense")
    }
    variants: list[tuple[str, int, int, int, int, str, str]] = []
    codebooks: dict[str, tuple[list[dict[str, torch.Tensor]], dict[str, float]]] = {}
    for calibration_tokens in calibration_tokens_values:
        calibration_batches = max(1, math.ceil(calibration_tokens / tokens_per_batch))
        actual_calibration_tokens = calibration_batches * tokens_per_batch
        for rank in ranks:
            for cluster_count in cluster_counts:
                for candidate_m in candidate_ms:
                    for score_mode in score_modes:
                        for aggregation in aggregations:
                            name = (
                                f"static_cluster_svd_calib{actual_calibration_tokens}_rank{rank}_"
                                f"c{cluster_count}_m{candidate_m}_{score_mode}_{aggregation}"
                            )
                            train_batches = streams.train_batches(config.training)
                            codebook, codebook_aux = _build_static_svd_cluster_pool_codebook(
                                dense,
                                svd_factors,
                                train_batches,
                                calibration_batches=calibration_batches,
                                device=device,
                                rank=rank,
                                cluster_count=cluster_count,
                                candidate_m=candidate_m,
                                score_mode=score_mode,
                                aggregation=aggregation,
                                cluster_iters=args.cluster_iters,
                            )
                            codebooks[name] = (codebook, codebook_aux)
                            variants.append(
                                (
                                    name,
                                    rank,
                                    cluster_count,
                                    candidate_m,
                                    actual_calibration_tokens,
                                    score_mode,
                                    aggregation,
                                )
                            )
                            bucket = _init_cluster_pool_bucket(
                                name,
                                variant_type="static_svd_cluster_pool",
                                rank=rank,
                                cluster_count=cluster_count,
                                candidate_m=candidate_m,
                                reference_k=args.reference_k,
                                score_mode=score_mode,
                                aggregation=aggregation,
                                calibration_tokens=actual_calibration_tokens,
                            )
                            bucket["calibration_avg_nonempty_clusters"] = codebook_aux[
                                "calibration_avg_nonempty_clusters"
                            ]
                            bucket["calibration_avg_cluster_imbalance"] = codebook_aux[
                                "calibration_avg_cluster_imbalance"
                            ]
                            buckets[name] = bucket
                            if args.progress:
                                print(
                                    json.dumps(
                                        {
                                            "event": "static_cluster_pool_codebook_built",
                                            "variant": name,
                                            "calibration_tokens": actual_calibration_tokens,
                                        }
                                    )
                                )
                            if device.type == "mps":
                                torch.mps.empty_cache()

    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)
        for name, rank, _cluster_count, _candidate_m, _calibration_tokens, _score_mode, _aggregation in variants:
            codebook, _codebook_aux = codebooks[name]
            logits, hidden, aux = _svd_static_cluster_pool_full_stack_logits(
                dense,
                svd_factors,
                codebook,
                tokens,
                rank=rank,
                reference_k=args.reference_k,
                profile=args.profile,
            )
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "score_retention_sum",
                "selection_count",
                "cluster_count_sum",
                "nonempty_cluster_count_sum",
                "empty_cluster_count_sum",
                "max_cluster_size_sum",
                "min_cluster_size_sum",
                "mean_cluster_size_sum",
                "cluster_imbalance_sum",
                "cluster_metric_count",
                "cluster_pool_ffn_flop_ratio_sum",
                "cluster_pool_exec_seconds",
            ):
                bucket[key] += float(aux.get(key, 0.0))
        if device.type == "mps":
            torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "static_cluster_pool_eval_batch", "batch": batch_idx + 1}))
    rows = [_finish_cluster_pool_bucket(bucket) for bucket in buckets.values()]
    for row in rows:
        bucket = buckets[row["variant"]]
        if "calibration_avg_nonempty_clusters" in bucket:
            row["calibration_avg_nonempty_clusters"] = bucket["calibration_avg_nonempty_clusters"]
            row["calibration_avg_cluster_imbalance"] = bucket["calibration_avg_cluster_imbalance"]
    report = {
        "mode": "deferred_neuron_static_cluster_pool_oracle",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "calibration_tokens": calibration_tokens_values,
        "ranks": ranks,
        "clusters": cluster_counts,
        "candidate_m": candidate_ms,
        "reference_k": args.reference_k,
        "score_modes": score_modes,
        "aggregations": aggregations,
        "cluster_iters": args.cluster_iters,
        "temperature": args.temperature,
        "attention": "dense",
        "notes": "candidate pools and cluster centers are calibrated once from train stream, then held fixed on eval",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _finish_union_eval_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    tokens = max(int(bucket["tokens"]), 1)
    samples = max(int(bucket["samples"]), 1)
    row: dict[str, Any] = {
        "variant": bucket["variant"],
        "variant_type": bucket["variant_type"],
        "nll_per_token": bucket["loss_sum"] / tokens,
        "kl_to_dense": bucket["kl_sum"] / tokens,
        "final_hidden_cosine": bucket["final_hidden_cosine_sum"] / samples,
    }
    for key in (
        "calibration_tokens",
        "rank",
        "cluster_count",
        "candidate_m",
        "reference_k",
        "score_mode",
        "aggregation",
        "active_cap",
        "active_cap_by_layer",
    ):
        if bucket.get(key) is not None:
            row[key] = bucket[key]
    selection_count = max(float(bucket.get("selection_count", 0.0)), 1.0)
    if bucket.get("selection_count", 0.0):
        row["avg_active_neurons"] = bucket["active_size_sum"] / selection_count
        row["union_recall"] = bucket["union_recall_sum"] / selection_count
        row["union_score_ratio_vs_reference_topk"] = bucket["union_score_ratio_sum"] / selection_count
    metric_count = max(float(bucket.get("union_metric_count", 0.0)), 1.0)
    if bucket.get("union_metric_count", 0.0):
        row["avg_active_fraction"] = bucket["active_fraction_sum"] / metric_count
    return row


def cmd_static_cluster_pool_union_eval(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    actual_calibration_tokens = calibration_batches * tokens_per_batch
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    svd_factors = _build_svd_factor_cache(dense, max_rank=args.rank, device=device)
    codebook, codebook_aux = _build_static_svd_cluster_pool_codebook(
        dense,
        svd_factors,
        streams.train_batches(config.training),
        calibration_batches=calibration_batches,
        device=device,
        rank=args.rank,
        cluster_count=args.clusters,
        candidate_m=args.candidate_m,
        score_mode=args.score_mode,
        aggregation=args.aggregation,
        cluster_iters=args.cluster_iters,
    )
    buckets: dict[str, dict[str, Any]] = {
        "dense": {
            "variant": "dense",
            "variant_type": "dense",
            "tokens": 0,
            "samples": 0,
            "loss_sum": 0.0,
            "kl_sum": 0.0,
            "final_hidden_cosine_sum": 0.0,
        },
        "static_cluster": _init_cluster_pool_bucket(
            "static_cluster",
            variant_type="static_svd_cluster_pool",
            rank=args.rank,
            cluster_count=args.clusters,
            candidate_m=args.candidate_m,
            reference_k=args.reference_k,
            score_mode=args.score_mode,
            aggregation=args.aggregation,
            calibration_tokens=actual_calibration_tokens,
        ),
        "global_union": {
            "variant": "global_union",
            "variant_type": "static_cluster_union",
            "rank": args.rank,
            "cluster_count": args.clusters,
            "candidate_m": args.candidate_m,
            "reference_k": args.reference_k,
            "score_mode": args.score_mode,
            "aggregation": args.aggregation,
            "calibration_tokens": actual_calibration_tokens,
            "tokens": 0,
            "samples": 0,
            "loss_sum": 0.0,
            "kl_sum": 0.0,
            "final_hidden_cosine_sum": 0.0,
            "active_size_sum": 0.0,
            "active_fraction_sum": 0.0,
            "union_recall_sum": 0.0,
            "union_score_ratio_sum": 0.0,
            "selection_count": 0.0,
            "union_metric_count": 0.0,
        },
    }
    union_caps = [int(value) for value in (args.union_caps or []) if int(value) > 0]
    union_layer_caps = [int(value) for value in (args.union_layer_caps or []) if int(value) > 0]
    if union_layer_caps and len(union_layer_caps) != len(dense.blocks):
        raise SystemExit(
            f"--union-layer-caps expects {len(dense.blocks)} values for this model, got {len(union_layer_caps)}"
        )
    for cap in union_caps:
        name = f"global_union_cap{cap}"
        buckets[name] = {
            "variant": name,
            "variant_type": "static_cluster_union_capped",
            "rank": args.rank,
            "cluster_count": args.clusters,
            "candidate_m": args.candidate_m,
            "active_cap": cap,
            "reference_k": args.reference_k,
            "score_mode": args.score_mode,
            "aggregation": args.aggregation,
            "calibration_tokens": actual_calibration_tokens,
            "tokens": 0,
            "samples": 0,
            "loss_sum": 0.0,
            "kl_sum": 0.0,
            "final_hidden_cosine_sum": 0.0,
            "active_size_sum": 0.0,
            "active_fraction_sum": 0.0,
            "union_recall_sum": 0.0,
            "union_score_ratio_sum": 0.0,
            "selection_count": 0.0,
            "union_metric_count": 0.0,
        }
    if union_layer_caps:
        buckets["global_union_layercaps"] = {
            "variant": "global_union_layercaps",
            "variant_type": "static_cluster_union_layer_capped",
            "rank": args.rank,
            "cluster_count": args.clusters,
            "candidate_m": args.candidate_m,
            "active_cap_by_layer": union_layer_caps,
            "reference_k": args.reference_k,
            "score_mode": args.score_mode,
            "aggregation": args.aggregation,
            "calibration_tokens": actual_calibration_tokens,
            "tokens": 0,
            "samples": 0,
            "loss_sum": 0.0,
            "kl_sum": 0.0,
            "final_hidden_cosine_sum": 0.0,
            "active_size_sum": 0.0,
            "active_fraction_sum": 0.0,
            "union_recall_sum": 0.0,
            "union_score_ratio_sum": 0.0,
            "selection_count": 0.0,
            "union_metric_count": 0.0,
        }
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        dense_bucket = buckets["dense"]
        dense_bucket["tokens"] += token_count
        dense_bucket["samples"] += sample_count
        dense_bucket["loss_sum"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        dense_bucket["final_hidden_cosine_sum"] += float(sample_count)

        variants = (
            (
                "static_cluster",
                _svd_static_cluster_pool_full_stack_logits(
                    dense,
                    svd_factors,
                    codebook,
                    tokens,
                    rank=args.rank,
                    reference_k=args.reference_k,
                    profile=False,
                ),
            ),
            (
                "global_union",
                _svd_static_union_full_stack_logits(
                    dense,
                    codebook,
                    tokens,
                    reference_k=args.reference_k,
                ),
            ),
        )
        cap_variants = tuple(
            (
                f"global_union_cap{cap}",
                _svd_static_union_full_stack_logits(
                    dense,
                    codebook,
                    tokens,
                    reference_k=args.reference_k,
                    cap=cap,
                ),
            )
            for cap in union_caps
        )
        layer_cap_variants = ()
        if union_layer_caps:
            layer_cap_variants = (
                (
                    "global_union_layercaps",
                    _svd_static_union_full_stack_logits(
                        dense,
                        codebook,
                        tokens,
                        reference_k=args.reference_k,
                        layer_caps=union_layer_caps,
                    ),
                ),
            )
        for name, (logits, hidden, aux) in (*variants, *cap_variants, *layer_cap_variants):
            bucket = buckets[name]
            loss = F.cross_entropy(logits.flatten(0, -2), target_flat, reduction="sum")
            cand_logp = F.log_softmax(logits / args.temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - cand_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            final_cos = F.cosine_similarity(
                hidden.float().flatten(1),
                dense_hidden.float().flatten(1),
                dim=-1,
            ).sum()
            bucket["tokens"] += token_count
            bucket["samples"] += sample_count
            bucket["loss_sum"] += float(loss.detach().cpu())
            bucket["kl_sum"] += float(kl.detach().cpu())
            bucket["final_hidden_cosine_sum"] += float(final_cos.detach().cpu())
            for key in (
                "candidate_size_sum",
                "candidate_recall_sum",
                "score_retention_sum",
                "selection_count",
                "cluster_count_sum",
                "nonempty_cluster_count_sum",
                "empty_cluster_count_sum",
                "max_cluster_size_sum",
                "min_cluster_size_sum",
                "mean_cluster_size_sum",
                "cluster_imbalance_sum",
                "cluster_metric_count",
                "cluster_pool_ffn_flop_ratio_sum",
                "active_size_sum",
                "active_fraction_sum",
                "union_recall_sum",
                "union_score_ratio_sum",
                "union_metric_count",
            ):
                if key in bucket:
                    bucket[key] += float(aux.get(key, 0.0))
        if device.type == "mps":
            torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "static_cluster_pool_union_eval_batch", "batch": batch_idx + 1}))
    rows = [
        _finish_cluster_pool_bucket(buckets["dense"]),
        _finish_cluster_pool_bucket(buckets["static_cluster"]),
        _finish_union_eval_bucket(buckets["global_union"]),
    ]
    rows.extend(_finish_union_eval_bucket(buckets[f"global_union_cap{cap}"]) for cap in union_caps)
    if union_layer_caps:
        rows.append(_finish_union_eval_bucket(buckets["global_union_layercaps"]))
    for row in rows:
        row["calibration_avg_nonempty_clusters"] = codebook_aux["calibration_avg_nonempty_clusters"]
        row["calibration_avg_cluster_imbalance"] = codebook_aux["calibration_avg_cluster_imbalance"]
    report = {
        "mode": "static_cluster_pool_union_eval",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "calibration_tokens": actual_calibration_tokens,
        "rank": args.rank,
        "clusters": args.clusters,
        "candidate_m": args.candidate_m,
        "union_caps": union_caps,
        "union_layer_caps": union_layer_caps,
        "reference_k": args.reference_k,
        "score_mode": args.score_mode,
        "aggregation": args.aggregation,
        "cluster_iters": args.cluster_iters,
        "temperature": args.temperature,
        "attention": "dense",
        "notes": "global_union uses the unique union of all static cluster-pool candidate ids per layer",
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


@torch.no_grad()
def _perturb_dense_ffn_weights(dense: DenseModel, *, relative_rms: float, seed: int) -> None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for block in dense.blocks:
        for param in (block.mlp.wug.weight, block.mlp.wd.weight):
            rms = param.detach().float().square().mean().sqrt().cpu()
            noise = torch.randn(param.shape, generator=gen, dtype=torch.float32) * (float(relative_rms) * rms)
            param.add_(noise.to(device=param.device, dtype=param.dtype))


def cmd_static_cluster_pool_staleness(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    actual_calibration_tokens = calibration_batches * tokens_per_batch
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    base_state = copy.deepcopy(dense.state_dict())
    base_factors = _build_svd_factor_cache(dense, max_rank=args.rank, device=device)
    base_codebook, base_codebook_aux = _build_static_svd_cluster_pool_codebook(
        dense,
        base_factors,
        streams.train_batches(config.training),
        calibration_batches=calibration_batches,
        device=device,
        rank=args.rank,
        cluster_count=args.clusters,
        candidate_m=args.candidate_m,
        score_mode=args.score_mode,
        aggregation=args.aggregation,
        cluster_iters=args.cluster_iters,
    )
    rows: list[dict[str, Any]] = []
    for pct in args.perturb_pct:
        dense.load_state_dict(base_state)
        relative = float(pct) / 100.0
        if relative > 0.0:
            _perturb_dense_ffn_weights(
                dense,
                relative_rms=relative,
                seed=(args.seed if args.seed is not None else config.training.seed) + int(round(pct * 1000)),
            )
        refreshed_factors = _build_svd_factor_cache(dense, max_rank=args.rank, device=device)
        refreshed_codebook, refreshed_codebook_aux = _build_static_svd_cluster_pool_codebook(
            dense,
            refreshed_factors,
            streams.train_batches(config.training),
            calibration_batches=calibration_batches,
            device=device,
            rank=args.rank,
            cluster_count=args.clusters,
            candidate_m=args.candidate_m,
            score_mode=args.score_mode,
            aggregation=args.aggregation,
            cluster_iters=args.cluster_iters,
        )
        totals = {
            "tokens": 0,
            "samples": 0,
            "dense_loss": 0.0,
            "stale_loss": 0.0,
            "refresh_loss": 0.0,
            "stale_kl": 0.0,
            "refresh_kl": 0.0,
            "stale_cos": 0.0,
            "refresh_cos": 0.0,
            "stale_recall": 0.0,
            "refresh_recall": 0.0,
            "selection_count": 0.0,
        }
        batches = streams.eval_batches(config.training)
        for batch_idx in range(eval_batches):
            tokens, targets = next(batches)
            tokens = tokens.to(device)
            targets = targets.to(device)
            dense_out = dense(tokens, targets, return_loss_per_sample=True)
            assert dense_out.logits is not None
            assert dense_out.loss_per_sample is not None
            assert dense_out.meta.hidden is not None
            dense_logp = F.log_softmax(dense_out.logits.detach() / args.temperature, dim=-1)
            target_flat = targets.flatten()
            stale_logits, stale_hidden, stale_aux = _svd_static_cluster_pool_full_stack_logits(
                dense,
                base_factors,
                base_codebook,
                tokens,
                rank=args.rank,
                reference_k=args.reference_k,
            )
            refresh_logits, refresh_hidden, refresh_aux = _svd_static_cluster_pool_full_stack_logits(
                dense,
                refreshed_factors,
                refreshed_codebook,
                tokens,
                rank=args.rank,
                reference_k=args.reference_k,
            )
            stale_logp = F.log_softmax(stale_logits / args.temperature, dim=-1)
            refresh_logp = F.log_softmax(refresh_logits / args.temperature, dim=-1)
            stale_loss = F.cross_entropy(stale_logits.flatten(0, -2), target_flat, reduction="sum")
            refresh_loss = F.cross_entropy(refresh_logits.flatten(0, -2), target_flat, reduction="sum")
            stale_kl = (dense_logp.exp() * (dense_logp - stale_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            refresh_kl = (dense_logp.exp() * (dense_logp - refresh_logp)).sum(dim=-1).sum() * (
                args.temperature * args.temperature
            )
            stale_cos = F.cosine_similarity(
                stale_hidden.float().flatten(1),
                dense_out.meta.hidden.float().flatten(1),
                dim=-1,
            ).sum()
            refresh_cos = F.cosine_similarity(
                refresh_hidden.float().flatten(1),
                dense_out.meta.hidden.float().flatten(1),
                dim=-1,
            ).sum()
            token_count = int(targets.numel())
            sample_count = int(targets.shape[0])
            totals["tokens"] += token_count
            totals["samples"] += sample_count
            totals["dense_loss"] += float(dense_out.loss_per_sample.sum().detach().cpu())
            totals["stale_loss"] += float(stale_loss.detach().cpu())
            totals["refresh_loss"] += float(refresh_loss.detach().cpu())
            totals["stale_kl"] += float(stale_kl.detach().cpu())
            totals["refresh_kl"] += float(refresh_kl.detach().cpu())
            totals["stale_cos"] += float(stale_cos.detach().cpu())
            totals["refresh_cos"] += float(refresh_cos.detach().cpu())
            totals["stale_recall"] += float(stale_aux.get("candidate_recall_sum", 0.0))
            totals["refresh_recall"] += float(refresh_aux.get("candidate_recall_sum", 0.0))
            totals["selection_count"] += float(stale_aux.get("selection_count", 0.0))
            if device.type == "mps":
                torch.mps.empty_cache()
            if args.progress and (batch_idx + 1) % args.progress == 0:
                print(json.dumps({"event": "staleness_eval_batch", "perturb_pct": pct, "batch": batch_idx + 1}))
        tokens_seen = max(int(totals["tokens"]), 1)
        samples_seen = max(int(totals["samples"]), 1)
        selection_count = max(float(totals["selection_count"]), 1.0)
        rows.append(
            {
                "perturb_pct": pct,
                "dense_nll_per_token": totals["dense_loss"] / tokens_seen,
                "stale_nll_per_token": totals["stale_loss"] / tokens_seen,
                "refreshed_nll_per_token": totals["refresh_loss"] / tokens_seen,
                "stale_kl_to_dense": totals["stale_kl"] / tokens_seen,
                "refreshed_kl_to_dense": totals["refresh_kl"] / tokens_seen,
                "stale_final_hidden_cosine": totals["stale_cos"] / samples_seen,
                "refreshed_final_hidden_cosine": totals["refresh_cos"] / samples_seen,
                "stale_candidate_recall": totals["stale_recall"] / selection_count,
                "refreshed_candidate_recall": totals["refresh_recall"] / selection_count,
                "base_calibration_avg_cluster_imbalance": base_codebook_aux[
                    "calibration_avg_cluster_imbalance"
                ],
                "refreshed_calibration_avg_cluster_imbalance": refreshed_codebook_aux[
                    "calibration_avg_cluster_imbalance"
                ],
            }
        )
    report = {
        "mode": "static_cluster_pool_staleness",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "calibration_tokens": actual_calibration_tokens,
        "rank": args.rank,
        "clusters": args.clusters,
        "candidate_m": args.candidate_m,
        "reference_k": args.reference_k,
        "score_mode": args.score_mode,
        "aggregation": args.aggregation,
        "cluster_iters": args.cluster_iters,
        "temperature": args.temperature,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_static_cluster_pool_continuation(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    training = config.training
    if args.batch_size is not None:
        training = dataclasses.replace(training, batch_size=args.batch_size)
    if args.seq_len is not None:
        training = dataclasses.replace(training, seq_len=args.seq_len)
    if args.lr is not None:
        training = dataclasses.replace(training, lr=args.lr)
    config = dataclasses.replace(config, training=training)
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    actual_calibration_tokens = calibration_batches * tokens_per_batch
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    lr = args.lr if args.lr is not None else config.training.lr
    weight_decay = args.weight_decay if args.weight_decay is not None else config.training.weight_decay
    grad_clip_norm = args.grad_clip_norm if args.grad_clip_norm is not None else config.training.grad_clip_norm
    checkpoint_payload = None
    optimizer_state = None
    if args.resume_optimizer_state:
        checkpoint_payload = torch.load(args.dense_checkpoint, map_location=device, weights_only=False)
        optimizer_state = checkpoint_payload.get("optimizer")
        if optimizer_state is None:
            raise SystemExit(f"checkpoint has no optimizer state: {args.dense_checkpoint}")
    eval_steps = {int(step) for step in (args.eval_steps or [0, args.steps])}
    eval_steps = {step for step in eval_steps if 0 <= step <= args.steps}
    eval_steps.add(0)
    eval_steps.add(args.steps)
    alignment_config = config
    if args.alignment_batch_size is not None:
        alignment_config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.alignment_batch_size),
        )
    rows: list[dict[str, Any]] = []
    sparse_ffn_kind = str(args.sparse_ffn_kind)
    active_union_cap = int(args.active_union_cap) if args.active_union_cap is not None else None
    active_union_layer_caps = [int(value) for value in (args.active_union_layer_caps or [])]
    if active_union_layer_caps and len(active_union_layer_caps) != config.model.n_dense_layers:
        raise SystemExit(
            f"--active-union-layer-caps expects {config.model.n_dense_layers} values, "
            f"got {len(active_union_layer_caps)}"
        )
    if sparse_ffn_kind == "active_union_packed" and any(int(x) > 0 for x in (args.refresh_intervals or [])):
        raise SystemExit("active_union_packed has no dense master for refresh; use --refresh-intervals 0")

    if args.run_dense:
        dense = _load_dense_model(config, args.dense_checkpoint, device)

        def dense_eval_callback() -> float:
            return _eval_lm_nll(
                dense,
                streams,
                config,
                batches=eval_batches,
                device=device,
            )

        train_metrics = _train_lm_continuation(
            dense,
            streams,
            config,
            steps=args.steps,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            grad_clip_norm=grad_clip_norm,
            optimizer_state=copy.deepcopy(optimizer_state) if optimizer_state is not None else None,
            eval_steps=eval_steps,
            eval_callback=dense_eval_callback,
            progress=args.progress,
            label="dense",
        )
        curve = {int(row["step"]): float(row["nll_per_token"]) for row in train_metrics["eval_curve"]}
        dense_initial_nll = curve.get(0, dense_eval_callback())
        dense_final_nll = curve.get(args.steps, dense_eval_callback())
        rows.append(
            {
                "variant": "dense_continuation",
                "initial_nll_per_token": dense_initial_nll,
                "final_nll_per_token": dense_final_nll,
                "nll_delta": dense_final_nll - dense_initial_nll,
                **train_metrics,
            }
        )
        del dense
        if device.type == "mps":
            torch.mps.empty_cache()

    refresh_intervals = args.refresh_intervals or [0]
    if not args.run_sparse:
        refresh_intervals = []
    for refresh_interval in refresh_intervals:
        sparse = _load_dense_model(config, args.dense_checkpoint, device)
        if sparse_ffn_kind == "static_cluster":
            refresh_aux = _refresh_static_cluster_pool_ffns(
                sparse,
                streams,
                config,
                calibration_batches=calibration_batches,
                device=device,
                rank=args.rank,
                cluster_count=args.clusters,
                candidate_m=args.candidate_m,
                score_mode=args.score_mode,
                aggregation=args.aggregation,
                cluster_iters=args.cluster_iters,
            )
        elif sparse_ffn_kind in {"active_union_indexed", "active_union_packed"}:
            refresh_aux = _refresh_active_union_ffns(
                sparse,
                streams,
                config,
                calibration_batches=calibration_batches,
                device=device,
                rank=args.rank,
                cluster_count=args.clusters,
                candidate_m=args.candidate_m,
                score_mode=args.score_mode,
                aggregation=args.aggregation,
                cluster_iters=args.cluster_iters,
                cap=active_union_cap,
                layer_caps=active_union_layer_caps or None,
                packed=sparse_ffn_kind == "active_union_packed",
            )
        else:
            raise AssertionError(f"unknown sparse FFN kind: {sparse_ffn_kind}")
        sparse_initial_nll = _eval_lm_nll(
            sparse,
            streams,
            config,
            batches=eval_batches,
            device=device,
        )
        initial_coverage = (
            _static_cluster_pool_coverage_metrics(sparse)
            if sparse_ffn_kind == "static_cluster"
            else _active_union_coverage_metrics(sparse)
        )
        initial_alignment = (
            _static_cluster_pool_input_grad_alignment(
                sparse,
                streams,
                alignment_config,
                device=device,
                batches=args.alignment_batches,
                seed=(args.seed if args.seed is not None else config.training.seed) + int(refresh_interval),
            )
            if args.include_gradient_alignment and sparse_ffn_kind == "static_cluster"
            else {}
        )

        def refresh_callback() -> dict[str, float]:
            if refresh_interval <= 0:
                return {}
            if sparse_ffn_kind == "static_cluster":
                return _refresh_static_cluster_pool_ffns(
                    sparse,
                    streams,
                    config,
                    calibration_batches=calibration_batches,
                    device=device,
                    rank=args.rank,
                    cluster_count=args.clusters,
                    candidate_m=args.candidate_m,
                    score_mode=args.score_mode,
                    aggregation=args.aggregation,
                    cluster_iters=args.cluster_iters,
                )
            return _refresh_active_union_ffns(
                sparse,
                streams,
                config,
                calibration_batches=calibration_batches,
                device=device,
                rank=args.rank,
                cluster_count=args.clusters,
                candidate_m=args.candidate_m,
                score_mode=args.score_mode,
                aggregation=args.aggregation,
                cluster_iters=args.cluster_iters,
                cap=active_union_cap,
                layer_caps=active_union_layer_caps or None,
                packed=False,
            )

        def sparse_eval_callback() -> float:
            return _eval_lm_nll(
                sparse,
                streams,
                config,
                batches=eval_batches,
                device=device,
            )

        train_metrics = _train_lm_continuation(
            sparse,
            streams,
            config,
            steps=args.steps,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            refresh_interval=max(0, int(refresh_interval)),
            refresh_callback=refresh_callback,
            grad_clip_norm=grad_clip_norm,
            optimizer_state=(
                None
                if sparse_ffn_kind == "active_union_packed"
                else copy.deepcopy(optimizer_state)
                if optimizer_state is not None
                else None
            ),
            eval_steps=eval_steps,
            eval_callback=sparse_eval_callback,
            progress=args.progress,
            label=f"{sparse_ffn_kind}_refresh_{refresh_interval}",
        )
        curve = {int(row["step"]): float(row["nll_per_token"]) for row in train_metrics["eval_curve"]}
        sparse_final_nll = curve.get(args.steps, sparse_eval_callback())
        final_coverage = (
            _static_cluster_pool_coverage_metrics(sparse)
            if sparse_ffn_kind == "static_cluster"
            else _active_union_coverage_metrics(sparse)
        )
        final_grad_row_metrics = (
            _static_cluster_pool_grad_row_metrics(sparse)
            if sparse_ffn_kind == "static_cluster"
            else {}
        )
        final_alignment = (
            _static_cluster_pool_input_grad_alignment(
                sparse,
                streams,
                alignment_config,
                device=device,
                batches=args.alignment_batches,
                seed=(args.seed if args.seed is not None else config.training.seed)
                + 10_000
                + int(refresh_interval),
            )
            if args.include_gradient_alignment and sparse_ffn_kind == "static_cluster"
            else {}
        )
        rows.append(
            {
                "variant": "static_cluster_sparse_ffn"
                if sparse_ffn_kind == "static_cluster"
                else sparse_ffn_kind,
                "refresh_interval": int(refresh_interval),
                "sparse_ffn_kind": sparse_ffn_kind,
                "active_union_cap": active_union_cap,
                "active_union_layer_caps": active_union_layer_caps,
                "initial_nll_per_token": sparse_initial_nll,
                "final_nll_per_token": sparse_final_nll,
                "nll_delta": sparse_final_nll - sparse_initial_nll,
                "initial_calibration_avg_cluster_imbalance": refresh_aux[
                    "calibration_avg_cluster_imbalance"
                ],
                "initial_calibration_avg_nonempty_clusters": refresh_aux[
                    "calibration_avg_nonempty_clusters"
                ],
                "initial_coverage": initial_coverage,
                "final_coverage": final_coverage,
                "final_grad_row_metrics": final_grad_row_metrics,
                "initial_gradient_alignment": initial_alignment,
                "final_gradient_alignment": final_alignment,
                **train_metrics,
            }
        )
        del sparse
        if device.type == "mps":
            torch.mps.empty_cache()

    dense_curve_by_step: dict[int, float] = {}
    for row in rows:
        if row["variant"] == "dense_continuation":
            dense_curve_by_step = {
                int(point["step"]): float(point["nll_per_token"])
                for point in row.get("eval_curve", [])
            }
            break
    if dense_curve_by_step:
        for row in rows:
            if row["variant"] == "dense_continuation":
                continue
            for point in row.get("eval_curve", []):
                step = int(point["step"])
                if step in dense_curve_by_step:
                    point["gap_vs_dense"] = float(point["nll_per_token"]) - dense_curve_by_step[step]
            row["initial_gap_vs_dense"] = row["initial_nll_per_token"] - dense_curve_by_step.get(
                0,
                row["initial_nll_per_token"],
            )
            row["final_gap_vs_dense"] = row["final_nll_per_token"] - dense_curve_by_step.get(
                args.steps,
                row["final_nll_per_token"],
            )

    report = {
        "mode": "static_cluster_pool_continuation",
        "checkpoint": args.dense_checkpoint,
        "device": str(device),
        "steps": args.steps,
        "batch_size": config.training.batch_size,
        "seq_len": config.training.seq_len,
        "tokens_per_batch": tokens_per_batch,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "eval_steps": sorted(eval_steps),
        "calibration_tokens": actual_calibration_tokens,
        "rank": args.rank,
        "clusters": args.clusters,
        "candidate_m": args.candidate_m,
        "sparse_ffn_kind": sparse_ffn_kind,
        "active_union_cap": active_union_cap,
        "active_union_layer_caps": active_union_layer_caps,
        "score_mode": args.score_mode,
        "aggregation": args.aggregation,
        "cluster_iters": args.cluster_iters,
        "lr": lr,
        "weight_decay": weight_decay,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1), dim=-1).item())


def _norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(a.float().norm().item() / max(b.float().norm().item(), 1e-12))


def cmd_static_cluster_pool_gradient_alignment(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = 1
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    factors = _build_svd_factor_cache(dense, max_rank=args.rank, device=device)
    codebook, codebook_aux = _build_static_svd_cluster_pool_codebook(
        dense,
        factors,
        streams.train_batches(config.training),
        calibration_batches=calibration_batches,
        device=device,
        rank=args.rank,
        cluster_count=args.clusters,
        candidate_m=args.candidate_m,
        score_mode=args.score_mode,
        aggregation=args.aggregation,
        cluster_iters=args.cluster_iters,
    )
    layer_totals: list[dict[str, float]] = [
        {
            "count": 0.0,
            "ffn_output_cosine": 0.0,
            "ffn_output_mse": 0.0,
            "input_grad_cosine": 0.0,
            "w_up_grad_cosine": 0.0,
            "w_gate_grad_cosine": 0.0,
            "w_down_grad_cosine": 0.0,
            "w_up_grad_norm_ratio": 0.0,
            "w_gate_grad_norm_ratio": 0.0,
            "w_down_grad_norm_ratio": 0.0,
            "selected_rows": 0.0,
        }
        for _ in dense.blocks
    ]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed if args.seed is not None else config.training.seed)
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, _ = next(batches)
        tokens = tokens.to(device)
        with torch.no_grad():
            x = dense.embed(tokens)
        for layer_idx, block in enumerate(dense.blocks):
            with torch.no_grad():
                u = x + block.attn(block.norm1(x))
                normed = block.norm2(u).detach()
            dense_mlp = copy.deepcopy(block.mlp).to(device)
            sparse_mlp = StaticClusterPoolSwiGLU(
                block.mlp,
                factors[layer_idx],
                codebook[layer_idx],
                rank=args.rank,
                sparse_enabled=True,
            ).to(device)
            dense_mlp.train()
            sparse_mlp.train()
            x_dense = normed.detach().clone().requires_grad_()
            x_sparse = normed.detach().clone().requires_grad_()
            dense_out = dense_mlp(x_dense)
            sparse_out = sparse_mlp(x_sparse)
            upstream = torch.randn(
                dense_out.shape,
                generator=gen,
                device="cpu",
                dtype=torch.float32,
            ).to(device=device, dtype=dense_out.dtype)
            (dense_out * upstream).sum().backward()
            (sparse_out * upstream).sum().backward()
            assignments = route_to_static_centers(
                normed.reshape(-1, normed.shape[-1]),
                sparse_mlp.up_a[:, : sparse_mlp.rank],
                sparse_mlp.gate_a[:, : sparse_mlp.rank],
                sparse_mlp.centers,
            )
            selected_rows = sparse_mlp.candidate_ids.index_select(0, assignments).reshape(-1).unique()
            dense_up_grad, dense_gate_grad = dense_mlp.wug.weight.grad.chunk(2, dim=0)
            sparse_up_grad, sparse_gate_grad = sparse_mlp.wug.weight.grad.chunk(2, dim=0)
            dense_down_grad = dense_mlp.wd.weight.grad.t().contiguous()
            sparse_down_grad = sparse_mlp.wd.weight.grad.t().contiguous()
            dense_up_sel = dense_up_grad.index_select(0, selected_rows)
            sparse_up_sel = sparse_up_grad.index_select(0, selected_rows)
            dense_gate_sel = dense_gate_grad.index_select(0, selected_rows)
            sparse_gate_sel = sparse_gate_grad.index_select(0, selected_rows)
            dense_down_sel = dense_down_grad.index_select(0, selected_rows)
            sparse_down_sel = sparse_down_grad.index_select(0, selected_rows)
            totals = layer_totals[layer_idx]
            totals["count"] += 1.0
            totals["ffn_output_cosine"] += _cosine_flat(dense_out.detach(), sparse_out.detach())
            totals["ffn_output_mse"] += float((dense_out.detach().float() - sparse_out.detach().float()).square().mean().item())
            totals["input_grad_cosine"] += _cosine_flat(x_dense.grad, x_sparse.grad)
            totals["w_up_grad_cosine"] += _cosine_flat(dense_up_sel, sparse_up_sel)
            totals["w_gate_grad_cosine"] += _cosine_flat(dense_gate_sel, sparse_gate_sel)
            totals["w_down_grad_cosine"] += _cosine_flat(dense_down_sel, sparse_down_sel)
            totals["w_up_grad_norm_ratio"] += _norm_ratio(sparse_up_sel, dense_up_sel)
            totals["w_gate_grad_norm_ratio"] += _norm_ratio(sparse_gate_sel, dense_gate_sel)
            totals["w_down_grad_norm_ratio"] += _norm_ratio(sparse_down_sel, dense_down_sel)
            totals["selected_rows"] += float(selected_rows.numel())
            with torch.no_grad():
                x = u + block.mlp(normed)
            del dense_mlp, sparse_mlp, x_dense, x_sparse, dense_out, sparse_out
            if device.type == "mps":
                torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "gradient_alignment_batch", "batch": batch_idx + 1}))
    rows: list[dict[str, Any]] = []
    for layer_idx, totals in enumerate(layer_totals):
        count = max(totals.pop("count"), 1.0)
        row = {"layer": layer_idx}
        row.update({key: value / count for key, value in totals.items()})
        rows.append(row)
    mean_row = {
        "layer": "mean",
        **{
            key: sum(float(row[key]) for row in rows) / max(len(rows), 1)
            for key in rows[0]
            if key != "layer"
        },
    }
    report = {
        "mode": "static_cluster_pool_gradient_alignment",
        "checkpoint": args.dense_checkpoint,
        "device": str(device),
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "calibration_tokens": calibration_batches * tokens_per_batch,
        "rank": args.rank,
        "clusters": args.clusters,
        "candidate_m": args.candidate_m,
        "score_mode": args.score_mode,
        "aggregation": args.aggregation,
        "cluster_iters": args.cluster_iters,
        "codebook_aux": codebook_aux,
        "rows": [*rows, mean_row],
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_deferred_neuron_svd_hot_eval(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    if args.batch_size is not None:
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, batch_size=args.batch_size),
        )
    tokens_per_batch = config.training.batch_size * config.training.seq_len
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = max(1, math.ceil(config.data.eval_tokens / tokens_per_batch))
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    dense.eval()
    svd_factors = _build_svd_factor_cache(dense, max_rank=args.rank, device=device)
    sparse_ffns = _build_svd_sparse_ffns(
        dense,
        rank=args.rank,
        k=args.k,
        up_m=args.factor_m,
        gate_m=args.factor_m,
        product_m=args.product_factor_m,
        device=device,
        candidate_mode=args.candidate_mode,
        svd_factors=svd_factors,
    )
    totals = {
        "dense_loss": 0.0,
        "oracle_loss": 0.0,
        "hot_loss": 0.0,
        "oracle_kl": 0.0,
        "hot_kl": 0.0,
        "oracle_cos": 0.0,
        "hot_cos": 0.0,
        "hot_avg_candidate_size_sum": 0.0,
        "hot_selection_count": 0.0,
        "tokens": 0,
        "samples": 0,
        "max_logit_abs_diff": 0.0,
        "max_hidden_abs_diff": 0.0,
        "selector_score_time": 0.0,
        "candidate_union_dedup_time": 0.0,
        "exact_candidate_activation_time": 0.0,
        "rerank_topk_time": 0.0,
        "down_sum_time": 0.0,
    }
    batches = streams.eval_batches(config.training)
    for batch_idx in range(eval_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        dense_out = dense(tokens, targets, return_loss_per_sample=True)
        assert dense_out.logits is not None
        assert dense_out.loss_per_sample is not None
        assert dense_out.meta.hidden is not None
        dense_logits = dense_out.logits
        dense_hidden = dense_out.meta.hidden
        dense_logp = F.log_softmax(dense_logits.detach() / args.temperature, dim=-1)
        target_flat = targets.flatten()
        oracle_logits, oracle_hidden, _ = _svd_factor_union_full_stack_logits(
            dense,
            svd_factors,
            tokens,
            rank=args.rank,
            k=args.k,
            up_m=args.factor_m,
            gate_m=args.factor_m,
            product_m=args.product_factor_m,
            reranker="norm",
        )
        hot_logits, hot_hidden, hot_aux = _svd_hot_full_stack_logits(
            dense,
            sparse_ffns,
            tokens,
            profile=args.profile,
        )
        token_count = int(targets.numel())
        sample_count = int(targets.shape[0])
        oracle_loss = F.cross_entropy(oracle_logits.flatten(0, -2), target_flat, reduction="sum")
        hot_loss = F.cross_entropy(hot_logits.flatten(0, -2), target_flat, reduction="sum")
        oracle_logp = F.log_softmax(oracle_logits / args.temperature, dim=-1)
        hot_logp = F.log_softmax(hot_logits / args.temperature, dim=-1)
        oracle_kl = (dense_logp.exp() * (dense_logp - oracle_logp)).sum(dim=-1).sum() * (
            args.temperature * args.temperature
        )
        hot_kl = (dense_logp.exp() * (dense_logp - hot_logp)).sum(dim=-1).sum() * (
            args.temperature * args.temperature
        )
        oracle_cos = F.cosine_similarity(
            oracle_hidden.float().flatten(1),
            dense_hidden.float().flatten(1),
            dim=-1,
        ).sum()
        hot_cos = F.cosine_similarity(
            hot_hidden.float().flatten(1),
            dense_hidden.float().flatten(1),
            dim=-1,
        ).sum()
        totals["dense_loss"] += float(dense_out.loss_per_sample.sum().detach().cpu())
        totals["oracle_loss"] += float(oracle_loss.detach().cpu())
        totals["hot_loss"] += float(hot_loss.detach().cpu())
        totals["oracle_kl"] += float(oracle_kl.detach().cpu())
        totals["hot_kl"] += float(hot_kl.detach().cpu())
        totals["oracle_cos"] += float(oracle_cos.detach().cpu())
        totals["hot_cos"] += float(hot_cos.detach().cpu())
        totals["hot_avg_candidate_size_sum"] += float(hot_aux["candidate_size_sum"])
        totals["hot_selection_count"] += float(hot_aux["selection_count"])
        for key in (
            "selector_score_time",
            "candidate_union_dedup_time",
            "exact_candidate_activation_time",
            "rerank_topk_time",
            "down_sum_time",
        ):
            totals[key] += float(hot_aux.get(key, 0.0))
        totals["tokens"] += token_count
        totals["samples"] += sample_count
        totals["max_logit_abs_diff"] = max(
            totals["max_logit_abs_diff"],
            float((hot_logits - oracle_logits).detach().abs().max().cpu()),
        )
        totals["max_hidden_abs_diff"] = max(
            totals["max_hidden_abs_diff"],
            float((hot_hidden - oracle_hidden).detach().abs().max().cpu()),
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        if args.progress and (batch_idx + 1) % args.progress == 0:
            print(json.dumps({"event": "svd_hot_eval_batch", "batch": batch_idx + 1}))
    tokens = max(int(totals["tokens"]), 1)
    samples = max(int(totals["samples"]), 1)
    report = {
        "mode": "deferred_neuron_svd_hot_eval",
        "checkpoint": args.dense_checkpoint,
        "eval_batches": eval_batches,
        "eval_tokens": eval_batches * tokens_per_batch,
        "requested_eval_tokens": config.data.eval_tokens,
        "rank": args.rank,
        "k": args.k,
        "factor_m": args.factor_m,
        "product_factor_m": args.product_factor_m,
        "candidate_mode": args.candidate_mode,
        "reranker": "norm",
        "temperature": args.temperature,
        "attention": "dense",
        "rows": [
            {
                "variant": "dense",
                "nll_per_token": totals["dense_loss"] / tokens,
                "kl_to_dense": 0.0,
                "final_hidden_cosine": 1.0,
            },
            {
                "variant": "svd_oracle_norm",
                "nll_per_token": totals["oracle_loss"] / tokens,
                "kl_to_dense": totals["oracle_kl"] / tokens,
                "final_hidden_cosine": totals["oracle_cos"] / samples,
            },
            {
                "variant": "svd_hot_norm",
                "nll_per_token": totals["hot_loss"] / tokens,
                "kl_to_dense": totals["hot_kl"] / tokens,
                "final_hidden_cosine": totals["hot_cos"] / samples,
                "avg_candidate_size": totals["hot_avg_candidate_size_sum"]
                / max(totals["hot_selection_count"], 1.0),
                "max_logit_abs_diff_vs_oracle": totals["max_logit_abs_diff"],
                "max_hidden_abs_diff_vs_oracle": totals["max_hidden_abs_diff"],
                "selector_score_time": totals["selector_score_time"] / max(eval_batches, 1),
                "candidate_union_dedup_time": totals["candidate_union_dedup_time"] / max(eval_batches, 1),
                "exact_candidate_activation_time": totals["exact_candidate_activation_time"] / max(eval_batches, 1),
                "rerank_topk_time": totals["rerank_topk_time"] / max(eval_batches, 1),
                "down_sum_time": totals["down_sum_time"] / max(eval_batches, 1),
            },
        ],
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _measure_cuda_graph_forward_backward(
    device: torch.device,
    forward_fn,
    grad_tensors: Sequence[torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[float | None, str | None]:
    """Measure a static-shape forward+backward CUDA graph replay.

    This is used only by synthetic FFN benchmarks. Dense and sparse variants
    call the same helper so graph launch savings are applied symmetrically.
    """

    if device.type != "cuda":
        return None, None
    try:
        for _ in range(max(1, int(warmup))):
            for tensor in grad_tensors:
                tensor.grad = None
            loss = forward_fn().square().mean()
            loss.backward()
        _sync_device(device)
        if any(tensor.grad is None for tensor in grad_tensors):
            raise RuntimeError("CUDA graph warmup did not populate all gradients")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for tensor in grad_tensors:
                assert tensor.grad is not None
                tensor.grad.zero_()
            loss = forward_fn().square().mean()
            loss.backward()
        _sync_device(device)
        graph.replay()
        _sync_device(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, int(iters))):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) / 1000.0) / max(1, int(iters)), None
    except Exception as exc:  # pragma: no cover - CUDA-only diagnostic path.
        return None, f"{type(exc).__name__}: {exc}"


def _parse_ffn_bench_size(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace(":", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("size must be d_modelxd_ffxtokens, e.g. 512x2048x128")
    d_model, d_ff, tokens = (int(part) for part in parts)
    if min(d_model, d_ff, tokens) <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return d_model, d_ff, tokens


@torch.no_grad()
def cmd_benchmark_svd_sparse_ffn(args: argparse.Namespace) -> None:
    device = default_device()
    rows: list[dict[str, Any]] = []
    sizes = args.size or [(64, 256, 2048), (512, 2048, 128)]
    for d_model, d_ff, tokens in sizes:
        rank = min(args.rank, d_model, d_ff)
        k = min(args.k, d_ff)
        factor_m = min(args.factor_m, d_ff)
        product_m = min(args.product_factor_m, d_ff)
        sparse = SVDFactorSparseFFN(
            d_model,
            d_ff,
            rank=rank,
            top_k=k,
            up_m=factor_m,
            product_m=product_m,
            candidate_mode=args.candidate_mode,
            refresh_on_init=False,
        ).to(device)
        sparse.eval()
        x = torch.randn(tokens, d_model, device=device)

        def dense_forward() -> torch.Tensor:
            up = x @ sparse.w_up.t()
            gate = x @ sparse.w_gate.t()
            return (up * F.silu(gate)) @ sparse.w_down

        def sparse_forward() -> torch.Tensor:
            return sparse(x)

        for _ in range(args.warmup):
            dense_forward()
            sparse_forward()
        _sync_device(device)

        def measure(fn) -> float:
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            _sync_device(device)
            return (time.perf_counter() - start) / max(args.iters, 1)

        dense_seconds = measure(dense_forward)
        sparse_seconds = measure(sparse_forward)
        profile_aux: dict[str, Any] = {}
        if args.profile:
            _, aux = sparse(x, return_aux=True, profile=True)
            profile_aux = {
                key: float(aux.get(key, 0.0))
                for key in (
                    "selector_score_time",
                    "candidate_union_dedup_time",
                    "exact_candidate_activation_time",
                    "rerank_topk_time",
                    "down_sum_time",
                    "triton_sparse_ffn_time",
                    "avg_candidate_size",
                    "candidate_slots",
                )
            }
        dense_ops = 3.0 * d_model * d_ff
        sparse_ops = (
            2.0 * d_model * rank
            + 2.0 * rank * d_ff
            + 2.0 * d_model * min(2 * factor_m + product_m, d_ff)
            + d_model * k
        )
        rows.append(
            {
                "d_model": d_model,
                "d_ff": d_ff,
                "tokens": tokens,
                "rank": rank,
                "factor_m": factor_m,
                "product_factor_m": product_m,
                "candidate_mode": args.candidate_mode,
                "k": k,
                "dense_ms": dense_seconds * 1000.0,
                "sparse_ms": sparse_seconds * 1000.0,
                "measured_speedup": dense_seconds / max(sparse_seconds, 1e-12),
                "estimated_dense_ops_per_token": dense_ops,
                "estimated_sparse_ops_per_token": sparse_ops,
                "estimated_speedup": dense_ops / max(sparse_ops, 1.0),
                **profile_aux,
            }
        )
        if device.type == "mps":
            torch.mps.empty_cache()
    report = {
        "mode": "benchmark_svd_sparse_ffn",
        "device": str(device),
        "iters": args.iters,
        "warmup": args.warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


@torch.no_grad()
def cmd_benchmark_cluster_pool_ffn(args: argparse.Namespace) -> None:
    device = default_device()
    rows: list[dict[str, Any]] = []
    sizes = args.size or [(2048, 8192, 4096)]
    for d_model, d_ff, tokens in sizes:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        scale = 1.0 / math.sqrt(d_model)
        x = torch.randn(tokens, d_model, device=device, dtype=dtype)
        w_up = torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale
        w_gate = torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale
        w_down = torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale
        rank = min(args.rank, d_model, d_ff)
        up_a = torch.randn(d_model, rank, device=device, dtype=dtype) * scale
        gate_a = torch.randn(d_model, rank, device=device, dtype=dtype) * scale

        def dense_forward() -> torch.Tensor:
            up = x @ w_up.t()
            gate = x @ w_gate.t()
            return (up * F.silu(gate)) @ w_down

        for clusters in args.clusters:
            clusters = min(int(clusters), tokens)
            assignments = balanced_synthetic_assignments(tokens, clusters, device)
            centers = synthetic_cluster_centers(x, up_a, gate_a, assignments, clusters)
            max_capacity = args.max_tokens_per_cluster or suggested_cluster_capacity(
                tokens,
                clusters,
                args.capacity_factor,
            )
            max_capacity = min(max_capacity, tokens)
            for candidate_m in args.candidate_m:
                candidate_m = min(int(candidate_m), d_ff)
                candidate_ids = torch.randint(
                    0,
                    d_ff,
                    (clusters, candidate_m),
                    device=device,
                    dtype=torch.long,
                )
                wup_pool, wgate_pool, wdown_pool = prepare_cluster_pool_weights(
                    w_up,
                    w_gate,
                    w_down,
                    candidate_ids,
                )

                def assignment_forward() -> torch.Tensor:
                    return cluster_pool_ffn_forward_from_assignments(
                        x,
                        assignments,
                        wup_pool,
                        wgate_pool,
                        wdown_pool,
                        max_tokens_per_cluster=max_capacity,
                        block_d=args.block_d,
                    )

                def routed_forward() -> torch.Tensor:
                    return cluster_pool_ffn_forward_static(
                        x,
                        up_a,
                        gate_a,
                        centers,
                        wup_pool,
                        wgate_pool,
                        wdown_pool,
                        max_tokens_per_cluster=max_capacity,
                        block_d=args.block_d,
                    )

                for _ in range(args.warmup):
                    dense_forward()
                    assignment_forward()
                    routed_forward()
                _sync_device(device)

                def measure(fn) -> float:
                    start = time.perf_counter()
                    for _ in range(args.iters):
                        fn()
                    _sync_device(device)
                    return (time.perf_counter() - start) / max(args.iters, 1)

                dense_seconds = measure(dense_forward)
                cluster_seconds = measure(assignment_forward)
                routed_seconds = measure(routed_forward)
                _, overflow = cluster_pool_ffn_forward_from_assignments(
                    x,
                    assignments,
                    wup_pool,
                    wgate_pool,
                    wdown_pool,
                    max_tokens_per_cluster=max_capacity,
                    block_d=args.block_d,
                    return_overflow=True,
                )
                _, routed_overflow = cluster_pool_ffn_forward_static(
                    x,
                    up_a,
                    gate_a,
                    centers,
                    wup_pool,
                    wgate_pool,
                    wdown_pool,
                    max_tokens_per_cluster=max_capacity,
                    block_d=args.block_d,
                    return_overflow=True,
                )
                _sync_device(device)
                overflow_value = int(overflow.detach().cpu().item())
                routed_overflow_value = int(routed_overflow.detach().cpu().item())
                rows.append(
                    {
                        "d_model": d_model,
                        "d_ff": d_ff,
                        "tokens": tokens,
                        "clusters": clusters,
                        "candidate_m": candidate_m,
                        "rank": rank,
                        "max_tokens_per_cluster": max_capacity,
                        "overflow": overflow_value,
                        "routed_overflow": routed_overflow_value,
                        "dense_ms": dense_seconds * 1000.0,
                        "cluster_pool_ms": cluster_seconds * 1000.0,
                        "cluster_pool_routed_ms": routed_seconds * 1000.0,
                        "measured_speedup_cluster_pool": dense_seconds
                        / max(cluster_seconds, 1e-12),
                        "measured_speedup_routed": dense_seconds / max(routed_seconds, 1e-12),
                        "ideal_ffn_flop_ratio": candidate_m / d_ff,
                        "ideal_ffn_math_speedup": d_ff / max(candidate_m, 1),
                    }
                )
    report = {
        "mode": "benchmark_cluster_pool_ffn",
        "device": str(device),
        "iters": args.iters,
        "warmup": args.warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _benchmark_static_cluster_grad_check(
    *,
    device: torch.device,
    dtype: torch.dtype,
    d_model: int,
    d_ff: int,
    tokens: int,
) -> dict[str, float]:
    scale = 1.0 / math.sqrt(d_model)
    x_dense = torch.randn(tokens, d_model, device=device, dtype=dtype, requires_grad=True)
    w_up = (torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
    w_gate = (torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
    w_down = (torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
    x_sparse = x_dense.detach().clone().requires_grad_()
    wup_pool = w_up.detach().t().unsqueeze(0).contiguous().requires_grad_()
    wgate_pool = w_gate.detach().t().unsqueeze(0).contiguous().requires_grad_()
    wdown_pool = w_down.detach().unsqueeze(0).contiguous().requires_grad_()
    assignments = torch.zeros(tokens, device=device, dtype=torch.long)
    pack_index, flat_gather, _, _ = build_static_pack_gather_indices(assignments, cluster_count=1)
    dense = ((x_dense @ w_up.t()) * F.silu(x_dense @ w_gate.t())) @ w_down
    sparse = cluster_pool_ffn_forward_preindexed(
        x_sparse,
        pack_index,
        flat_gather,
        wup_pool,
        wgate_pool,
        wdown_pool,
    )
    max_output_abs_diff = float((dense.detach() - sparse.detach()).abs().max().cpu())
    dense.square().mean().backward()
    sparse.square().mean().backward()
    assert x_dense.grad is not None and x_sparse.grad is not None
    assert w_up.grad is not None and w_gate.grad is not None and w_down.grad is not None
    assert wup_pool.grad is not None and wgate_pool.grad is not None and wdown_pool.grad is not None
    return {
        "full_mode_max_output_abs_diff": max_output_abs_diff,
        "full_mode_max_x_grad_abs_diff": float((x_dense.grad - x_sparse.grad).abs().max().cpu()),
        "full_mode_max_w_up_grad_abs_diff": float((w_up.grad - wup_pool.grad[0].t()).abs().max().cpu()),
        "full_mode_max_w_gate_grad_abs_diff": float(
            (w_gate.grad - wgate_pool.grad[0].t()).abs().max().cpu()
        ),
        "full_mode_max_w_down_grad_abs_diff": float((w_down.grad - wdown_pool.grad[0]).abs().max().cpu()),
    }


def cmd_benchmark_static_cluster_pool_ffn_train(args: argparse.Namespace) -> None:
    device = default_device()
    rows: list[dict[str, Any]] = []
    sizes = args.size or [(2048, 8192, 1024)]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    grad_check: dict[str, float] = {}
    if args.grad_check:
        grad_check = _benchmark_static_cluster_grad_check(
            device=device,
            dtype=torch.float32,
            d_model=min(args.grad_check_d_model, 128),
            d_ff=min(args.grad_check_d_ff, 256),
            tokens=min(args.grad_check_tokens, 128),
        )
        _sync_device(device)
    for d_model, d_ff, tokens in sizes:
        scale = 1.0 / math.sqrt(d_model)
        x_dense = torch.randn(tokens, d_model, device=device, dtype=dtype, requires_grad=True)
        x_sparse = x_dense.detach().clone().requires_grad_()
        wug = (torch.randn(2 * d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
        w_down = (torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
        clusters = min(args.clusters, tokens)
        candidate_m = min(args.candidate_m, d_ff)
        assignments = balanced_synthetic_assignments(tokens, clusters, device)
        pack_index, flat_gather, _, max_count = build_static_pack_gather_indices(
            assignments,
            cluster_count=clusters,
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed)
        candidate_ids = torch.randint(
            0,
            d_ff,
            (clusters, candidate_m),
            generator=gen,
            dtype=torch.long,
        ).to(device)
        with torch.no_grad():
            w_up, w_gate = wug.detach().chunk(2, dim=0)
            wup_init, wgate_init, wdown_init = prepare_cluster_pool_weights(
                w_up,
                w_gate,
                w_down.detach(),
                candidate_ids,
            )
        wup_pool = wup_init.detach().clone().requires_grad_()
        wgate_pool = wgate_init.detach().clone().requires_grad_()
        wdown_pool = wdown_init.detach().clone().requires_grad_()

        def zero_dense() -> None:
            for tensor in (x_dense, wug, w_down):
                tensor.grad = None

        def zero_sparse() -> None:
            for tensor in (x_sparse, wup_pool, wgate_pool, wdown_pool):
                tensor.grad = None

        def dense_forward() -> torch.Tensor:
            up, gate = (x_dense @ wug.t()).chunk(2, dim=-1)
            return (up * F.silu(gate)) @ w_down

        def sparse_forward() -> torch.Tensor:
            return cluster_pool_ffn_forward_preindexed(
                x_sparse,
                pack_index,
                flat_gather,
                wup_pool,
                wgate_pool,
                wdown_pool,
            )

        def dense_step() -> torch.Tensor:
            zero_dense()
            out = dense_forward()
            loss = out.square().mean()
            loss.backward()
            return loss

        def sparse_step() -> torch.Tensor:
            zero_sparse()
            out = sparse_forward()
            loss = out.square().mean()
            loss.backward()
            return loss

        def scatter_grads() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if wup_pool.grad is None or wgate_pool.grad is None or wdown_pool.grad is None:
                raise RuntimeError("sparse gradients are not populated")
            return scatter_cluster_pool_grads(
                candidate_ids,
                wup_pool.grad,
                wgate_pool.grad,
                wdown_pool.grad,
                d_ff=d_ff,
            )

        for _ in range(args.warmup):
            dense_step()
            sparse_step()
            scatter_grads()
        _sync_device(device)

        def measure(fn) -> float:
            start = time.perf_counter()
            for _ in range(args.iters):
                fn()
            _sync_device(device)
            return (time.perf_counter() - start) / max(args.iters, 1)

        dense_fwd_seconds = measure(dense_forward)
        sparse_fwd_seconds = measure(sparse_forward)
        dense_step_seconds = measure(dense_step)
        sparse_step_seconds = measure(sparse_step)
        # Make sure scatter timing measures actual populated grads.
        sparse_step()
        grad_scatter_seconds = measure(scatter_grads)
        dense_graph_seconds = sparse_graph_seconds = None
        dense_graph_error = sparse_graph_error = None
        if args.cuda_graphs:
            dense_graph_seconds, dense_graph_error = _measure_cuda_graph_forward_backward(
                device,
                dense_forward,
                (x_dense, wug, w_down),
                warmup=args.cuda_graph_warmup,
                iters=args.iters,
            )
            sparse_graph_seconds, sparse_graph_error = _measure_cuda_graph_forward_backward(
                device,
                sparse_forward,
                (x_sparse, wup_pool, wgate_pool, wdown_pool),
                warmup=args.cuda_graph_warmup,
                iters=args.iters,
            )
        row = {
            "d_model": d_model,
            "d_ff": d_ff,
            "tokens": tokens,
            "clusters": clusters,
            "candidate_m": candidate_m,
            "max_tokens_per_cluster": max_count,
            "dense_forward_ms": dense_fwd_seconds * 1000.0,
            "sparse_forward_ms": sparse_fwd_seconds * 1000.0,
            "dense_forward_backward_ms": dense_step_seconds * 1000.0,
            "sparse_forward_backward_ms": sparse_step_seconds * 1000.0,
            "grad_scatter_ms": grad_scatter_seconds * 1000.0,
            "sparse_train_step_with_scatter_ms": (sparse_step_seconds + grad_scatter_seconds) * 1000.0,
            "forward_speedup": dense_fwd_seconds / max(sparse_fwd_seconds, 1e-12),
            "forward_backward_speedup": dense_step_seconds / max(sparse_step_seconds, 1e-12),
            "forward_backward_scatter_speedup": dense_step_seconds
            / max(sparse_step_seconds + grad_scatter_seconds, 1e-12),
            "ideal_ffn_flop_ratio": candidate_m / d_ff,
            "ideal_ffn_math_speedup": d_ff / max(candidate_m, 1),
            "dense_baseline": "fused_wug",
            "cuda_graphs_requested": bool(args.cuda_graphs),
        }
        if dense_graph_seconds is not None:
            row["dense_cuda_graph_forward_backward_ms"] = dense_graph_seconds * 1000.0
        if sparse_graph_seconds is not None:
            row["sparse_cuda_graph_forward_backward_ms"] = sparse_graph_seconds * 1000.0
        if dense_graph_seconds is not None and sparse_graph_seconds is not None:
            row["cuda_graph_forward_backward_speedup"] = dense_graph_seconds / max(sparse_graph_seconds, 1e-12)
        if dense_graph_error:
            row["dense_cuda_graph_error"] = dense_graph_error
        if sparse_graph_error:
            row["sparse_cuda_graph_error"] = sparse_graph_error
        rows.append(
            row
        )
        if device.type == "mps":
            torch.mps.empty_cache()
    report = {
        "mode": "benchmark_static_cluster_pool_ffn_train",
        "device": str(device),
        "dtype": str(dtype),
        "iters": args.iters,
        "warmup": args.warmup,
        "cuda_graphs": bool(args.cuda_graphs),
        "cuda_graph_warmup": args.cuda_graph_warmup,
        "grad_check": grad_check,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_benchmark_active_union_ffn_train(args: argparse.Namespace) -> None:
    device = default_device()
    rows: list[dict[str, Any]] = []
    sizes = args.size or [(512, 2048, 512), (2048, 8192, 128)]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    active_values = [int(value) for value in (args.active_m or [320])]
    use_triton_swiglu = bool(args.triton_swiglu_backward and triton_swiglu_available() and device.type == "cuda")
    for d_model, d_ff, tokens in sizes:
        for requested_active_m in active_values:
            scale = 1.0 / math.sqrt(d_model)
            active_m = min(int(requested_active_m), d_ff)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(args.seed + active_m)
            ids_cpu = torch.randperm(d_ff, generator=gen)[:active_m]
            ids = ids_cpu.to(device=device, dtype=torch.long)
            wug_ids = torch.cat([ids, ids + d_ff], dim=0)

            x_dense = torch.randn(tokens, d_model, device=device, dtype=dtype, requires_grad=True)
            x_indexed = x_dense.detach().clone().requires_grad_()
            x_packed_split = x_dense.detach().clone().requires_grad_()
            x_packed_fused = x_dense.detach().clone().requires_grad_()
            wug = (torch.randn(2 * d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
            w_down = (torch.randn(d_ff, d_model, device=device, dtype=dtype) * scale).requires_grad_()
            with torch.no_grad():
                up_full, gate_full = wug.detach().chunk(2, dim=0)
                wup_active_init = up_full.index_select(0, ids).detach().clone()
                wgate_active_init = gate_full.index_select(0, ids).detach().clone()
                wug_active_init = torch.cat([wup_active_init, wgate_active_init], dim=0).contiguous()
                wdown_active_init = w_down.index_select(0, ids).detach().clone()
            wup_active = wup_active_init.requires_grad_()
            wgate_active = wgate_active_init.requires_grad_()
            wug_active = wug_active_init.requires_grad_()
            wdown_active_split = wdown_active_init.detach().clone().requires_grad_()
            wdown_active_fused = wdown_active_init.detach().clone().requires_grad_()

            def zero_dense() -> None:
                for tensor in (x_dense, wug, w_down):
                    tensor.grad = None

            def zero_indexed() -> None:
                for tensor in (x_indexed, wug, w_down):
                    tensor.grad = None

            def zero_packed_split() -> None:
                for tensor in (x_packed_split, wup_active, wgate_active, wdown_active_split):
                    tensor.grad = None

            def zero_packed_fused() -> None:
                for tensor in (x_packed_fused, wug_active, wdown_active_fused):
                    tensor.grad = None

            def dense_forward() -> torch.Tensor:
                if use_triton_swiglu:
                    return triton_packed_swiglu_ffn(x_dense, wug, w_down)
                up, gate = (x_dense @ wug.t()).chunk(2, dim=-1)
                return (up * F.silu(gate)) @ w_down

            def indexed_fused_forward() -> torch.Tensor:
                wug_sel = wug.index_select(0, wug_ids)
                down_w = w_down.index_select(0, ids)
                if use_triton_swiglu:
                    return triton_packed_swiglu_ffn(x_indexed, wug_sel, down_w)
                up, gate = (x_indexed @ wug_sel.t()).chunk(2, dim=-1)
                return (up * F.silu(gate)) @ down_w

            def packed_split_forward() -> torch.Tensor:
                return (
                    (x_packed_split @ wup_active.t())
                    * F.silu(x_packed_split @ wgate_active.t())
                ) @ wdown_active_split

            def packed_fused_forward() -> torch.Tensor:
                if use_triton_swiglu:
                    return triton_packed_swiglu_ffn(x_packed_fused, wug_active, wdown_active_fused)
                up, gate = (x_packed_fused @ wug_active.t()).chunk(2, dim=-1)
                return (up * F.silu(gate)) @ wdown_active_fused

            def dense_step() -> torch.Tensor:
                zero_dense()
                out = dense_forward()
                loss = out.square().mean()
                loss.backward()
                return loss

            def indexed_fused_step() -> torch.Tensor:
                zero_indexed()
                out = indexed_fused_forward()
                loss = out.square().mean()
                loss.backward()
                return loss

            def packed_split_step() -> torch.Tensor:
                zero_packed_split()
                out = packed_split_forward()
                loss = out.square().mean()
                loss.backward()
                return loss

            def packed_fused_step() -> torch.Tensor:
                zero_packed_fused()
                out = packed_fused_forward()
                loss = out.square().mean()
                loss.backward()
                return loss

            for _ in range(args.warmup):
                dense_step()
                indexed_fused_step()
                packed_split_step()
                packed_fused_step()
            _sync_device(device)

            def measure(fn) -> float:
                start = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                _sync_device(device)
                return (time.perf_counter() - start) / max(args.iters, 1)

            dense_fwd_seconds = measure(dense_forward)
            indexed_fwd_seconds = measure(indexed_fused_forward)
            packed_split_fwd_seconds = measure(packed_split_forward)
            packed_fused_fwd_seconds = measure(packed_fused_forward)
            dense_step_seconds = measure(dense_step)
            indexed_step_seconds = measure(indexed_fused_step)
            packed_split_step_seconds = measure(packed_split_step)
            packed_fused_step_seconds = measure(packed_fused_step)
            dense_graph_seconds = indexed_graph_seconds = packed_split_graph_seconds = packed_fused_graph_seconds = None
            dense_graph_error = indexed_graph_error = packed_split_graph_error = packed_fused_graph_error = None
            if args.cuda_graphs:
                dense_graph_seconds, dense_graph_error = _measure_cuda_graph_forward_backward(
                    device,
                    dense_forward,
                    (x_dense, wug, w_down),
                    warmup=args.cuda_graph_warmup,
                    iters=args.iters,
                )
                indexed_graph_seconds, indexed_graph_error = _measure_cuda_graph_forward_backward(
                    device,
                    indexed_fused_forward,
                    (x_indexed, wug, w_down),
                    warmup=args.cuda_graph_warmup,
                    iters=args.iters,
                )
                packed_split_graph_seconds, packed_split_graph_error = _measure_cuda_graph_forward_backward(
                    device,
                    packed_split_forward,
                    (x_packed_split, wup_active, wgate_active, wdown_active_split),
                    warmup=args.cuda_graph_warmup,
                    iters=args.iters,
                )
                packed_fused_graph_seconds, packed_fused_graph_error = _measure_cuda_graph_forward_backward(
                    device,
                    packed_fused_forward,
                    (x_packed_fused, wug_active, wdown_active_fused),
                    warmup=args.cuda_graph_warmup,
                    iters=args.iters,
                )
            with torch.no_grad():
                idx_out = indexed_fused_forward()
                split_out = packed_split_forward()
                fused_out = packed_fused_forward()
                max_indexed_packed_abs_diff = float((idx_out - fused_out).abs().max().detach().cpu())
                max_split_fused_abs_diff = float((split_out - fused_out).abs().max().detach().cpu())
            row = {
                "d_model": d_model,
                "d_ff": d_ff,
                "tokens": tokens,
                "active_m": active_m,
                "active_fraction": active_m / max(d_ff, 1),
                "dense_baseline": "fused_wug",
                "dense_forward_ms": dense_fwd_seconds * 1000.0,
                "active_indexed_fused_forward_ms": indexed_fwd_seconds * 1000.0,
                "active_packed_split_forward_ms": packed_split_fwd_seconds * 1000.0,
                "active_packed_fused_forward_ms": packed_fused_fwd_seconds * 1000.0,
                "dense_forward_backward_ms": dense_step_seconds * 1000.0,
                "active_indexed_fused_forward_backward_ms": indexed_step_seconds * 1000.0,
                "active_packed_split_forward_backward_ms": packed_split_step_seconds * 1000.0,
                "active_packed_fused_forward_backward_ms": packed_fused_step_seconds * 1000.0,
                "active_indexed_fused_forward_speedup": dense_fwd_seconds
                / max(indexed_fwd_seconds, 1e-12),
                "active_packed_split_forward_speedup": dense_fwd_seconds
                / max(packed_split_fwd_seconds, 1e-12),
                "active_packed_fused_forward_speedup": dense_fwd_seconds
                / max(packed_fused_fwd_seconds, 1e-12),
                "active_indexed_fused_forward_backward_speedup": dense_step_seconds
                / max(indexed_step_seconds, 1e-12),
                "active_packed_split_forward_backward_speedup": dense_step_seconds
                / max(packed_split_step_seconds, 1e-12),
                "active_packed_fused_forward_backward_speedup": dense_step_seconds
                / max(packed_fused_step_seconds, 1e-12),
                "max_indexed_packed_abs_diff": max_indexed_packed_abs_diff,
                "max_split_fused_abs_diff": max_split_fused_abs_diff,
                "ideal_ffn_flop_ratio": active_m / max(d_ff, 1),
                "ideal_ffn_math_speedup": d_ff / max(active_m, 1),
                "cuda_graphs_requested": bool(args.cuda_graphs),
                "triton_swiglu_backward_requested": bool(args.triton_swiglu_backward),
                "triton_swiglu_backward_used": use_triton_swiglu,
                "swiglu_impl": "triton_custom_autograd" if use_triton_swiglu else "torch_autograd",
            }
            graph_rows = (
                ("dense_cuda_graph_forward_backward_ms", dense_graph_seconds, dense_graph_error),
                (
                    "active_indexed_fused_cuda_graph_forward_backward_ms",
                    indexed_graph_seconds,
                    indexed_graph_error,
                ),
                (
                    "active_packed_split_cuda_graph_forward_backward_ms",
                    packed_split_graph_seconds,
                    packed_split_graph_error,
                ),
                (
                    "active_packed_fused_cuda_graph_forward_backward_ms",
                    packed_fused_graph_seconds,
                    packed_fused_graph_error,
                ),
            )
            for key, seconds, error in graph_rows:
                if seconds is not None:
                    row[key] = seconds * 1000.0
                if error:
                    row[key.replace("_ms", "_error")] = error
            if dense_graph_seconds is not None and indexed_graph_seconds is not None:
                row["active_indexed_fused_cuda_graph_forward_backward_speedup"] = dense_graph_seconds / max(
                    indexed_graph_seconds,
                    1e-12,
                )
            if dense_graph_seconds is not None and packed_split_graph_seconds is not None:
                row["active_packed_split_cuda_graph_forward_backward_speedup"] = dense_graph_seconds / max(
                    packed_split_graph_seconds,
                    1e-12,
                )
            if dense_graph_seconds is not None and packed_fused_graph_seconds is not None:
                row["active_packed_fused_cuda_graph_forward_backward_speedup"] = dense_graph_seconds / max(
                    packed_fused_graph_seconds,
                    1e-12,
                )
            rows.append(row)
            if device.type == "mps":
                torch.mps.empty_cache()
    report = {
        "mode": "benchmark_active_union_ffn_train",
        "device": str(device),
        "dtype": str(dtype),
        "iters": args.iters,
        "warmup": args.warmup,
        "cuda_graphs": bool(args.cuda_graphs),
        "cuda_graph_warmup": args.cuda_graph_warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _benchmark_dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = value.lower()
    if normalized == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if normalized == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"unsupported benchmark dtype: {value}")


def _time_model_forward_components(
    model: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    """Forward-only component timing for the full model.

    This is intentionally applied to dense and sparse models through the same
    hand-unrolled Transformer path. It is a diagnostic split, while the real
    train-step timing below uses the model's normal forward/backward path.
    """

    was_training = model.training
    model.eval()

    def run_once() -> dict[str, float]:
        times = {
            "embed_forward_seconds": 0.0,
            "attention_forward_seconds": 0.0,
            "ffn_forward_seconds": 0.0,
            "output_forward_seconds": 0.0,
        }

        _sync_device(device)
        start = time.perf_counter()
        x = model.embed(tokens)
        _sync_device(device)
        times["embed_forward_seconds"] += time.perf_counter() - start

        for block in model.blocks:
            _sync_device(device)
            start = time.perf_counter()
            u = x + block.attn(block.norm1(x))
            _sync_device(device)
            times["attention_forward_seconds"] += time.perf_counter() - start

            _sync_device(device)
            start = time.perf_counter()
            x = u + block.mlp(block.norm2(u))
            _sync_device(device)
            times["ffn_forward_seconds"] += time.perf_counter() - start

        _sync_device(device)
        start = time.perf_counter()
        hidden = model.final_norm(x)
        logits = hidden @ model.vocab_weight.t()
        loss = lm_loss_per_sample(logits, targets).mean() / float(targets.shape[1])
        _sync_device(device)
        times["output_forward_seconds"] += time.perf_counter() - start
        # Keep the scalar live so eager execution cannot discard the loss path.
        times["loss_value"] = float(loss.detach().float().cpu())
        return times

    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            run_once()
        totals: dict[str, float] = {}
        for _ in range(max(1, int(iters))):
            row = run_once()
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + float(value)
    model.train(was_training)
    denom = float(max(1, int(iters)))
    out: dict[str, float] = {}
    for key, value in totals.items():
        if key.endswith("_seconds"):
            out[key.replace("_seconds", "_ms")] = (value / denom) * 1000.0
        else:
            out[key] = value / denom
    return out


def _time_model_train_step(
    model: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    def step_once() -> tuple[float, float, float, float]:
        _sync_device(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(tokens, targets, return_loss_per_sample=True)
        assert out.loss_per_sample is not None
        loss = out.loss_per_sample.mean() / float(targets.shape[1])
        loss.backward()
        _sync_device(device)
        fwd_bwd_end = time.perf_counter()
        optimizer.step()
        _sync_device(device)
        end = time.perf_counter()
        return (
            fwd_bwd_end - start,
            end - fwd_bwd_end,
            end - start,
            float(loss.detach().float().cpu()),
        )

    for _ in range(max(0, int(warmup))):
        step_once()
    fwd_bwd = 0.0
    optimizer_time = 0.0
    total = 0.0
    loss_total = 0.0
    for _ in range(max(1, int(iters))):
        fwd_bwd_i, optimizer_i, total_i, loss_i = step_once()
        fwd_bwd += fwd_bwd_i
        optimizer_time += optimizer_i
        total += total_i
        loss_total += loss_i
    denom = float(max(1, int(iters)))
    return {
        "forward_backward_ms": (fwd_bwd / denom) * 1000.0,
        "optimizer_ms": (optimizer_time / denom) * 1000.0,
        "total_step_ms": (total / denom) * 1000.0,
        "loss": loss_total / denom,
    }


def _event_profile_model_train_step(
    model: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    """CUDA-event component profile for one full model train step.

    Forward ranges are timed directly. Backward ranges are timed with tensor
    hooks on each segment boundary: the output-gradient hook starts the range,
    and the input-gradient hook ends it. This gives a much cleaner answer than
    sync-heavy forward-only component timers.
    """

    if device.type != "cuda":
        return {"available": False, "reason": f"CUDA events require cuda, got {device}"}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    def event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def run_once(measure: bool) -> tuple[dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]], float]:
        pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "embedding_forward": [],
            "attention_forward": [],
            "ffn_forward": [],
            "output_head_forward": [],
            "attention_backward": [],
            "ffn_backward": [],
            "output_head_backward": [],
            "optimizer": [],
            "total_step": [],
        }

        def timed_forward(name: str, fn):
            if measure:
                start, end = event_pair()
                torch.cuda.nvtx.range_push(name)
                start.record()
                out = fn()
                end.record()
                torch.cuda.nvtx.range_pop()
                pairs[name].append((start, end))
                return out
            return fn()

        def add_backward_segment(name: str, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
            if not measure:
                return
            start, end = event_pair()

            def start_hook(grad):
                torch.cuda.nvtx.range_push(name)
                start.record()
                return grad

            def end_hook(grad):
                end.record()
                torch.cuda.nvtx.range_pop()
                return grad

            output_tensor.register_hook(start_hook)
            input_tensor.register_hook(end_hook)
            pairs[name].append((start, end))

        total_start = total_end = None
        if measure:
            total_start, total_end = event_pair()
            total_start.record()
        optimizer.zero_grad(set_to_none=True)
        x = timed_forward("embedding_forward", lambda: model.embed(tokens))
        for block in model.blocks:
            x_in = x
            u = timed_forward("attention_forward", lambda block=block, x_in=x_in: x_in + block.attn(block.norm1(x_in)))
            add_backward_segment("attention_backward", x_in, u)
            x_out = timed_forward("ffn_forward", lambda block=block, u=u: u + block.mlp(block.norm2(u)))
            add_backward_segment("ffn_backward", u, x_out)
            x = x_out
        if measure:
            oh_fwd_start, oh_fwd_end = event_pair()
            torch.cuda.nvtx.range_push("output_head_forward")
            oh_fwd_start.record()
        hidden = model.final_norm(x)
        logits = hidden @ model.vocab_weight.t()
        loss_per_sample = lm_loss_per_sample(logits, targets)
        loss = loss_per_sample.mean() / float(targets.shape[1])
        if measure:
            oh_fwd_end.record()
            torch.cuda.nvtx.range_pop()
            pairs["output_head_forward"].append((oh_fwd_start, oh_fwd_end))
        out_start = out_end = None
        if measure:
            out_start, out_end = event_pair()

            def output_head_end_hook(grad):
                assert out_end is not None
                out_end.record()
                torch.cuda.nvtx.range_pop()
                return grad

            hidden.register_hook(output_head_end_hook)
        if measure:
            assert out_start is not None and out_end is not None
            torch.cuda.nvtx.range_push("output_head_backward")
            out_start.record()
            pairs["output_head_backward"].append((out_start, out_end))
        loss.backward()
        if measure:
            opt_start, opt_end = event_pair()
            torch.cuda.nvtx.range_push("optimizer")
            opt_start.record()
            optimizer.step()
            opt_end.record()
            torch.cuda.nvtx.range_pop()
            pairs["optimizer"].append((opt_start, opt_end))
            assert total_start is not None and total_end is not None
            total_end.record()
            pairs["total_step"].append((total_start, total_end))
        else:
            optimizer.step()
        return pairs, float(loss.detach().float().cpu())

    for _ in range(max(0, int(warmup))):
        run_once(False)
    totals: dict[str, float] = {}
    losses: list[float] = []
    for _ in range(max(1, int(iters))):
        pairs, loss_value = run_once(True)
        losses.append(loss_value)
        torch.cuda.synchronize()
        for key, ranges in pairs.items():
            totals[key] = totals.get(key, 0.0) + sum(start.elapsed_time(end) for start, end in ranges)
    denom = float(max(1, int(iters)))
    avg = {f"{key}_ms": value / denom for key, value in totals.items()}
    known = (
        avg.get("embedding_forward_ms", 0.0)
        + avg.get("attention_forward_ms", 0.0)
        + avg.get("attention_backward_ms", 0.0)
        + avg.get("ffn_forward_ms", 0.0)
        + avg.get("ffn_backward_ms", 0.0)
        + avg.get("output_head_forward_ms", 0.0)
        + avg.get("output_head_backward_ms", 0.0)
        + avg.get("optimizer_ms", 0.0)
    )
    total = avg.get("total_step_ms", 0.0)
    avg["attention_forward_backward_ms"] = avg.get("attention_forward_ms", 0.0) + avg.get(
        "attention_backward_ms",
        0.0,
    )
    avg["ffn_forward_backward_ms"] = avg.get("ffn_forward_ms", 0.0) + avg.get("ffn_backward_ms", 0.0)
    avg["output_head_forward_backward_ms"] = avg.get("output_head_forward_ms", 0.0) + avg.get(
        "output_head_backward_ms",
        0.0,
    )
    avg["misc_overhead_ms"] = total - known
    avg["loss"] = sum(losses) / max(len(losses), 1)
    avg["available"] = True
    return avg


def cmd_benchmark_active_union_model_train_step(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    model = config.model
    if args.d_model is not None:
        model = dataclasses.replace(model, d_model=args.d_model)
    if args.d_ff is not None:
        model = dataclasses.replace(model, d_ff=args.d_ff)
    if args.layers is not None:
        model = dataclasses.replace(model, n_dense_layers=args.layers)
    if args.heads is not None:
        model = dataclasses.replace(model, n_heads=args.heads)
    if args.vocab_size is not None:
        model = dataclasses.replace(model, vocab_size=args.vocab_size)
    config = dataclasses.replace(config, model=dataclasses.replace(model, topology="dense"))
    training = config.training
    if args.batch_size is not None:
        training = dataclasses.replace(training, batch_size=args.batch_size)
    if args.seq_len is not None:
        training = dataclasses.replace(training, seq_len=args.seq_len)
    config = dataclasses.replace(config, training=training)
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    dtype = _benchmark_dtype(args.dtype, device)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, targets = next(streams.train_batches(config.training))
    tokens = tokens.to(device)
    targets = targets.to(device)
    tokens_per_batch = int(tokens.numel())
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    caps = [int(value) for value in (args.active_union_cap or [0, 320])]
    rows: list[dict[str, Any]] = []

    if not args.random_init and not args.dense_checkpoint:
        raise SystemExit("--dense-checkpoint is required unless --random-init is set")
    random_state = None
    if args.random_init:
        random_state = copy.deepcopy(
            DenseModel(dataclasses.replace(config.model, topology="dense")).state_dict()
        )

    def make_dense_model() -> DenseModel:
        if args.random_init:
            model = DenseModel(dataclasses.replace(config.model, topology="dense")).to(device)
            assert random_state is not None
            model.load_state_dict(random_state, strict=True)
            return model
        assert args.dense_checkpoint is not None
        return _load_dense_model(config, args.dense_checkpoint, device)

    dense = make_dense_model()
    if args.pack_dense_ffn:
        _pack_dense_ffns_for_benchmark(dense)
    dense.to(dtype=dtype)
    _set_triton_swiglu_backward(dense, args.triton_swiglu_backward)
    dense_components = _time_model_forward_components(
        dense,
        tokens,
        targets,
        device=device,
        iters=args.component_iters,
        warmup=args.component_warmup,
    )
    dense_train = _time_model_train_step(
        dense,
        tokens,
        targets,
        lr=args.lr if args.lr is not None else config.training.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else config.training.weight_decay,
        device=device,
        iters=args.iters,
        warmup=args.warmup,
    )
    dense_event_profile = _event_profile_model_train_step(
        dense,
        tokens,
        targets,
        lr=args.lr if args.lr is not None else config.training.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else config.training.weight_decay,
        device=device,
        iters=args.event_iters,
        warmup=args.event_warmup,
    ) if args.event_profile else {}
    dense_param_count_value = sum(param.numel() for param in dense.parameters())
    del dense
    if device.type == "mps":
        torch.mps.empty_cache()

    for cap in caps:
        sparse = make_dense_model()
        refresh_aux = _refresh_active_union_ffns(
            sparse,
            streams,
            config,
            calibration_batches=calibration_batches,
            device=device,
            rank=args.rank,
            cluster_count=args.clusters,
            candidate_m=args.candidate_m,
            score_mode=args.score_mode,
            aggregation=args.aggregation,
            cluster_iters=args.cluster_iters,
            cap=None if cap <= 0 else cap,
            packed=True,
        )
        sparse.to(dtype=dtype)
        _set_triton_swiglu_backward(sparse, args.triton_swiglu_backward)
        sparse_components = _time_model_forward_components(
            sparse,
            tokens,
            targets,
            device=device,
            iters=args.component_iters,
            warmup=args.component_warmup,
        )
        sparse_train = _time_model_train_step(
            sparse,
            tokens,
            targets,
            lr=args.lr if args.lr is not None else config.training.lr,
            weight_decay=args.weight_decay if args.weight_decay is not None else config.training.weight_decay,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
        )
        sparse_event_profile = _event_profile_model_train_step(
            sparse,
            tokens,
            targets,
            lr=args.lr if args.lr is not None else config.training.lr,
            weight_decay=args.weight_decay if args.weight_decay is not None else config.training.weight_decay,
            device=device,
            iters=args.event_iters,
            warmup=args.event_warmup,
        ) if args.event_profile else {}
        coverage = _active_union_coverage_metrics(sparse)
        sparse_param_count_value = sum(param.numel() for param in sparse.parameters())
        row = {
            "active_union_cap": None if cap <= 0 else cap,
            "dtype": str(dtype),
            "batch_size": config.training.batch_size,
            "seq_len": config.training.seq_len,
            "tokens": tokens_per_batch,
            "rank": args.rank,
            "clusters": args.clusters,
            "candidate_m": args.candidate_m,
            "calibration_tokens": calibration_batches * tokens_per_batch,
            "dense_packed_ffn_baseline": bool(args.pack_dense_ffn),
            "triton_swiglu_backward_requested": bool(args.triton_swiglu_backward),
            "triton_swiglu_backward_used": bool(args.triton_swiglu_backward and triton_swiglu_available()),
            "dense_param_count": dense_param_count_value,
            "sparse_param_count": sparse_param_count_value,
            "sparse_param_fraction": sparse_param_count_value / max(dense_param_count_value, 1),
            "dense_forward_backward_ms": dense_train["forward_backward_ms"],
            "dense_optimizer_ms": dense_train["optimizer_ms"],
            "dense_total_step_ms": dense_train["total_step_ms"],
            "dense_loss": dense_train["loss"],
            "sparse_forward_backward_ms": sparse_train["forward_backward_ms"],
            "sparse_optimizer_ms": sparse_train["optimizer_ms"],
            "sparse_total_step_ms": sparse_train["total_step_ms"],
            "sparse_loss": sparse_train["loss"],
            "total_step_speedup": dense_train["total_step_ms"] / max(sparse_train["total_step_ms"], 1e-12),
            "forward_backward_speedup": dense_train["forward_backward_ms"]
            / max(sparse_train["forward_backward_ms"], 1e-12),
            "optimizer_speedup": dense_train["optimizer_ms"] / max(sparse_train["optimizer_ms"], 1e-12),
            "dense_components": dense_components,
            "sparse_components": sparse_components,
            "attention_forward_speedup": dense_components.get("attention_forward_ms", 0.0)
            / max(sparse_components.get("attention_forward_ms", 0.0), 1e-12),
            "ffn_forward_speedup": dense_components.get("ffn_forward_ms", 0.0)
            / max(sparse_components.get("ffn_forward_ms", 0.0), 1e-12),
            "output_forward_speedup": dense_components.get("output_forward_ms", 0.0)
            / max(sparse_components.get("output_forward_ms", 0.0), 1e-12),
            "coverage": coverage,
            "calibration_avg_cluster_imbalance": refresh_aux["calibration_avg_cluster_imbalance"],
            "calibration_avg_nonempty_clusters": refresh_aux["calibration_avg_nonempty_clusters"],
            "dense_event_profile": dense_event_profile,
            "sparse_event_profile": sparse_event_profile,
        }
        if dense_event_profile.get("available") and sparse_event_profile.get("available"):
            row["event_total_step_speedup"] = dense_event_profile["total_step_ms"] / max(
                sparse_event_profile["total_step_ms"],
                1e-12,
            )
            row["event_attention_fwd_bwd_speedup"] = dense_event_profile[
                "attention_forward_backward_ms"
            ] / max(sparse_event_profile["attention_forward_backward_ms"], 1e-12)
            row["event_ffn_fwd_bwd_speedup"] = dense_event_profile["ffn_forward_backward_ms"] / max(
                sparse_event_profile["ffn_forward_backward_ms"],
                1e-12,
            )
            row["event_output_head_fwd_bwd_speedup"] = dense_event_profile[
                "output_head_forward_backward_ms"
            ] / max(sparse_event_profile["output_head_forward_backward_ms"], 1e-12)
        rows.append(row)
        del sparse
        if device.type == "mps":
            torch.mps.empty_cache()

    report = {
        "mode": "benchmark_active_union_model_train_step",
        "checkpoint": args.dense_checkpoint,
        "random_init": bool(args.random_init),
        "device": str(device),
        "iters": args.iters,
        "warmup": args.warmup,
        "component_iters": args.component_iters,
        "component_warmup": args.component_warmup,
        "event_profile": bool(args.event_profile),
        "event_iters": args.event_iters,
        "event_warmup": args.event_warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def _time_cuda_or_wall(
    device: torch.device,
    fn,
    *,
    iters: int,
    warmup: int,
) -> float:
    for _ in range(max(0, int(warmup))):
        fn()
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(max(1, int(iters))):
            fn()
        end.record()
        torch.cuda.synchronize()
        return (start.elapsed_time(end) / max(1, int(iters)))
    _sync_device(device)
    start_time = time.perf_counter()
    for _ in range(max(1, int(iters))):
        fn()
    _sync_device(device)
    return ((time.perf_counter() - start_time) / max(1, int(iters))) * 1000.0


def _zero_module_grads(*modules: torch.nn.Module) -> None:
    for module in modules:
        for param in module.parameters():
            param.grad = None


def _benchmark_block_parts(
    dense_block: torch.nn.Module,
    sparse_block: torch.nn.Module,
    x_base: torch.Tensor,
    *,
    device: torch.device,
    iters: int,
    warmup: int,
) -> dict[str, float]:
    with torch.no_grad():
        u_base = x_base + dense_block.attn(dense_block.norm1(x_base))
    u_base = u_base.detach()

    def dense_block_step() -> None:
        _zero_module_grads(dense_block)
        x = x_base.detach().clone().requires_grad_(True)
        y = dense_block(x)
        y.float().square().mean().backward()

    def sparse_block_step() -> None:
        _zero_module_grads(sparse_block)
        x = x_base.detach().clone().requires_grad_(True)
        y = sparse_block(x)
        y.float().square().mean().backward()

    def dense_attention_step() -> None:
        _zero_module_grads(dense_block.norm1, dense_block.attn)
        x = x_base.detach().clone().requires_grad_(True)
        y = x + dense_block.attn(dense_block.norm1(x))
        y.float().square().mean().backward()

    def sparse_attention_step() -> None:
        _zero_module_grads(sparse_block.norm1, sparse_block.attn)
        x = x_base.detach().clone().requires_grad_(True)
        y = x + sparse_block.attn(sparse_block.norm1(x))
        y.float().square().mean().backward()

    def dense_ffn_step() -> None:
        _zero_module_grads(dense_block.norm2, dense_block.mlp)
        u = u_base.detach().clone().requires_grad_(True)
        y = u + dense_block.mlp(dense_block.norm2(u))
        y.float().square().mean().backward()

    def sparse_ffn_step() -> None:
        _zero_module_grads(sparse_block.norm2, sparse_block.mlp)
        u = u_base.detach().clone().requires_grad_(True)
        y = u + sparse_block.mlp(sparse_block.norm2(u))
        y.float().square().mean().backward()

    dense_block_ms = _time_cuda_or_wall(device, dense_block_step, iters=iters, warmup=warmup)
    sparse_block_ms = _time_cuda_or_wall(device, sparse_block_step, iters=iters, warmup=warmup)
    dense_attn_ms = _time_cuda_or_wall(device, dense_attention_step, iters=iters, warmup=warmup)
    sparse_attn_ms = _time_cuda_or_wall(device, sparse_attention_step, iters=iters, warmup=warmup)
    dense_ffn_ms = _time_cuda_or_wall(device, dense_ffn_step, iters=iters, warmup=warmup)
    sparse_ffn_ms = _time_cuda_or_wall(device, sparse_ffn_step, iters=iters, warmup=warmup)
    return {
        "dense_block_forward_backward_ms": dense_block_ms,
        "sparse_block_forward_backward_ms": sparse_block_ms,
        "block_forward_backward_speedup": dense_block_ms / max(sparse_block_ms, 1e-12),
        "dense_attention_forward_backward_ms": dense_attn_ms,
        "sparse_attention_forward_backward_ms": sparse_attn_ms,
        "attention_forward_backward_speedup": dense_attn_ms / max(sparse_attn_ms, 1e-12),
        "dense_ffn_forward_backward_ms": dense_ffn_ms,
        "sparse_ffn_forward_backward_ms": sparse_ffn_ms,
        "ffn_forward_backward_speedup": dense_ffn_ms / max(sparse_ffn_ms, 1e-12),
    }


def cmd_benchmark_active_union_block_train_step(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "dense_exact")
    training = config.training
    if args.batch_size is not None:
        training = dataclasses.replace(training, batch_size=args.batch_size)
    if args.seq_len is not None:
        training = dataclasses.replace(training, seq_len=args.seq_len)
    config = dataclasses.replace(config, training=training)
    set_seed(args.seed if args.seed is not None else config.training.seed)
    device = default_device()
    dtype = _benchmark_dtype(args.dtype, device)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, _ = next(streams.train_batches(config.training))
    tokens = tokens.to(device)
    tokens_per_batch = int(tokens.numel())
    calibration_batches = max(1, math.ceil(args.calibration_tokens / tokens_per_batch))
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    sparse = _load_dense_model(config, args.dense_checkpoint, device)
    _pack_dense_ffns_for_benchmark(dense)
    _refresh_active_union_ffns(
        sparse,
        streams,
        config,
        calibration_batches=calibration_batches,
        device=device,
        rank=args.rank,
        cluster_count=args.clusters,
        candidate_m=args.candidate_m,
        score_mode=args.score_mode,
        aggregation=args.aggregation,
        cluster_iters=args.cluster_iters,
        cap=None if args.active_union_cap <= 0 else args.active_union_cap,
        packed=True,
    )
    dense.to(dtype=dtype)
    sparse.to(dtype=dtype)
    _set_triton_swiglu_backward(dense, args.triton_swiglu_backward)
    _set_triton_swiglu_backward(sparse, args.triton_swiglu_backward)
    layer_indices = args.layers or [0]
    rows: list[dict[str, Any]] = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= config.model.n_dense_layers:
            raise SystemExit(f"layer index {layer_idx} out of range for {config.model.n_dense_layers} layers")
        with torch.no_grad():
            x = dense.embed(tokens)
            for idx in range(layer_idx):
                x = dense.blocks[idx](x)
            x_base = x.detach()
        row = _benchmark_block_parts(
            dense.blocks[layer_idx],
            sparse.blocks[layer_idx],
            x_base,
            device=device,
            iters=args.iters,
            warmup=args.warmup,
        )
        row.update(
            {
                "layer": layer_idx,
                "active_union_cap": None if args.active_union_cap <= 0 else args.active_union_cap,
                "dtype": str(dtype),
                "tokens": tokens_per_batch,
                "triton_swiglu_backward_used": bool(args.triton_swiglu_backward and triton_swiglu_available()),
            }
        )
        rows.append(row)
    report = {
        "mode": "benchmark_active_union_block_train_step",
        "checkpoint": args.dense_checkpoint,
        "device": str(device),
        "iters": args.iters,
        "warmup": args.warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_benchmark_mlx_svd_sparse_ffn(args: argparse.Namespace) -> None:
    from recursive_training_engine.mlx_svd_ffn import benchmark_mlx_svd_sparse_ffn

    sizes = args.size or [(64, 256, 2048), (512, 2048, 128)]
    rows = [
        dataclasses.asdict(row)
        for row in benchmark_mlx_svd_sparse_ffn(
            sizes=sizes,
            rank=args.rank,
            factor_m=args.factor_m,
            product_factor_m=args.product_factor_m,
            k=args.k,
            iters=args.iters,
            warmup=args.warmup,
            seed=args.seed,
            backend=args.backend,
        )
    ]
    report = {
        "mode": "benchmark_mlx_svd_sparse_ffn",
        "backend": "mlx",
        "kernel_backend": args.backend,
        "iters": args.iters,
        "warmup": args.warmup,
        "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def cmd_benchmark_kernels(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    torch.manual_seed(config.training.seed)
    b, s, d = config.training.batch_size, config.training.seq_len, config.model.d_model
    x = torch.randn(b, s, d, device=device)
    w = torch.ones(d, device=device)
    q = torch.randn(b, config.model.n_heads, s, d // config.model.n_heads, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rows = []
    for name, fn, args_tuple in [
        ("k_fused_rmsnorm", optimized.k_fused_rmsnorm, (x, w)),
        ("k_flash_causal_dense", optimized.k_flash_causal_dense, (q, k, v)),
    ]:
        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start is not None and end is not None:
            start.record()
            for _ in range(args.iters):
                fn(*args_tuple)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / args.iters
        else:
            import time

            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn(*args_tuple)
            ms = (time.perf_counter() - t0) * 1000.0 / args.iters
        rows.append({"kernel": name, "device": str(device), "ms": ms})
    rows.append({"kernel": "triton_available", "value": optimized.triton_available()})
    _print_json(rows)


def cmd_train_macro_teacher(args: argparse.Namespace) -> None:
    config = _with_run_dir(load_config(args.config), args.run_dir)
    fixed_depth = args.fixed_depth if args.fixed_depth is not None else config.training.fixed_depth
    fixed_recipe = args.fixed_recipe if args.fixed_recipe is not None else config.training.fixed_recipe
    if fixed_depth is None or fixed_recipe is None:
        raise SystemExit("train-macro-teacher requires fixed_recipe and fixed_depth")
    if args.teacher_checkpoint is None and not args.allow_random_teacher:
        raise SystemExit("train-macro-teacher requires --teacher-checkpoint")
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, topology="recursive"),
        training=dataclasses.replace(
            config.training,
            mode="recursive_macro_distill_only",
            fixed_depth=fixed_depth,
            fixed_recipe=fixed_recipe,
            audit_p_min=0.0,
            audit_p_max=0.0,
        ),
    )
    set_seed(config.training.seed)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    device = default_device()
    model = RecursiveModel(config.model, config.output).to(device)
    teacher_loaded = False
    if args.teacher_checkpoint is not None:
        payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        if payload.get("model") is None:
            raise SystemExit(f"teacher checkpoint has no model state: {args.teacher_checkpoint}")
        teacher_config = payload.get("config")
        teacher_schedule = getattr(getattr(teacher_config, "training", None), "fixed_recipe_schedule", None)
        if args.teacher_mode == "deferred_grouped" and teacher_schedule:
            config = dataclasses.replace(
                config,
                training=dataclasses.replace(
                    config.training,
                    fixed_recipe_schedule=[int(recipe_id) for recipe_id in teacher_schedule],
                ),
            )
        load_compatible_state_dict(model, payload["model"], skip_prefixes=("macro.",))
        teacher_loaded = True
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.macro.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.macro.parameters(),
        lr=args.lr if args.lr is not None else config.training.effective_lr_macro,
        weight_decay=config.training.weight_decay,
    )
    batches = streams.train_batches(config.training)
    boundary_batches = None
    if args.boundary_cache:
        boundary_payload = torch.load(args.boundary_cache, map_location="cpu", weights_only=False)
        boundary_batches = boundary_payload.get("batches") or []
        if not boundary_batches:
            raise SystemExit(f"boundary cache is empty: {args.boundary_cache}")
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from recursive_training_engine.config import save_config
    from recursive_training_engine.reporting import JsonlLogger

    save_config(config, run_dir / "resolved_config.yaml")
    write_json(
        run_dir / "manifest.json",
        build_manifest(
            config,
            extra={
                "data_fingerprint": streams.data_fingerprint,
                "projection_lane": config.data.vocab_projection,
                "tokenizer": streams.tokenizer_name,
                "train_tokens": int(streams.train.numel()),
                "eval_tokens": int(streams.eval.numel()),
                "teacher_checkpoint_source": args.teacher_checkpoint,
                "teacher_mode": args.teacher_mode,
                "macro_checkpoint_source": None,
                "teacher_loaded": teacher_loaded,
                "boundary_cache": args.boundary_cache,
            },
        ),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    rows = []
    try:
        for step in range(1, args.steps + 1):
            if boundary_batches is not None:
                entry = boundary_batches[(step - 1) % len(boundary_batches)]
                tokens = entry["tokens"].to(device)
                targets = entry["targets"].to(device)
                cached_endpoint = entry.get("recurrent_hidden")
                if cached_endpoint is None:
                    cached_endpoint = entry["states"][fixed_depth]
                exact_endpoint = cached_endpoint.to(device)
                with torch.no_grad():
                    _, exact_logits = model._coda_logits(exact_endpoint)
                    exact_loss_per_sample = lm_loss_per_sample(exact_logits, targets)
                    h0_for_loss = entry.get("h0")
                    h0_for_loss = h0_for_loss.to(device) if h0_for_loss is not None else None
            else:
                tokens, targets = next(batches)
                tokens = tokens.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    if args.teacher_mode == "deferred_grouped":
                        exact = model.forward_deferred_grouped_exact(
                            tokens,
                            targets,
                            return_loss_per_sample=True,
                            fixed_recipe=fixed_recipe,
                            fixed_recipe_schedule=config.training.fixed_recipe_schedule,
                            fixed_depth=fixed_depth,
                        )
                    else:
                        exact = model.forward_exact(
                            tokens,
                            targets,
                            return_loss_per_sample=True,
                            fixed_recipe=fixed_recipe,
                            fixed_depth=fixed_depth,
                        )
                exact_endpoint = (
                    exact.meta.recurrent_hidden
                    if exact.meta.recurrent_hidden is not None
                    else exact.meta.hidden
                )
                exact_logits = exact.meta.logits
                exact_loss_per_sample = exact.loss_per_sample
                h0_for_loss = exact.meta.h0
            if (
                config.model.macro_type == "v2_delta_radius"
                and config.model.macro_radius_init_from_teacher
                and h0_for_loss is not None
                and exact_endpoint is not None
            ):
                idx = model.macro.stride_to_idx[fixed_depth]
                current_radius = model.macro.teacher_delta_rms[fixed_recipe, idx]
                if float(current_radius.detach().cpu()) == 0.0:
                    radius = (
                        exact_endpoint.detach() - h0_for_loss.detach()
                    ).float().pow(2).mean(dim=(1, 2)).sqrt().mean()
                    model.macro.initialize_radius_from_teacher_delta(
                        fixed_recipe,
                        fixed_depth,
                        radius,
                    )
            hot = model.forward_macro(
                tokens,
                targets,
                return_loss_per_sample=True,
                fixed_recipe=fixed_recipe,
                fixed_depth=fixed_depth,
            )
            hot_endpoint = (
                hot.meta.recurrent_hidden if hot.meta.recurrent_hidden is not None else hot.meta.hidden
            )
            if hot_endpoint is None or exact_endpoint is None:
                raise RuntimeError("macro teacher requires hidden states")
            if exact_loss_per_sample is None:
                raise RuntimeError("macro teacher requires exact per-sample loss")
            loss_hot_endpoint = apply_macro_rms_clamp(
                hot_endpoint,
                exact_endpoint,
                enabled=config.training.macro_rms_clamp_early,
                min_scale=config.training.macro_rms_clamp_min,
                max_scale=config.training.macro_rms_clamp_max,
            )
            losses = macro_distill_loss(
                loss_hot_endpoint,
                exact_endpoint,
                hot.meta.logits,
                exact_logits,
                h0=hot.meta.h0 if hot.meta.h0 is not None else h0_for_loss,
                lambda_hid=0.0,
                lambda_cos=0.0,
                lambda_kl=args.lambda_kl,
                lambda_norm=0.0,
                lambda_delta_dir=config.training.lambda_delta_dir,
                lambda_delta_rms=config.training.lambda_delta_rms,
                lambda_endpoint_normed=config.training.lambda_endpoint_normed,
                lambda_endpoint_raw=config.training.lambda_endpoint_raw,
                lambda_macro_rms_trust=config.training.lambda_macro_rms_trust
                if config.training.macro_rms_trust_region
                else 0.0,
                temperature=args.temperature,
            )
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            macro_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.macro.parameters(),
                config.training.grad_clip_norm
                if config.training.grad_clip_norm is not None
                else float("inf"),
            )
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                with torch.no_grad():
                    refreshed_hot = model.forward_macro(
                        tokens,
                        targets,
                        return_loss_per_sample=True,
                        fixed_recipe=fixed_recipe,
                        fixed_depth=fixed_depth,
                    )
                    assert refreshed_hot.loss_per_sample is not None
                    refreshed_endpoint = (
                        refreshed_hot.meta.recurrent_hidden
                        if refreshed_hot.meta.recurrent_hidden is not None
                        else refreshed_hot.meta.hidden
                    )
                    hot_nll = refreshed_hot.loss_per_sample.mean() / float(targets.shape[1])
                    exact_nll = exact_loss_per_sample.mean() / float(targets.shape[1])
                    align = macro_alignment_metrics(
                        refreshed_endpoint,
                        exact_endpoint,
                        h0=refreshed_hot.meta.h0 if refreshed_hot.meta.h0 is not None else h0_for_loss,
                        hot_logits=refreshed_hot.meta.logits,
                        exact_logits=exact_logits,
                        hot_nll=hot_nll,
                        exact_nll=exact_nll,
                    )
                row = {
                    "event": "macro_teacher",
                    "mode": "recursive_macro_distill_only",
                    "step": step,
                    "fixed_recipe": fixed_recipe,
                    "fixed_depth": fixed_depth,
                    "teacher_checkpoint_source": args.teacher_checkpoint,
                    "teacher_mode": args.teacher_mode,
                    "boundary_cache": args.boundary_cache,
                    "coda_trainable": any(param.requires_grad for param in model.coda.parameters()),
                    "macro_lr": optimizer.param_groups[0]["lr"],
                    "coda_lr": 0.0,
                    "macro_grad_norm": float(macro_grad_norm.detach().float().cpu()),
                    "coda_grad_norm": 0.0,
                    "loss": float(loss.detach().float().cpu()),
                    "macro_distill_loss": float(loss.detach().float().cpu()),
                    "macro_hidden_loss": float(losses["hid"].detach().float().cpu()),
                    "macro_logit_kl_loss": float(losses["kl"].detach().float().cpu()),
                    "macro_norm_loss": float(losses["norm"].detach().float().cpu()),
                    "macro_delta_dir_loss": float(losses["delta_dir"].detach().float().cpu()),
                    "macro_delta_rms_loss": float(losses["delta_rms"].detach().float().cpu()),
                    "macro_endpoint_normed_loss": float(
                        losses["endpoint_normed"].detach().float().cpu()
                    ),
                    "macro_endpoint_raw_loss": float(losses["endpoint_raw"].detach().float().cpu()),
                    "macro_rms_trust_loss": float(losses["rms_trust"].detach().float().cpu()),
                    "exact_eval_nll": float(exact_nll.detach().float().cpu()),
                    "hot_eval_nll": float(hot_nll.detach().float().cpu()),
                    **{
                        key: float(value.detach().float().cpu())
                        for key, value in align.items()
                    },
                    **{
                        f"macro_{key}": float(value.detach().float().cpu())
                        for key, value in losses.items()
                    },
                }
                rows.append(row)
                logger.write(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        logger.close()
    save_checkpoint = args.save_checkpoint
    if save_checkpoint is None and args.run_dir is not None:
        save_checkpoint = str(run_dir / "checkpoint.pt")
    if save_checkpoint:
        path = Path(save_checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rows": rows,
                "teacher_checkpoint_source": args.teacher_checkpoint,
            },
            path,
        )


def cmd_run_ablations(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    results = []
    specs = build_ablation_configs(base)
    if args.max_ablations is not None:
        specs = specs[: args.max_ablations]
    for spec in specs:
        cfg = spec.config
        streams = load_token_streams(cfg.data, cfg.training, cfg.model.vocab_size)
        engine = TrainEngine(cfg)
        try:
            batches = streams.train_batches(cfg.training)
            result = None
            for _ in range(args.steps):
                result = engine.train_step(next(batches))
            assert result is not None
            results.append(
                {
                    "name": spec.name,
                    "category": spec.category,
                    "mode": cfg.training.mode,
                    "loss": float(result.loss.detach().float().cpu()),
                    "stored_params": dense_param_count(cfg.model)
                    if cfg.training.mode == "dense_exact"
                    else recursive_param_count(cfg.model),
                }
            )
        finally:
            engine.close()
    _print_json(results)


def cmd_compare_ttq(args: argparse.Namespace) -> None:
    rows = []
    for path in Path(args.runs_dir).glob("*/metrics.jsonl"):
        mode_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        reached = [r for r in mode_rows if r.get("loss", float("inf")) <= args.target_loss]
        rows.append(
            {
                "run": path.parent.name,
                "target_loss": args.target_loss,
                "reached": bool(reached),
                "first_step": reached[0]["step"] if reached else None,
            }
        )
    _print_json(rows)


def cmd_summarize_run(args: argparse.Namespace) -> None:
    _print_json(summarize_metrics(args.run))


def cmd_summarize_comparison(args: argparse.Namespace) -> None:
    summaries = [summarize_metrics(path) for path in args.runs]
    rows = []
    for summary in summaries:
        last = summary["last"]
        rows.append(
            {
                "run": summary["metrics_path"],
                "rows": summary["rows"],
                "last_step": summary["last_step"],
                "mode": last.get("mode"),
                "nll_per_token": last.get("nll_per_token"),
                "tokens_per_sec": last.get("tokens_per_sec"),
                "active_param_equiv_per_token": last.get("active_param_equiv_per_token"),
                "audit_rate": last.get("audit_audit_rate"),
                "backend": summary.get("manifest", {}).get("backend", {}).get("mode"),
            }
        )
    _print_json(rows)


def _metrics_rows(run: str | Path) -> list[dict[str, Any]]:
    path = Path(run)
    metrics = path if path.is_file() else path / "metrics.jsonl"
    return [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]


def cmd_summarize_alignment(args: argparse.Namespace) -> None:
    rows = _metrics_rows(args.run)
    numeric = [
        row
        for row in rows
        if any(key in row for key in ("exact_eval_nll", "eval_exact_nll_per_token", "hot_eval_nll"))
    ]
    if not numeric:
        raise SystemExit(f"no alignment metrics found in {args.run}")

    def values(*keys: str) -> list[float]:
        out = []
        for row in numeric:
            for key in keys:
                if isinstance(row.get(key), (int, float)):
                    out.append(float(row[key]))
                    break
        return out

    exact_vals = values("exact_eval_nll", "eval_exact_nll_per_token")
    hot_vals = values("hot_eval_nll", "eval_hot_nll_per_token")
    hidden_mse_vals = values("hidden_mse_exact_macro", "audit_hidden_mse_exact_macro")
    hidden_cos_vals = values("hidden_cosine_exact_macro", "audit_hidden_cosine_exact_macro")
    kl_vals = values("logit_kl_exact_macro", "audit_logit_kl_exact_macro")
    residual_vals = values("audit_residual_var", "audit_audit_residual_var")
    delta_ratio_vals = values("delta_rms_ratio", "audit_delta_rms_ratio")
    macro_norm_vals = values("macro_norm", "audit_macro_norm")
    speed_vals = values("tokens_per_sec")
    last = numeric[-1]
    summary = {
        "run": str(args.run),
        "best_exact_eval_nll": min(exact_vals) if exact_vals else None,
        "best_hot_eval_nll": min(hot_vals) if hot_vals else None,
        "final_exact_hot_gap": last.get("hot_exact_nll_gap"),
        "minimum_hidden_mse": min(hidden_mse_vals) if hidden_mse_vals else None,
        "maximum_hidden_cosine": max(hidden_cos_vals) if hidden_cos_vals else None,
        "minimum_logit_kl": min(kl_vals) if kl_vals else None,
        "delta_rms_ratio_trend": {
            "first": delta_ratio_vals[0] if delta_ratio_vals else None,
            "last": delta_ratio_vals[-1] if delta_ratio_vals else None,
        },
        "macro_norm_trend": {
            "first": macro_norm_vals[0] if macro_norm_vals else None,
            "last": macro_norm_vals[-1] if macro_norm_vals else None,
        },
        "audit_residual_trend": {
            "first_var": residual_vals[0] if residual_vals else None,
            "last_var": residual_vals[-1] if residual_vals else None,
        },
        "last_tokens_per_sec": last.get("tokens_per_sec"),
        "best_tokens_per_sec": max(speed_vals) if speed_vals else None,
        "speed_vs_dense": None,
        "time_to_dense_target_nll": None,
    }
    _print_json(summary)


@torch.no_grad()
def cmd_collect_boundaries(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    batches = streams.train_batches(config.training)
    cache: dict[str, Any] = {
        "config": dataclasses.asdict(config),
        "checkpoint": args.checkpoint,
        "fixed_recipe": config.training.fixed_recipe,
        "fixed_depth": config.training.fixed_depth,
        "data_fingerprint": streams.data_fingerprint,
        "batches": [],
    }
    for _ in range(args.num_batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        out = model.forward_exact(
            tokens,
            targets,
            return_states=True,
            return_loss_per_sample=True,
            fixed_recipe=config.training.fixed_recipe,
            fixed_depth=config.training.fixed_depth,
        )
        cache["batches"].append(
            {
                "tokens": tokens.detach().cpu(),
                "targets": targets.detach().cpu(),
                "h0": out.meta.h0.detach().cpu() if out.meta.h0 is not None else None,
                "states": {
                    depth: state.detach().cpu()
                    for depth, state in (out.meta.states or {}).items()
                },
                "recurrent_hidden": out.meta.recurrent_hidden.detach().cpu()
                if out.meta.recurrent_hidden is not None
                else None,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    _print_json({"output": str(output), "num_batches": args.num_batches})


@torch.no_grad()
def cmd_diagnose_coda_collusion(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = default_device()
    model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    teacher = None
    if args.teacher_checkpoint:
        teacher = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
        teacher_payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        teacher.load_state_dict(teacher_payload["model"], strict=True)
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    tokens, targets = next(streams.eval_batches(config.training))
    tokens = tokens.to(device)
    targets = targets.to(device)
    exact = model.forward_exact(
        tokens,
        targets,
        return_loss_per_sample=True,
        fixed_recipe=config.training.fixed_recipe,
        fixed_depth=config.training.fixed_depth,
    )
    macro = model.forward_macro(
        tokens,
        targets,
        return_loss_per_sample=True,
        fixed_recipe=config.training.fixed_recipe,
        fixed_depth=config.training.fixed_depth,
    )
    assert exact.loss_per_sample is not None and macro.loss_per_sample is not None
    exact_h = exact.meta.recurrent_hidden
    macro_h = macro.meta.recurrent_hidden
    if exact_h is None or macro_h is None:
        raise SystemExit("checkpoint did not expose recurrent hidden endpoints")
    teacher_exact_ce = None
    teacher_macro_ce = None
    if teacher is not None:
        _, teacher_exact_logits = teacher._coda_logits(exact_h)
        _, teacher_macro_logits = teacher._coda_logits(macro_h)
        from recursive_training_engine.models import lm_loss_per_sample

        teacher_exact_ce = float(
            (lm_loss_per_sample(teacher_exact_logits, targets).mean() / targets.shape[1]).cpu()
        )
        teacher_macro_ce = float(
            (lm_loss_per_sample(teacher_macro_logits, targets).mean() / targets.shape[1]).cpu()
        )
    _print_json(
        {
            "ce_coda_h_exact": float((exact.loss_per_sample.mean() / targets.shape[1]).cpu()),
            "ce_coda_h_macro": float((macro.loss_per_sample.mean() / targets.shape[1]).cpu()),
            "ce_teacher_coda_h_exact": teacher_exact_ce,
            "ce_teacher_coda_h_macro": teacher_macro_ce,
            "hidden_mse_exact_macro": float((exact_h - macro_h).square().mean().cpu()),
            "hidden_cosine_exact_macro": float(hidden_cosine(macro_h, exact_h).mean().cpu()),
            "logit_kl_exact_macro": float(logit_kl(exact.logits, macro.logits).mean().cpu())
            if exact.logits is not None and macro.logits is not None
            else None,
            "coda_collusion_suspected": bool(
                macro.loss_per_sample.mean() + 0.05 * targets.shape[1]
                < exact.loss_per_sample.mean()
            ),
        }
    )


@torch.no_grad()
def cmd_diagnose_macro_range(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    fixed_depth = args.fixed_depth if args.fixed_depth is not None else config.training.fixed_depth
    fixed_recipe = args.fixed_recipe if args.fixed_recipe is not None else config.training.fixed_recipe
    if fixed_depth is None or fixed_recipe is None:
        raise SystemExit("diagnose-macro-range requires fixed_depth and fixed_recipe")
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, topology="recursive"),
        training=dataclasses.replace(config.training, fixed_depth=fixed_depth, fixed_recipe=fixed_recipe),
    )
    device = default_device()
    model = RecursiveModel(config.model, config.output).to(device)
    if not Path(args.teacher_checkpoint).exists():
        raise SystemExit(f"teacher checkpoint not found: {args.teacher_checkpoint}")
    payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    if payload.get("model") is None:
        raise SystemExit(f"teacher checkpoint has no model state: {args.teacher_checkpoint}")
    load_compatible_state_dict(model, payload["model"], skip_prefixes=("macro.",))
    model.eval()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    batches = streams.eval_batches(config.training)
    rows: list[dict[str, torch.Tensor]] = []
    hard_bound = None
    idx = model.macro.stride_to_idx[fixed_depth]
    if config.model.macro_type != "v2_delta_radius":
        scale = model.macro.update_scale[fixed_recipe, idx]
        hard_bound = float((config.model.macro_update_scale * scale.abs().max()).detach().cpu())
    for _ in range(args.batches):
        tokens, targets = next(batches)
        tokens = tokens.to(device)
        targets = targets.to(device)
        exact = model.forward_exact(
            tokens,
            targets,
            return_loss_per_sample=True,
            fixed_recipe=fixed_recipe,
            fixed_depth=fixed_depth,
        )
        exact_h = exact.meta.recurrent_hidden
        if exact_h is None or exact.meta.h0 is None or exact.loss_per_sample is None:
            raise SystemExit("exact path did not expose h0/recurrent hidden/per-sample loss")
        if (
            config.model.macro_type == "v2_delta_radius"
            and config.model.macro_radius_init_from_teacher
            and float(model.macro.teacher_delta_rms[fixed_recipe, idx].detach().cpu()) == 0.0
        ):
            radius = (exact_h - exact.meta.h0).float().pow(2).mean(dim=(1, 2)).sqrt().mean()
            model.macro.initialize_radius_from_teacher_delta(fixed_recipe, fixed_depth, radius)
        macro = model.forward_macro(
            tokens,
            targets,
            return_loss_per_sample=True,
            fixed_recipe=fixed_recipe,
            fixed_depth=fixed_depth,
        )
        macro_h = macro.meta.recurrent_hidden
        if macro_h is None or macro.loss_per_sample is None:
            raise SystemExit("macro path did not expose recurrent hidden/per-sample loss")
        exact_delta = exact_h - exact.meta.h0
        macro_delta = macro_h - exact.meta.h0
        exact_nll = exact.loss_per_sample.mean() / float(targets.shape[1])
        hot_nll = macro.loss_per_sample.mean() / float(targets.shape[1])
        align = macro_alignment_metrics(
            macro_h,
            exact_h,
            h0=exact.meta.h0,
            hot_logits=macro.meta.logits,
            exact_logits=exact.meta.logits,
            hot_nll=hot_nll,
            exact_nll=exact_nll,
        )
        rows.append(
            {
                "rms_h0": exact.meta.h0.float().pow(2).mean().sqrt(),
                "rms_h_exact": exact_h.float().pow(2).mean().sqrt(),
                "rms_delta_exact": exact_delta.float().pow(2).mean().sqrt(),
                "max_abs_delta_exact": exact_delta.abs().max(),
                "rms_delta_macro_initial": macro_delta.float().pow(2).mean().sqrt(),
                "max_abs_delta_macro_initial": macro_delta.abs().max(),
                **align,
            }
        )

    def avg(key: str) -> float:
        values = [row[key].detach().float().cpu() for row in rows if key in row]
        return float(torch.stack(values).mean()) if values else 0.0

    macro_capacity = hard_bound
    if config.model.macro_type == "v2_delta_radius":
        macro_capacity = float(model.macro.current_radius(
            torch.tensor([fixed_recipe], device=device),
            torch.tensor([idx], device=device),
        )[0].detach().cpu())
    report = {
        "rms_h0": avg("rms_h0"),
        "rms_h_exact": avg("rms_h_exact"),
        "rms_delta_exact": avg("rms_delta_exact"),
        "max_abs_delta_exact": max(float(row["max_abs_delta_exact"].detach().cpu()) for row in rows),
        "rms_delta_macro_initial": avg("rms_delta_macro_initial"),
        "max_abs_delta_macro_initial": max(
            float(row["max_abs_delta_macro_initial"].detach().cpu()) for row in rows
        ),
        "macro_update_scale": config.model.macro_update_scale,
        "macro_delta_capacity_estimate": macro_capacity,
        "ratio_delta_exact_to_macro": avg("rms_delta_exact") / max(float(macro_capacity or 0.0), 1e-12),
        "hidden_cosine_initial": avg("hidden_cosine_exact_macro"),
        "hidden_mse_initial": avg("hidden_mse_exact_macro"),
        "logit_kl_initial": avg("logit_kl_exact_macro"),
        "hot_exact_nll_gap_initial": avg("hot_exact_nll_gap"),
        "macro_norm_initial": avg("macro_norm"),
        "exact_norm_initial": avg("exact_norm"),
        "macro_exact_norm_ratio_initial": avg("macro_exact_norm_ratio"),
        "max_possible_per_component_macro_delta": hard_bound,
    }
    _print_json(report)


def _capacity_candidate_configs(config: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    specs = [
        ("A_old_rank32", {"macro_type": "bounded_residual", "macro_rank": 32}),
        ("B_v2_rank32", {"macro_type": "v2_delta_radius", "macro_rank": 32}),
        ("C_v2_rank64", {"macro_type": "v2_delta_radius", "macro_rank": 64}),
        ("D_v2_rank128", {"macro_type": "v2_delta_radius", "macro_rank": 128}),
        ("E_v2_rank64_hidden4", {"macro_type": "v2_delta_radius", "macro_rank": 64, "macro_hidden_mult": 4}),
        ("F_v2_rank128_hidden4", {"macro_type": "v2_delta_radius", "macro_rank": 128, "macro_hidden_mult": 4}),
    ]
    out = []
    for name, overrides in specs:
        model = dataclasses.replace(
            config.model,
            **{
                **overrides,
                "macro_radius_init_from_teacher": overrides.get("macro_type")
                == "v2_delta_radius",
                "macro_use_depth_embedding": True
                if overrides.get("macro_type") == "v2_delta_radius"
                else config.model.macro_use_depth_embedding,
                "macro_use_recipe_embedding": True
                if overrides.get("macro_type") == "v2_delta_radius"
                else config.model.macro_use_recipe_embedding,
                "macro_use_delta_to_h0": True
                if overrides.get("macro_type") == "v2_delta_radius"
                else config.model.macro_use_delta_to_h0,
            },
        )
        out.append((name, dataclasses.replace(config, model=model)))
    return out


def cmd_run_macro_capacity_ladder(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    fixed_depth = args.fixed_depth if args.fixed_depth is not None else base.training.fixed_depth
    fixed_recipe = args.fixed_recipe if args.fixed_recipe is not None else base.training.fixed_recipe
    if fixed_depth is None or fixed_recipe is None:
        raise SystemExit("run-macro-capacity-ladder requires fixed_depth and fixed_recipe")
    base = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, topology="recursive"),
        training=dataclasses.replace(
            base.training,
            mode="recursive_macro_distill_only",
            fixed_depth=fixed_depth,
            fixed_recipe=fixed_recipe,
            audit_p_min=0.0,
            audit_p_max=0.0,
        ),
    )
    device = default_device()
    streams = load_token_streams(base.data, base.training, base.model.vocab_size)
    results = []
    import time

    for name, config in _capacity_candidate_configs(base):
        model = RecursiveModel(config.model, config.output).to(device)
        if not Path(args.teacher_checkpoint).exists():
            raise SystemExit(f"teacher checkpoint not found: {args.teacher_checkpoint}")
        payload = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
        if payload.get("model") is None:
            raise SystemExit(f"teacher checkpoint has no model state: {args.teacher_checkpoint}")
        load_compatible_state_dict(model, payload["model"], skip_prefixes=("macro.",))
        for param in model.parameters():
            param.requires_grad_(False)
        for param in model.macro.parameters():
            param.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.macro.parameters(),
            lr=config.training.effective_lr_macro,
            weight_decay=config.training.weight_decay,
        )
        batches = streams.train_batches(config.training)
        last_row: dict[str, torch.Tensor] | None = None
        start = time.perf_counter()
        tokens_seen = 0
        iterations = max(1, args.steps)
        for step in range(iterations):
            tokens, targets = next(batches)
            tokens = tokens.to(device)
            targets = targets.to(device)
            tokens_seen += int(tokens.numel())
            with torch.no_grad():
                exact = model.forward_exact(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    fixed_recipe=fixed_recipe,
                    fixed_depth=fixed_depth,
                )
            exact_h = exact.meta.recurrent_hidden
            if exact_h is None or exact.meta.h0 is None or exact.loss_per_sample is None:
                raise SystemExit("exact path did not expose h0/recurrent hidden/per-sample loss")
            if (
                config.model.macro_type == "v2_delta_radius"
                and config.model.macro_radius_init_from_teacher
            ):
                idx = model.macro.stride_to_idx[fixed_depth]
                if float(model.macro.teacher_delta_rms[fixed_recipe, idx].detach().cpu()) == 0.0:
                    radius = (exact_h - exact.meta.h0).float().pow(2).mean(dim=(1, 2)).sqrt().mean()
                    model.macro.initialize_radius_from_teacher_delta(fixed_recipe, fixed_depth, radius)
            hot = model.forward_macro(
                tokens,
                targets,
                return_loss_per_sample=True,
                fixed_recipe=fixed_recipe,
                fixed_depth=fixed_depth,
            )
            hot_h = hot.meta.recurrent_hidden
            if hot_h is None or hot.loss_per_sample is None:
                raise SystemExit("macro path did not expose recurrent hidden/per-sample loss")
            losses = macro_distill_loss(
                apply_macro_rms_clamp(
                    hot_h,
                    exact_h,
                    enabled=config.training.macro_rms_clamp_early,
                    min_scale=config.training.macro_rms_clamp_min,
                    max_scale=config.training.macro_rms_clamp_max,
                ),
                exact_h,
                hot.meta.logits,
                exact.meta.logits,
                h0=hot.meta.h0,
                lambda_kl=config.training.effective_lambda_kl,
                lambda_delta_dir=config.training.lambda_delta_dir,
                lambda_delta_rms=config.training.lambda_delta_rms,
                lambda_endpoint_normed=config.training.lambda_endpoint_normed,
                lambda_endpoint_raw=config.training.lambda_endpoint_raw,
                temperature=config.training.distill_temperature,
            )
            loss = sum(losses.values())
            if args.steps > 0:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            exact_nll = exact.loss_per_sample.mean() / float(targets.shape[1])
            hot_nll = hot.loss_per_sample.mean() / float(targets.shape[1])
            last_row = macro_alignment_metrics(
                hot_h,
                exact_h,
                h0=hot.meta.h0,
                hot_logits=hot.meta.logits,
                exact_logits=exact.meta.logits,
                hot_nll=hot_nll,
                exact_nll=exact_nll,
            )
            if step + 1 >= args.steps and args.steps > 0:
                break
        elapsed = max(time.perf_counter() - start, 1e-9)
        assert last_row is not None
        results.append(
            {
                "name": name,
                "macro_type": config.model.macro_type,
                "rank": config.model.macro_rank,
                "hidden_mult": config.model.macro_hidden_mult,
                "delta_rms_ratio": float(last_row.get("delta_rms_ratio", torch.tensor(0.0)).detach().cpu()),
                "hidden_cosine": float(
                    last_row["hidden_cosine_exact_macro"].detach().float().cpu()
                ),
                "hidden_mse": float(last_row["hidden_mse_exact_macro"].detach().float().cpu()),
                "logit_kl": float(last_row.get("logit_kl_exact_macro", torch.tensor(0.0)).detach().cpu()),
                "hot_exact_gap": float(last_row.get("hot_exact_nll_gap", torch.tensor(0.0)).detach().cpu()),
                "tokens_per_sec": tokens_seen / elapsed,
            }
        )
    _print_json(results)


def cmd_compare_configs(args: argparse.Namespace) -> None:
    old = load_config(args.old_config)
    new = load_config(args.new_config)
    fields_to_compare = {
        "macro_type": (old.model.macro_type, new.model.macro_type),
        "macro_update_scale": (old.model.macro_update_scale, new.model.macro_update_scale),
        "macro_rank": (old.model.macro_rank, new.model.macro_rank),
        "macro_hidden_mult": (old.model.macro_hidden_mult, new.model.macro_hidden_mult),
        "lambda_delta_dir": (old.training.lambda_delta_dir, new.training.lambda_delta_dir),
        "lambda_delta_rms": (old.training.lambda_delta_rms, new.training.lambda_delta_rms),
        "lambda_endpoint_normed": (
            old.training.lambda_endpoint_normed,
            new.training.lambda_endpoint_normed,
        ),
        "lambda_endpoint_raw": (old.training.lambda_endpoint_raw, new.training.lambda_endpoint_raw),
        "lambda_logit_kl": (old.training.effective_lambda_kl, new.training.effective_lambda_kl),
        "loss_normalization": (old.training.loss_normalization, new.training.loss_normalization),
        "fixed_depth": (old.training.fixed_depth, new.training.fixed_depth),
        "fixed_recipe": (old.training.fixed_recipe, new.training.fixed_recipe),
        "data_lane": (old.data.projection_lane, new.data.projection_lane),
        "vocab_projection_lane": (old.data.vocab_projection, new.data.vocab_projection),
        "coda_warmup": (old.training.coda_warmup_steps, new.training.coda_warmup_steps),
        "audit_mode": (old.training.audit_mode, new.training.audit_mode),
        "audit_cap": (old.training.audit_cap, new.training.audit_cap),
        "audit_count": (
            old.training.audit_fixed_count_per_batch,
            new.training.audit_fixed_count_per_batch,
        ),
        "output_mode": (old.output.mode, new.output.mode),
        "teacher_checkpoint_source": (
            old.training.aligned_lm_teacher_checkpoint,
            new.training.aligned_lm_teacher_checkpoint,
        ),
        "teacher_frozen": (old.training.aligned_lm_freeze_teacher, new.training.aligned_lm_freeze_teacher),
        "phase_a_trainable": (old.training.aligned_lm_phase_a_train, new.training.aligned_lm_phase_a_train),
        "phase_b_trainable": (old.training.aligned_lm_phase_b_train, new.training.aligned_lm_phase_b_train),
        "phase_c_trainable": (old.training.aligned_lm_phase_c_train, new.training.aligned_lm_phase_c_train),
    }
    _print_json(
        {
            key: {"old": old_value, "new": new_value, "changed": old_value != new_value}
            for key, (old_value, new_value) in fields_to_compare.items()
        }
    )


def _clone_layer_map(config: ExperimentConfig, depth: int, raw: list[int] | None = None) -> list[int]:
    if raw:
        layers = [int(value) for value in raw]
    elif config.training.dense_hidden_layer_map:
        layers = [int(value) for value in config.training.dense_hidden_layer_map]
    else:
        pre_coda = config.model.n_dense_layers - config.model.n_coda
        layers = [
            round(config.model.n_prelude + (idx + 1) * (pre_coda - config.model.n_prelude) / depth)
            for idx in range(depth)
        ]
    if len(layers) < depth:
        raise ValueError(f"dense layer map has {len(layers)} entries but depth={depth}")
    layers = layers[:depth]
    max_layer = config.model.n_dense_layers
    if any(layer < 1 or layer > max_layer for layer in layers):
        raise ValueError(f"dense layer map entries must be in [1, {max_layer}], got {layers}")
    return layers


def _schedule_for_depth(config: ExperimentConfig, depth: int, raw: list[int] | None = None) -> list[int]:
    schedule = list(raw or config.training.fixed_recipe_schedule or [])
    if not schedule:
        schedule = [config.training.fixed_recipe if config.training.fixed_recipe is not None else 1]
    while len(schedule) < depth:
        schedule.append(schedule[-1])
    return [int(value) for value in schedule[:depth]]


def _rms_unit(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)


def _load_dense_model(config: ExperimentConfig, checkpoint: str, device: torch.device) -> DenseModel:
    model = DenseModel(dataclasses.replace(config.model, topology="dense")).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def _load_recursive_model(config: ExperimentConfig, checkpoint: str | None, device: torch.device) -> RecursiveModel:
    model = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
    return model


def _copy_dense_block(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=True)


def _transplant_dense_to_recursive(
    dense: DenseModel,
    recursive: RecursiveModel,
    config: ExperimentConfig,
) -> dict[str, Any]:
    copied: list[str] = []
    recursive.embed.load_state_dict(dense.embed.state_dict(), strict=True)
    copied.append("embed")
    for idx, block in enumerate(recursive.prelude):
        dense_idx = idx
        if dense_idx >= len(dense.blocks):
            break
        _copy_dense_block(block, dense.blocks[dense_idx])
        copied.append(f"prelude.{idx}<-dense.blocks.{dense_idx}")
    coda_start = config.model.n_dense_layers - config.model.n_coda
    for idx, block in enumerate(recursive.coda):
        dense_idx = coda_start + idx
        if dense_idx >= len(dense.blocks):
            break
        _copy_dense_block(block, dense.blocks[dense_idx])
        copied.append(f"coda.{idx}<-dense.blocks.{dense_idx}")
    recursive.final_norm.load_state_dict(dense.final_norm.state_dict(), strict=True)
    copied.append("final_norm")
    if recursive.lm_head is not None and dense.lm_head is not None:
        recursive.lm_head.load_state_dict(dense.lm_head.state_dict(), strict=True)
        copied.append("lm_head")

    middle = list(dense.blocks[config.model.n_prelude : coda_start])
    if middle:
        with torch.no_grad():
            avg_qkv = torch.stack([block.attn.wqkv.weight.detach().t() for block in middle]).mean(dim=0)
            avg_wo = torch.stack([block.attn.wo.weight.detach().t() for block in middle]).mean(dim=0)
            avg_wug = torch.stack([block.mlp.wug.weight.detach().t() for block in middle]).mean(dim=0)
            avg_wd = torch.stack([block.mlp.wd.weight.detach().t() for block in middle]).mean(dim=0)
            avg_norm1 = torch.stack([block.norm1.weight.detach() for block in middle]).mean(dim=0)
            avg_norm2 = torch.stack([block.norm2.weight.detach() for block in middle]).mean(dim=0)
            recursive.core.attn.wqkv.copy_(avg_qkv.unsqueeze(0).expand_as(recursive.core.attn.wqkv))
            recursive.core.attn.wo.copy_(avg_wo.unsqueeze(0).expand_as(recursive.core.attn.wo))
            recursive.core.mlp.wug.copy_(avg_wug.unsqueeze(0).expand_as(recursive.core.mlp.wug))
            recursive.core.mlp.wd.copy_(avg_wd.unsqueeze(0).expand_as(recursive.core.mlp.wd))
            recursive.core.norm1.weight.copy_(avg_norm1)
            recursive.core.norm2.weight.copy_(avg_norm2)
            if recursive.core.alpha_inj is not None:
                recursive.core.alpha_inj.zero_()
        copied.append(f"core<-average_dense_blocks.{config.model.n_prelude}:{coda_start}")
    return {"copied": copied, "middle_block_count": len(middle)}


def cmd_transplant_dense_to_recursive(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.recursive_config), "recursive_exact")
    if args.use_global_lowrank_corrector:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                use_global_lowrank_corrector=True,
                global_corrector_rank=args.global_corrector_rank,
            ),
        )
    device = default_device()
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    recursive = RecursiveModel(dataclasses.replace(config.model, topology="recursive"), config.output).to(device)
    summary = _transplant_dense_to_recursive(dense, recursive, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model": recursive.state_dict(),
            "transplant": summary,
            "dense_checkpoint": args.dense_checkpoint,
        },
        output,
    )
    save_config(config, output.parent / "resolved_config.yaml")
    write_json(output.parent / "manifest.json", {"event": "dense_recursive_transplant", **summary})
    _print_json({"output": str(output), **summary})


def _one_step_clone_losses(
    model: RecursiveModel,
    dense: DenseModel,
    tokens: torch.Tensor,
    *,
    depth: int,
    layer_map: list[int],
    schedule: list[int],
    lambda_hidden: float,
    lambda_delta: float,
    lambda_cos: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dense_out = dense(tokens, return_states=True)
    assert dense_out.meta.states is not None
    h0 = dense_out.meta.states[model.config.n_prelude].detach()
    dense_prev = h0
    total = h0.new_zeros(())
    metrics: dict[str, Any] = {}
    batch = tokens.shape[0]
    for pass_idx, dense_layer in enumerate(layer_map):
        input_layer = model.config.n_prelude if pass_idx == 0 else layer_map[pass_idx - 1]
        input_state = dense_out.meta.states[input_layer].detach()
        target = dense_out.meta.states[dense_layer].detach()
        recipe_ids = torch.full((batch,), schedule[pass_idx], dtype=torch.long, device=tokens.device)
        pred = model.core.forward_step(
            input_state,
            h0,
            recipe_ids,
            torch.ones(batch, dtype=torch.bool, device=tokens.device),
            pass_idx=pass_idx,
        )
        pred_norm = _rms_unit(pred)
        target_norm = _rms_unit(target)
        hidden = F.smooth_l1_loss(pred_norm, target_norm)
        pred_delta = pred.float() - input_state.float()
        target_delta = target.float() - dense_prev.float()
        delta = F.smooth_l1_loss(pred_delta, target_delta)
        cos = F.cosine_similarity(pred_norm.flatten(1), target_norm.flatten(1), dim=-1).mean()
        step_loss = lambda_hidden * hidden + lambda_delta * delta + lambda_cos * (1.0 - cos)
        total = total + step_loss
        metrics[f"step_hidden_loss_t{pass_idx + 1}"] = hidden.detach()
        metrics[f"step_delta_loss_t{pass_idx + 1}"] = delta.detach()
        metrics[f"step_cos_t{pass_idx + 1}"] = cos.detach()
        dense_prev = target
    metrics["step_clone_loss"] = total.detach()
    metrics["step_cos_mean"] = torch.stack(
        [metrics[f"step_cos_t{idx + 1}"] for idx in range(depth)]
    ).mean()
    return total / float(depth), metrics


def cmd_operator_clone(args: argparse.Namespace) -> None:
    config = _with_run_dir(_model_for_mode(load_config(args.config), "recursive_exact"), args.run_dir)
    if args.use_global_lowrank_corrector:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                use_global_lowrank_corrector=True,
                global_corrector_rank=args.global_corrector_rank,
            ),
        )
    set_seed(config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    recursive = _load_recursive_model(config, args.recursive_checkpoint, device)
    recursive.train()
    for param in recursive.parameters():
        param.requires_grad_(False)
    for param in recursive.core.parameters():
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [param for param in recursive.core.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "resolved_config.yaml")
    layer_map = _clone_layer_map(config, args.depth, args.dense_layer_map)
    schedule = _schedule_for_depth(config, args.depth, args.recipe_schedule)
    write_json(
        run_dir / "manifest.json",
        {
            "event": "operator_clone",
            "dense_checkpoint": args.dense_checkpoint,
            "recursive_checkpoint": args.recursive_checkpoint,
            "layer_map": layer_map,
            "recipe_schedule": schedule,
            "steps": args.steps,
        },
    )
    metrics_path = run_dir / "metrics.jsonl"
    batches = streams.train_batches(config.training)
    last_metrics: dict[str, Any] = {}
    for step in range(args.steps):
        tokens, _ = next(batches)
        tokens = tokens.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _one_step_clone_losses(
            recursive,
            dense,
            tokens,
            depth=args.depth,
            layer_map=layer_map,
            schedule=schedule,
            lambda_hidden=args.lambda_hidden,
            lambda_delta=args.lambda_delta,
            lambda_cos=args.lambda_cos,
        )
        loss.backward()
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(recursive.core.parameters(), args.grad_clip_norm)
        grad_sq = 0.0
        for param in recursive.core.parameters():
            if param.grad is not None:
                grad_sq += float(param.grad.detach().float().square().sum().cpu())
        optimizer.step()
        printable = {
            key: (float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else value)
            for key, value in metrics.items()
        }
        printable.update(
            {
                "step": step + 1,
                "loss": float(loss.detach().float().cpu()),
                "grad_norm_core": grad_sq**0.5,
                "layer_map": layer_map,
                "recipe_schedule": schedule,
            }
        )
        last_metrics = printable
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(printable, sort_keys=True) + "\n")
        if step % args.log_every == 0:
            print(json.dumps(printable, sort_keys=True))
    save_path = Path(args.save_checkpoint) if args.save_checkpoint else run_dir / "checkpoint.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model": recursive.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": args.steps,
            "operator_clone": {
                "layer_map": layer_map,
                "recipe_schedule": schedule,
                "last_metrics": last_metrics,
            },
        },
        save_path,
    )
    _print_json({"checkpoint": str(save_path), "last_metrics": last_metrics})


def _rollout_similarity_metrics(
    recursive: RecursiveModel,
    dense: DenseModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    *,
    depth: int,
    layer_map: list[int],
    schedule: list[int],
    temperature: float,
) -> dict[str, float]:
    dense_out = dense(tokens, targets, return_loss_per_sample=True, return_states=True)
    rec_out = recursive.forward_exact(
        tokens,
        targets,
        return_states=True,
        return_loss_per_sample=True,
        fixed_recipe=schedule[0],
        fixed_recipe_schedule=schedule,
        fixed_depth=depth,
        state_depths=list(range(1, depth + 1)),
    )
    assert dense_out.loss_per_sample is not None and rec_out.loss_per_sample is not None
    assert dense_out.meta.states is not None and rec_out.meta.states is not None
    h0 = dense_out.meta.states[recursive.config.n_prelude].detach()
    dense_prev = h0
    rec_prev = rec_out.meta.h0 if rec_out.meta.h0 is not None else h0
    hidden_mse = 0.0
    delta_mse = 0.0
    cos = 0.0
    for rec_depth, dense_layer in zip(range(1, depth + 1), layer_map, strict=True):
        rec_state = rec_out.meta.states[rec_depth]
        dense_state = dense_out.meta.states[dense_layer].detach()
        rec_norm = _rms_unit(rec_state)
        dense_norm = _rms_unit(dense_state)
        hidden_mse += float((rec_norm - dense_norm).square().mean().detach().cpu())
        delta_mse += float(
            (rec_state.float() - rec_prev.float() - (dense_state.float() - dense_prev.float()))
            .square()
            .mean()
            .detach()
            .cpu()
        )
        cos += float(F.cosine_similarity(rec_norm.flatten(1), dense_norm.flatten(1), dim=-1).mean().detach().cpu())
        rec_prev = rec_state
        dense_prev = dense_state
    dense_logp = F.log_softmax(dense_out.meta.logits.detach() / temperature, dim=-1)
    rec_logp = F.log_softmax(rec_out.meta.logits / temperature, dim=-1)
    kl = (dense_logp.exp() * (dense_logp - rec_logp)).sum(dim=-1).mean() * (temperature * temperature)
    return {
        "dense_nll": float(dense_out.loss_per_sample.mean().detach().cpu()) / float(targets.shape[1]),
        "recursive_nll": float(rec_out.loss_per_sample.mean().detach().cpu()) / float(targets.shape[1]),
        "rollout_hidden_mse": hidden_mse / float(depth),
        "rollout_delta_mse": delta_mse / float(depth),
        "rollout_hidden_cosine": cos / float(depth),
        "dense_kl": float(kl.detach().cpu()),
    }


def cmd_evaluate_rollout_similarity(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "recursive_exact")
    if args.use_global_lowrank_corrector:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                use_global_lowrank_corrector=True,
                global_corrector_rank=args.global_corrector_rank,
            ),
        )
    set_seed(config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    recursive = _load_recursive_model(config, args.recursive_checkpoint, device)
    dense.eval()
    recursive.eval()
    depth = args.depth
    layer_map = _clone_layer_map(config, depth, args.dense_layer_map)
    schedule = _schedule_for_depth(config, depth, args.recipe_schedule)
    totals: dict[str, float] = {}
    with torch.no_grad():
        batches = streams.eval_batches(config.training)
        for _ in range(args.num_batches):
            tokens, targets = next(batches)
            metrics = _rollout_similarity_metrics(
                recursive,
                dense,
                tokens.to(device),
                targets.to(device),
                depth=depth,
                layer_map=layer_map,
                schedule=schedule,
                temperature=args.temperature,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
    averaged = {key: value / float(args.num_batches) for key, value in totals.items()}
    averaged.update(
        {
            "num_batches": args.num_batches,
            "eval_tokens": args.num_batches * config.training.batch_size * config.training.seq_len,
            "layer_map": layer_map,
            "recipe_schedule": schedule,
        }
    )
    _print_json(averaged)


def _parse_rank_list(raw: str | list[int] | None) -> list[int]:
    if raw is None:
        return [0, 4, 8, 16, 32, 64]
    if isinstance(raw, list):
        ranks = [int(value) for value in raw]
    else:
        ranks = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not ranks:
        raise ValueError("candidate rank list must not be empty")
    if any(rank < 0 for rank in ranks):
        raise ValueError("candidate ranks must be non-negative")
    return sorted(set(ranks))


def _flatten_sample_positions(
    tensor: torch.Tensor,
    *,
    positions: torch.Tensor,
) -> torch.Tensor:
    flat = tensor.reshape(-1, tensor.shape[-1])
    return flat.index_select(0, positions)


def _compiler_feature_matrix(
    input_state: torch.Tensor,
    h0: torch.Tensor,
    *,
    positions: torch.Tensor,
) -> torch.Tensor:
    x = _flatten_sample_positions(_rms_unit(input_state), positions=positions)
    x0 = _flatten_sample_positions(_rms_unit(h0), positions=positions)
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return torch.cat([x, x0, ones], dim=-1).float()


def _compiler_sample_positions(
    tokens: torch.Tensor,
    *,
    positions_per_batch: int,
    generator: torch.Generator,
) -> torch.Tensor:
    total = tokens.numel()
    count = min(max(1, int(positions_per_batch)), total)
    if count == total:
        return torch.arange(total, device=tokens.device)
    return torch.randperm(total, generator=generator, device=tokens.device)[:count]


@torch.no_grad()
def _dense_suffix_logits_from_state(
    dense: DenseModel,
    state: torch.Tensor,
    *,
    state_layer: int,
) -> torch.Tensor:
    x = state
    for block in list(dense.blocks)[state_layer:]:
        x = block(x)
    hidden = dense.final_norm(x)
    return hidden @ dense.vocab_weight.t()


def _compiler_transition_cost(
    row: dict[str, Any],
    rank: int,
    metric: str,
) -> float:
    if metric == "residual_rms":
        return float(row[f"residual_rms_after_rank_{rank}"])
    if metric == "hidden_cosine":
        return 1.0 - float(row[f"hidden_cosine_rank_{rank}"])
    raise ValueError(f"unsupported compiler cost metric: {metric}")


def _choose_compiler_partition(
    rows: list[dict[str, Any]],
    *,
    start_layer: int,
    end_layer: int,
    depth: int,
    ranks: list[int],
    rank_budget: int,
    metric: str,
) -> dict[str, Any]:
    by_key = {
        (int(row["pass_idx"]), int(row["from_layer"]), int(row["to_layer"])): row
        for row in rows
    }
    dp: dict[tuple[int, int, int], tuple[float, list[dict[str, Any]]]] = {
        (0, start_layer, 0): (0.0, [])
    }
    for pass_idx in range(depth):
        next_dp: dict[tuple[int, int, int], tuple[float, list[dict[str, Any]]]] = {}
        for (used_passes, layer, used_rank), (cost, path) in dp.items():
            if used_passes != pass_idx:
                continue
            min_remaining = depth - pass_idx - 1
            for to_layer in range(layer + 1, end_layer + 1):
                if end_layer - to_layer < min_remaining:
                    continue
                row = by_key.get((pass_idx, layer, to_layer))
                if row is None:
                    continue
                for rank in ranks:
                    new_rank = used_rank + rank
                    if new_rank > rank_budget:
                        continue
                    edge_cost = _compiler_transition_cost(row, rank, metric)
                    step = {
                        "pass_idx": pass_idx,
                        "recipe_id": row["recipe_id"],
                        "from_layer": layer,
                        "to_layer": to_layer,
                        "rank": rank,
                        "cost": edge_cost,
                        "residual_rms_before": row["residual_rms_before"],
                        "residual_rms_after": row[f"residual_rms_after_rank_{rank}"],
                        "hidden_cosine": row[f"hidden_cosine_rank_{rank}"],
                    }
                    key = (pass_idx + 1, to_layer, new_rank)
                    value = (cost + edge_cost, [*path, step])
                    if key not in next_dp or value[0] < next_dp[key][0]:
                        next_dp[key] = value
        dp = next_dp
    candidates = [
        (cost, rank, path)
        for (used_passes, layer, rank), (cost, path) in dp.items()
        if used_passes == depth and layer == end_layer
    ]
    if not candidates:
        raise RuntimeError("no compiler partition satisfied the depth/rank constraints")
    cost, rank, path = min(candidates, key=lambda item: item[0])
    return {
        "cost_metric": metric,
        "total_cost": cost,
        "total_rank": rank,
        "rank_budget": rank_budget,
        "steps": path,
    }


def _compiler_suffix_kl_for_partition(
    dense: DenseModel,
    recursive: RecursiveModel,
    streams,
    config: ExperimentConfig,
    *,
    partition: dict[str, Any],
    rank_mats: dict[tuple[int, int, int, int], torch.Tensor],
    batches: int,
    positions_per_batch: int,
    temperature: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    totals: dict[tuple[int, int, int, int], dict[str, float]] = {}
    eval_batches = streams.eval_batches(config.training)
    for _ in range(batches):
        tokens, _ = next(eval_batches)
        tokens = tokens.to(device)
        dense_out = dense(tokens, return_states=True)
        assert dense_out.meta.states is not None
        h0 = dense_out.meta.states[config.model.n_prelude].detach()
        positions = _compiler_sample_positions(
            tokens,
            positions_per_batch=positions_per_batch,
            generator=generator,
        )
        for step in partition["steps"]:
            pass_idx = int(step["pass_idx"])
            from_layer = int(step["from_layer"])
            to_layer = int(step["to_layer"])
            rank = int(step["rank"])
            input_state = dense_out.meta.states[from_layer].detach()
            target_state = dense_out.meta.states[to_layer].detach()
            recipe_ids = torch.full(
                (tokens.shape[0],),
                int(step["recipe_id"]),
                dtype=torch.long,
                device=device,
            )
            sparse = recursive.core.forward_step(
                input_state,
                h0,
                recipe_ids,
                torch.ones(tokens.shape[0], dtype=torch.bool, device=device),
                pass_idx=pass_idx,
            )
            x = torch.cat(
                [
                    _rms_unit(input_state),
                    _rms_unit(h0),
                    torch.ones(*input_state.shape[:-1], 1, dtype=input_state.dtype, device=device),
                ],
                dim=-1,
            ).float()
            correction = x @ rank_mats[(pass_idx, from_layer, to_layer, rank)].to(device)
            approx = sparse + correction.to(sparse.dtype)
            dense_logits = _dense_suffix_logits_from_state(dense, target_state, state_layer=to_layer)
            approx_logits = _dense_suffix_logits_from_state(dense, approx, state_layer=to_layer)
            dense_logp = F.log_softmax(dense_logits.reshape(-1, dense_logits.shape[-1]).index_select(0, positions) / temperature, dim=-1)
            approx_logp = F.log_softmax(approx_logits.reshape(-1, approx_logits.shape[-1]).index_select(0, positions) / temperature, dim=-1)
            kl = (dense_logp.exp() * (dense_logp - approx_logp)).sum(dim=-1).mean() * (
                temperature * temperature
            )
            key = (pass_idx, from_layer, to_layer, rank)
            bucket = totals.setdefault(key, {"suffix_kl": 0.0, "count": 0.0})
            bucket["suffix_kl"] += float(kl.detach().cpu())
            bucket["count"] += 1.0
    results = []
    for (pass_idx, from_layer, to_layer, rank), values in totals.items():
        results.append(
            {
                "pass_idx": pass_idx,
                "from_layer": from_layer,
                "to_layer": to_layer,
                "rank": rank,
                "suffix_kl": values["suffix_kl"] / max(values["count"], 1.0),
            }
        )
    return sorted(results, key=lambda item: (item["pass_idx"], item["from_layer"], item["to_layer"]))


def cmd_analyze_operator_compiler(args: argparse.Namespace) -> None:
    config = _model_for_mode(load_config(args.config), "recursive_exact")
    if args.use_global_lowrank_corrector:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                use_global_lowrank_corrector=True,
                global_corrector_rank=args.global_corrector_rank,
            ),
        )
    ranks = _parse_rank_list(args.candidate_ranks)
    depth = int(args.max_depth)
    rank_budget = int(args.rank_budget if args.rank_budget is not None else max(ranks) * depth)
    set_seed(config.training.seed)
    device = default_device()
    streams = load_token_streams(config.data, config.training, config.model.vocab_size)
    dense = _load_dense_model(config, args.dense_checkpoint, device)
    recursive = _load_recursive_model(config, args.recursive_checkpoint, device)
    dense.eval()
    recursive.eval()

    start_layer = config.model.n_prelude
    end_layer = config.model.n_dense_layers - config.model.n_coda
    input_layers = list(range(start_layer, end_layer))
    candidate_edges = [(i, j) for i in input_layers for j in range(i + 1, end_layer + 1)]
    schedule = _schedule_for_depth(config, depth, args.recipe_schedule)
    d_model = config.model.d_model
    feature_dim = 2 * d_model + 1
    fit_dtype = torch.float32
    max_rank = min(feature_dim, d_model)

    xtx: dict[int, torch.Tensor] = {
        layer: torch.zeros(feature_dim, feature_dim, dtype=fit_dtype, device=device)
        for layer in input_layers
    }
    xty: dict[tuple[int, int, int], torch.Tensor] = {
        (pass_idx, i, j): torch.zeros(feature_dim, d_model, dtype=fit_dtype, device=device)
        for pass_idx in range(depth)
        for i, j in candidate_edges
    }
    train_batches = streams.train_batches(config.training)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed if args.seed is not None else config.training.seed)
    with torch.no_grad():
        for batch_idx in range(args.batches):
            tokens, _ = next(train_batches)
            tokens = tokens.to(device)
            dense_out = dense(tokens, return_states=True)
            assert dense_out.meta.states is not None
            h0 = dense_out.meta.states[start_layer].detach()
            positions = _compiler_sample_positions(
                tokens,
                positions_per_batch=args.positions_per_batch,
                generator=generator,
            )
            feature_by_layer: dict[int, torch.Tensor] = {}
            sparse_by_pass_layer: dict[tuple[int, int], torch.Tensor] = {}
            for i in input_layers:
                input_state = dense_out.meta.states[i].detach()
                x = _compiler_feature_matrix(input_state, h0, positions=positions).to(fit_dtype)
                feature_by_layer[i] = x
                xtx[i].add_(x.t() @ x)
            active_mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=device)
            for pass_idx in range(depth):
                recipe_ids = torch.full(
                    (tokens.shape[0],),
                    schedule[pass_idx],
                    dtype=torch.long,
                    device=device,
                )
                for i in input_layers:
                    sparse_by_pass_layer[(pass_idx, i)] = recursive.core.forward_step(
                        dense_out.meta.states[i].detach(),
                        h0,
                        recipe_ids,
                        active_mask,
                        pass_idx=pass_idx,
                    ).detach()
            for pass_idx in range(depth):
                for i, j in candidate_edges:
                    y = _flatten_sample_positions(
                        dense_out.meta.states[j].detach() - sparse_by_pass_layer[(pass_idx, i)],
                        positions=positions,
                    ).to(fit_dtype)
                    xty[(pass_idx, i, j)].add_(feature_by_layer[i].t() @ y)
            if args.progress and (batch_idx + 1) % max(1, args.progress) == 0:
                print(json.dumps({"event": "compiler_probe_fit_batch", "batch": batch_idx + 1}))

    identity_cpu = torch.eye(feature_dim, dtype=torch.float64)
    rank_mats: dict[tuple[int, int, int, int], torch.Tensor] = {}
    for pass_idx in range(depth):
        for i, j in candidate_edges:
            lhs = xtx[i].detach().cpu().to(torch.float64) + args.ridge * identity_cpu
            rhs = xty[(pass_idx, i, j)].detach().cpu().to(torch.float64)
            lhs = torch.nan_to_num(lhs)
            rhs = torch.nan_to_num(rhs)
            try:
                mat = torch.linalg.solve(lhs, rhs)
            except RuntimeError:
                mat = torch.linalg.pinv(lhs) @ rhs
            mat = mat.to(torch.float32)
            if max(ranks) > 0:
                u, s, vh = torch.linalg.svd(mat.to(torch.float64), full_matrices=False)
            for rank in ranks:
                if rank == 0:
                    rank_mats[(pass_idx, i, j, rank)] = torch.zeros_like(mat, dtype=torch.float32).cpu()
                else:
                    r = min(rank, s.shape[0])
                    rank_mats[(pass_idx, i, j, rank)] = ((u[:, :r] * s[:r]) @ vh[:r, :]).to(
                        torch.float32
                    ).cpu()

    stats: dict[tuple[int, int, int], dict[str, Any]] = {}
    for pass_idx in range(depth):
        for i, j in candidate_edges:
            bucket: dict[str, Any] = {
                "pass_idx": pass_idx,
                "recipe_id": schedule[pass_idx],
                "from_layer": i,
                "to_layer": j,
                "residual_sse_before": 0.0,
                "residual_count": 0,
            }
            for rank in ranks:
                bucket[f"residual_sse_after_rank_{rank}"] = 0.0
                bucket[f"hidden_cos_sum_rank_{rank}"] = 0.0
                bucket[f"hidden_cos_count_rank_{rank}"] = 0
            stats[(pass_idx, i, j)] = bucket

    eval_batches = streams.eval_batches(config.training)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed((args.seed if args.seed is not None else config.training.seed) + 10_000)
    with torch.no_grad():
        for batch_idx in range(args.eval_batches):
            tokens, _ = next(eval_batches)
            tokens = tokens.to(device)
            dense_out = dense(tokens, return_states=True)
            assert dense_out.meta.states is not None
            h0 = dense_out.meta.states[start_layer].detach()
            positions = _compiler_sample_positions(
                tokens,
                positions_per_batch=args.positions_per_batch,
                generator=eval_generator,
            )
            feature_by_layer = {
                i: _compiler_feature_matrix(
                    dense_out.meta.states[i].detach(),
                    h0,
                    positions=positions,
                )
                for i in input_layers
            }
            active_mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=device)
            sparse_by_pass_layer = {}
            for pass_idx in range(depth):
                recipe_ids = torch.full(
                    (tokens.shape[0],),
                    schedule[pass_idx],
                    dtype=torch.long,
                    device=device,
                )
                for i in input_layers:
                    sparse_by_pass_layer[(pass_idx, i)] = recursive.core.forward_step(
                        dense_out.meta.states[i].detach(),
                        h0,
                        recipe_ids,
                        active_mask,
                        pass_idx=pass_idx,
                    ).detach()
            for pass_idx in range(depth):
                for i, j in candidate_edges:
                    target = _flatten_sample_positions(dense_out.meta.states[j].detach(), positions=positions)
                    sparse = _flatten_sample_positions(sparse_by_pass_layer[(pass_idx, i)], positions=positions)
                    residual = target.float() - sparse.float()
                    bucket = stats[(pass_idx, i, j)]
                    bucket["residual_sse_before"] += float(residual.square().sum().detach().cpu())
                    bucket["residual_count"] += int(residual.numel())
                    x = feature_by_layer[i].to(torch.float32)
                    for rank in ranks:
                        correction = x @ rank_mats[(pass_idx, i, j, rank)].to(device)
                        approx = sparse.float() + correction
                        after = target.float() - approx
                        bucket[f"residual_sse_after_rank_{rank}"] += float(
                            after.square().sum().detach().cpu()
                        )
                        cos_values = F.cosine_similarity(approx, target.float(), dim=-1)
                        bucket[f"hidden_cos_sum_rank_{rank}"] += float(cos_values.sum().detach().cpu())
                        bucket[f"hidden_cos_count_rank_{rank}"] += int(cos_values.numel())
            if args.progress and (batch_idx + 1) % max(1, args.progress) == 0:
                print(json.dumps({"event": "compiler_probe_eval_batch", "batch": batch_idx + 1}))

    rows: list[dict[str, Any]] = []
    for key in sorted(stats):
        bucket = stats[key]
        count = max(int(bucket["residual_count"]), 1)
        row: dict[str, Any] = {
            "pass_idx": bucket["pass_idx"],
            "recipe_id": bucket["recipe_id"],
            "from_layer": bucket["from_layer"],
            "to_layer": bucket["to_layer"],
            "residual_rms_before": math.sqrt(float(bucket["residual_sse_before"]) / count),
        }
        for rank in ranks:
            row[f"residual_rms_after_rank_{rank}"] = math.sqrt(
                float(bucket[f"residual_sse_after_rank_{rank}"]) / count
            )
            row[f"hidden_cosine_rank_{rank}"] = float(
                bucket[f"hidden_cos_sum_rank_{rank}"]
            ) / max(int(bucket[f"hidden_cos_count_rank_{rank}"]), 1)
        rows.append(row)

    partition = _choose_compiler_partition(
        rows,
        start_layer=start_layer,
        end_layer=end_layer,
        depth=depth,
        ranks=ranks,
        rank_budget=rank_budget,
        metric=args.dp_metric,
    )
    suffix = []
    if args.suffix_kl_batches > 0:
        suffix = _compiler_suffix_kl_for_partition(
            dense,
            recursive,
            streams,
            config,
            partition=partition,
            rank_mats=rank_mats,
            batches=args.suffix_kl_batches,
            positions_per_batch=args.positions_per_batch,
            temperature=args.temperature,
            device=device,
            seed=(args.seed if args.seed is not None else config.training.seed) + 20_000,
        )
        suffix_by_key = {
            (item["pass_idx"], item["from_layer"], item["to_layer"], item["rank"]): item["suffix_kl"]
            for item in suffix
        }
        for step in partition["steps"]:
            step["suffix_kl"] = suffix_by_key.get(
                (step["pass_idx"], step["from_layer"], step["to_layer"], step["rank"])
            )

    report = {
        "start_layer": start_layer,
        "end_layer": end_layer,
        "depth": depth,
        "recipe_schedule": schedule,
        "candidate_ranks": ranks,
        "effective_max_rank": max_rank,
        "rank_budget": rank_budget,
        "ridge": args.ridge,
        "fit_batches": args.batches,
        "eval_batches": args.eval_batches,
        "positions_per_batch": args.positions_per_batch,
        "candidate_transition_table": rows,
        "chosen_partition": partition,
        "chosen_partition_suffix_kl": suffix,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    _print_json(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rte")
    sub = parser.add_subparsers(required=True)
    fairness = sub.add_parser("fairness")
    fairness.add_argument("--config", required=True)
    fairness.add_argument("--max-banks", type=int, default=128)
    fairness.add_argument("--strict", action="store_true")
    fairness.set_defaults(func=cmd_fairness)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--mode", choices=TRAIN_MODES)
    train.add_argument("--steps", type=int, default=1)
    train.add_argument("--save-checkpoint")
    train.add_argument("--resume")
    train.add_argument("--run-dir")
    train.add_argument("--teacher-checkpoint")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--mode", choices=TRAIN_MODES)
    evaluate.add_argument("--checkpoint")
    evaluate.set_defaults(func=cmd_evaluate)

    transplant = sub.add_parser("transplant-dense-to-recursive")
    transplant.add_argument("--dense-checkpoint", required=True)
    transplant.add_argument("--recursive-config", required=True)
    transplant.add_argument("--output", required=True)
    transplant.add_argument("--use-global-lowrank-corrector", action="store_true")
    transplant.add_argument("--global-corrector-rank", type=int, default=16)
    transplant.set_defaults(func=cmd_transplant_dense_to_recursive)

    clone = sub.add_parser("operator-clone")
    clone.add_argument("--dense-checkpoint", required=True)
    clone.add_argument("--recursive-checkpoint", required=True)
    clone.add_argument("--config", required=True)
    clone.add_argument("--steps", type=int, default=300)
    clone.add_argument("--teacher-forcing", type=float, default=1.0)
    clone.add_argument("--train", default="core,pass_film,global_lowrank")
    clone.add_argument("--run-dir", required=True)
    clone.add_argument("--save-checkpoint")
    clone.add_argument("--depth", type=int, default=4)
    clone.add_argument("--dense-layer-map", type=int, nargs="+")
    clone.add_argument("--recipe-schedule", type=int, nargs="+")
    clone.add_argument("--lr", type=float, default=0.001)
    clone.add_argument("--weight-decay", type=float, default=0.0)
    clone.add_argument("--lambda-hidden", type=float, default=1.0)
    clone.add_argument("--lambda-delta", type=float, default=0.5)
    clone.add_argument("--lambda-cos", type=float, default=0.5)
    clone.add_argument("--grad-clip-norm", type=float, default=1.0)
    clone.add_argument("--log-every", type=int, default=10)
    clone.add_argument("--use-global-lowrank-corrector", action="store_true")
    clone.add_argument("--global-corrector-rank", type=int, default=16)
    clone.set_defaults(func=cmd_operator_clone)

    rollout = sub.add_parser("evaluate-rollout-similarity")
    rollout.add_argument("--dense-checkpoint", required=True)
    rollout.add_argument("--recursive-checkpoint", required=True)
    rollout.add_argument("--config", required=True)
    rollout.add_argument("--depth", type=int, default=4)
    rollout.add_argument("--dense-layer-map", type=int, nargs="+")
    rollout.add_argument("--recipe-schedule", type=int, nargs="+")
    rollout.add_argument("--num-batches", type=int, default=8)
    rollout.add_argument("--temperature", type=float, default=2.0)
    rollout.add_argument("--use-global-lowrank-corrector", action="store_true")
    rollout.add_argument("--global-corrector-rank", type=int, default=16)
    rollout.set_defaults(func=cmd_evaluate_rollout_similarity)

    compiler = sub.add_parser("analyze-operator-compiler")
    compiler.add_argument("--dense-checkpoint", required=True)
    compiler.add_argument("--recursive-checkpoint", required=True)
    compiler.add_argument("--config", required=True)
    compiler.add_argument("--batches", type=int, default=32)
    compiler.add_argument("--eval-batches", type=int, default=8)
    compiler.add_argument("--positions-per-batch", type=int, default=2048)
    compiler.add_argument("--candidate-ranks", default="0,4,8,16,32,64")
    compiler.add_argument("--max-depth", type=int, default=4)
    compiler.add_argument("--rank-budget", type=int)
    compiler.add_argument("--recipe-schedule", type=int, nargs="+")
    compiler.add_argument("--ridge", type=float, default=1e-3)
    compiler.add_argument("--dp-metric", choices=["residual_rms", "hidden_cosine"], default="residual_rms")
    compiler.add_argument("--suffix-kl-batches", type=int, default=1)
    compiler.add_argument("--temperature", type=float, default=2.0)
    compiler.add_argument("--use-global-lowrank-corrector", action="store_true")
    compiler.add_argument("--global-corrector-rank", type=int, default=16)
    compiler.add_argument("--seed", type=int)
    compiler.add_argument("--progress", type=int, default=0)
    compiler.add_argument("--output")
    compiler.set_defaults(func=cmd_analyze_operator_compiler)

    route_oracle = sub.add_parser("route-oracle")
    route_oracle.add_argument("--config", required=True)
    route_oracle.add_argument("--checkpoint")
    route_oracle.add_argument("--candidate-counts", type=int, nargs="+", default=[4, 8, 16])
    route_oracle.add_argument("--num-batches", type=int, default=1)
    route_oracle.add_argument("--batch-size", type=int)
    route_oracle.add_argument("--depth", type=int)
    route_oracle.add_argument("--proposal", choices=["random", "router"], default="random")
    route_oracle.add_argument("--allow-dense-fallback", action="store_true")
    route_oracle.add_argument("--include-static", action=argparse.BooleanOptionalAction, default=True)
    route_oracle.add_argument("--seed", type=int)
    route_oracle.set_defaults(func=cmd_route_oracle)

    ffn_regroup = sub.add_parser("ffn-regroup-oracle")
    ffn_regroup.add_argument("--config", required=True)
    ffn_regroup.add_argument("--dense-checkpoint", required=True)
    ffn_regroup.add_argument("--profile-batches", type=int, default=8)
    ffn_regroup.add_argument("--eval-batches", type=int, default=8)
    ffn_regroup.add_argument("--topk", type=int, nargs="+")
    ffn_regroup.add_argument("--cluster-iters", type=int, default=25)
    ffn_regroup.add_argument("--batch-size", type=int)
    ffn_regroup.add_argument("--seed", type=int)
    ffn_regroup.add_argument("--progress", type=int, default=0)
    ffn_regroup.add_argument("--include-permutations", action="store_true")
    ffn_regroup.add_argument("--output")
    ffn_regroup.set_defaults(func=cmd_ffn_regroup_oracle)

    head_regroup = sub.add_parser("head-regroup-oracle")
    head_regroup.add_argument("--config", required=True)
    head_regroup.add_argument("--dense-checkpoint", required=True)
    head_regroup.add_argument("--profile-batches", type=int, default=8)
    head_regroup.add_argument("--eval-batches", type=int, default=8)
    head_regroup.add_argument("--topk", type=int, nargs="+")
    head_regroup.add_argument("--cluster-iters", type=int, default=25)
    head_regroup.add_argument("--batch-size", type=int)
    head_regroup.add_argument("--seed", type=int)
    head_regroup.add_argument("--progress", type=int, default=0)
    head_regroup.add_argument("--include-permutations", action="store_true")
    head_regroup.add_argument("--output")
    head_regroup.set_defaults(func=cmd_head_regroup_oracle)

    deferred_block = sub.add_parser("deferred-grouped-block-oracle")
    deferred_block.add_argument("--config", required=True)
    deferred_block.add_argument("--dense-checkpoint", required=True)
    deferred_block.add_argument("--layer", type=int, default=2)
    deferred_block.add_argument("--num-batches", type=int, default=1)
    deferred_block.add_argument("--batch-size", type=int)
    deferred_block.add_argument("--topk", type=int, nargs="+")
    deferred_block.add_argument("--temperature", type=float, default=2.0)
    deferred_block.add_argument("--seed", type=int)
    deferred_block.add_argument("--include-per-batch", action="store_true")
    deferred_block.add_argument("--output")
    deferred_block.set_defaults(func=cmd_deferred_grouped_block_oracle)

    deferred_full_stack = sub.add_parser("deferred-grouped-full-stack-oracle")
    deferred_full_stack.add_argument("--config", required=True)
    deferred_full_stack.add_argument("--dense-checkpoint", required=True)
    deferred_full_stack.add_argument("--eval-batches", type=int)
    deferred_full_stack.add_argument("--batch-size", type=int)
    deferred_full_stack.add_argument("--topk", type=int, nargs="+")
    deferred_full_stack.add_argument(
        "--selectors",
        nargs="+",
        choices=["norm", "omp_unit", "omp_ls"],
        default=["norm", "omp_unit", "omp_ls"],
    )
    deferred_full_stack.add_argument("--temperature", type=float, default=2.0)
    deferred_full_stack.add_argument("--ridge", type=float, default=1e-4)
    deferred_full_stack.add_argument("--seed", type=int)
    deferred_full_stack.add_argument("--progress", type=int, default=0)
    deferred_full_stack.add_argument("--output")
    deferred_full_stack.set_defaults(func=cmd_deferred_grouped_full_stack_oracle)

    deferred_neuron = sub.add_parser("deferred-neuron-full-stack-oracle")
    deferred_neuron.add_argument("--config", required=True)
    deferred_neuron.add_argument("--dense-checkpoint", required=True)
    deferred_neuron.add_argument("--eval-batches", type=int)
    deferred_neuron.add_argument("--batch-size", type=int)
    deferred_neuron.add_argument("--k", type=int, nargs="+")
    deferred_neuron.add_argument(
        "--selectors",
        nargs="+",
        choices=["norm", "omp_unit", "omp_ls"],
        default=["norm", "omp_unit", "omp_ls"],
    )
    deferred_neuron.add_argument("--temperature", type=float, default=2.0)
    deferred_neuron.add_argument("--ridge", type=float, default=1e-4)
    deferred_neuron.add_argument("--seed", type=int)
    deferred_neuron.add_argument("--progress", type=int, default=0)
    deferred_neuron.add_argument("--output")
    deferred_neuron.set_defaults(func=cmd_deferred_neuron_full_stack_oracle)

    neuron_selector = sub.add_parser("deferred-neuron-selector-oracle")
    neuron_selector.add_argument("--config", required=True)
    neuron_selector.add_argument("--dense-checkpoint", required=True)
    neuron_selector.add_argument("--train-batches", type=int, default=64)
    neuron_selector.add_argument("--selector-checkpoint")
    neuron_selector.add_argument("--save-selector")
    neuron_selector.add_argument("--eval-batches", type=int)
    neuron_selector.add_argument("--batch-size", type=int)
    neuron_selector.add_argument("--k", type=int, nargs="+")
    neuron_selector.add_argument("--candidate-m", type=int, nargs="+")
    neuron_selector.add_argument("--selector-rank", type=int, default=32)
    neuron_selector.add_argument("--label-k", type=int)
    neuron_selector.add_argument("--selector-train-tokens", type=int, default=2048)
    neuron_selector.add_argument(
        "--rerankers",
        nargs="+",
        choices=["norm", "omp_unit", "omp_ridge1", "omp_ls"],
        default=["norm"],
    )
    neuron_selector.add_argument("--rerank-ridge", type=float, nargs="+")
    neuron_selector.add_argument("--rerank-clamp", nargs="+")
    neuron_selector.add_argument("--selector-lr", type=float, default=0.003)
    neuron_selector.add_argument("--selector-weight-decay", type=float, default=0.01)
    neuron_selector.add_argument("--temperature", type=float, default=2.0)
    neuron_selector.add_argument("--ridge", type=float, default=1e-4)
    neuron_selector.add_argument("--seed", type=int)
    neuron_selector.add_argument("--train-progress", type=int, default=0)
    neuron_selector.add_argument("--progress", type=int, default=0)
    neuron_selector.add_argument("--output")
    neuron_selector.set_defaults(func=cmd_deferred_neuron_selector_oracle)

    factor_union = sub.add_parser("deferred-neuron-factor-union-oracle")
    factor_union.add_argument("--config", required=True)
    factor_union.add_argument("--dense-checkpoint", required=True)
    factor_union.add_argument("--eval-batches", type=int)
    factor_union.add_argument("--batch-size", type=int)
    factor_union.add_argument("--k", type=int, nargs="+")
    factor_union.add_argument("--factor-m", type=int, default=64)
    factor_union.add_argument("--product-factor-m", type=int, default=48)
    factor_union.add_argument(
        "--rerankers",
        nargs="+",
        choices=["norm", "omp_unit"],
        default=["norm", "omp_unit"],
    )
    factor_union.add_argument("--temperature", type=float, default=2.0)
    factor_union.add_argument("--ridge", type=float, default=1e-4)
    factor_union.add_argument("--seed", type=int)
    factor_union.add_argument("--progress", type=int, default=0)
    factor_union.add_argument("--output")
    factor_union.set_defaults(func=cmd_deferred_neuron_factor_union_oracle)

    factor_selector = sub.add_parser("deferred-neuron-factor-selector-oracle")
    factor_selector.add_argument("--config", required=True)
    factor_selector.add_argument("--dense-checkpoint", required=True)
    factor_selector.add_argument("--train-batches", type=int, default=64)
    factor_selector.add_argument("--selector-checkpoint")
    factor_selector.add_argument("--save-selector")
    factor_selector.add_argument("--eval-batches", type=int)
    factor_selector.add_argument("--batch-size", type=int)
    factor_selector.add_argument("--k", type=int, nargs="+")
    factor_selector.add_argument("--factor-m", type=int, default=64)
    factor_selector.add_argument("--product-factor-m", type=int, default=48)
    factor_selector.add_argument("--selector-rank", type=int, default=32)
    factor_selector.add_argument("--selector-train-tokens", type=int, default=1024)
    factor_selector.add_argument("--selector-lr", type=float, default=0.003)
    factor_selector.add_argument("--selector-weight-decay", type=float, default=0.01)
    factor_selector.add_argument(
        "--rerankers",
        nargs="+",
        choices=["norm", "omp_unit"],
        default=["norm"],
    )
    factor_selector.add_argument("--temperature", type=float, default=2.0)
    factor_selector.add_argument("--ridge", type=float, default=1e-4)
    factor_selector.add_argument("--seed", type=int)
    factor_selector.add_argument("--train-progress", type=int, default=0)
    factor_selector.add_argument("--progress", type=int, default=0)
    factor_selector.add_argument("--output")
    factor_selector.set_defaults(func=cmd_deferred_neuron_factor_selector_oracle)

    svd_factor = sub.add_parser("deferred-neuron-svd-factor-union-oracle")
    svd_factor.add_argument("--config", required=True)
    svd_factor.add_argument("--dense-checkpoint", required=True)
    svd_factor.add_argument("--eval-batches", type=int)
    svd_factor.add_argument("--batch-size", type=int)
    svd_factor.add_argument("--k", type=int, nargs="+")
    svd_factor.add_argument("--ranks", type=int, nargs="+", default=[64])
    svd_factor.add_argument("--factor-m", type=int, nargs="+", default=[64])
    svd_factor.add_argument("--product-factor-m", type=int, default=0)
    svd_factor.add_argument(
        "--rerankers",
        nargs="+",
        choices=["norm", "omp_unit"],
        default=["norm"],
    )
    svd_factor.add_argument("--temperature", type=float, default=2.0)
    svd_factor.add_argument("--ridge", type=float, default=1e-4)
    svd_factor.add_argument("--seed", type=int)
    svd_factor.add_argument("--progress", type=int, default=0)
    svd_factor.add_argument("--output")
    svd_factor.set_defaults(func=cmd_deferred_neuron_svd_factor_union_oracle)

    cluster_pool = sub.add_parser("deferred-neuron-cluster-pool-oracle")
    cluster_pool.add_argument("--config", required=True)
    cluster_pool.add_argument("--dense-checkpoint", required=True)
    cluster_pool.add_argument("--eval-batches", type=int)
    cluster_pool.add_argument("--batch-size", type=int)
    cluster_pool.add_argument("--ranks", type=int, nargs="+", default=[64])
    cluster_pool.add_argument("--clusters", type=int, nargs="+", default=[8, 16, 32, 64])
    cluster_pool.add_argument("--candidate-m", type=int, nargs="+", default=[128, 192, 256])
    cluster_pool.add_argument("--reference-k", type=int, default=64)
    cluster_pool.add_argument(
        "--score-modes",
        nargs="+",
        choices=["sum", "upgate", "product"],
        default=["sum"],
    )
    cluster_pool.add_argument(
        "--aggregations",
        nargs="+",
        choices=["mean", "max"],
        default=["mean"],
    )
    cluster_pool.add_argument("--cluster-iters", type=int, default=8)
    cluster_pool.add_argument("--profile", action="store_true")
    cluster_pool.add_argument("--temperature", type=float, default=2.0)
    cluster_pool.add_argument("--seed", type=int)
    cluster_pool.add_argument("--progress", type=int, default=0)
    cluster_pool.add_argument("--output")
    cluster_pool.set_defaults(func=cmd_deferred_neuron_cluster_pool_oracle)

    static_cluster_pool = sub.add_parser("deferred-neuron-static-cluster-pool-oracle")
    static_cluster_pool.add_argument("--config", required=True)
    static_cluster_pool.add_argument("--dense-checkpoint", required=True)
    static_cluster_pool.add_argument("--eval-batches", type=int)
    static_cluster_pool.add_argument("--batch-size", type=int)
    static_cluster_pool.add_argument("--calibration-tokens", type=int, nargs="+", default=[8192])
    static_cluster_pool.add_argument("--ranks", type=int, nargs="+", default=[64])
    static_cluster_pool.add_argument("--clusters", type=int, nargs="+", default=[8, 16])
    static_cluster_pool.add_argument("--candidate-m", type=int, nargs="+", default=[192])
    static_cluster_pool.add_argument("--reference-k", type=int, default=64)
    static_cluster_pool.add_argument(
        "--score-modes",
        nargs="+",
        choices=["sum", "upgate", "product"],
        default=["sum"],
    )
    static_cluster_pool.add_argument(
        "--aggregations",
        nargs="+",
        choices=["mean", "max"],
        default=["mean"],
    )
    static_cluster_pool.add_argument("--cluster-iters", type=int, default=8)
    static_cluster_pool.add_argument("--profile", action="store_true")
    static_cluster_pool.add_argument("--temperature", type=float, default=2.0)
    static_cluster_pool.add_argument("--seed", type=int)
    static_cluster_pool.add_argument("--progress", type=int, default=0)
    static_cluster_pool.add_argument("--output")
    static_cluster_pool.set_defaults(func=cmd_deferred_neuron_static_cluster_pool_oracle)

    union_eval = sub.add_parser("static-cluster-pool-union-eval")
    union_eval.add_argument("--config", required=True)
    union_eval.add_argument("--dense-checkpoint", required=True)
    union_eval.add_argument("--eval-batches", type=int)
    union_eval.add_argument("--batch-size", type=int)
    union_eval.add_argument("--calibration-tokens", type=int, default=8192)
    union_eval.add_argument("--rank", type=int, default=64)
    union_eval.add_argument("--clusters", type=int, default=16)
    union_eval.add_argument("--candidate-m", type=int, default=192)
    union_eval.add_argument(
        "--union-caps",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional active-neuron caps for additional global-union variants. "
            "These change the sparse active set and are quality experiments."
        ),
    )
    union_eval.add_argument(
        "--union-layer-caps",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional per-layer active-neuron caps for one additional global-union variant. "
            "Length must match the dense layer count; this is a quality-changing experiment."
        ),
    )
    union_eval.add_argument("--reference-k", type=int, default=64)
    union_eval.add_argument(
        "--score-mode",
        choices=["sum", "upgate", "product"],
        default="sum",
    )
    union_eval.add_argument(
        "--aggregation",
        choices=["mean", "max"],
        default="mean",
    )
    union_eval.add_argument("--cluster-iters", type=int, default=4)
    union_eval.add_argument("--temperature", type=float, default=2.0)
    union_eval.add_argument("--seed", type=int)
    union_eval.add_argument("--progress", type=int, default=0)
    union_eval.add_argument("--output")
    union_eval.set_defaults(func=cmd_static_cluster_pool_union_eval)

    staleness = sub.add_parser("static-cluster-pool-staleness")
    staleness.add_argument("--config", required=True)
    staleness.add_argument("--dense-checkpoint", required=True)
    staleness.add_argument("--eval-batches", type=int)
    staleness.add_argument("--batch-size", type=int)
    staleness.add_argument("--calibration-tokens", type=int, default=8192)
    staleness.add_argument("--rank", type=int, default=64)
    staleness.add_argument("--clusters", type=int, default=16)
    staleness.add_argument("--candidate-m", type=int, default=192)
    staleness.add_argument("--reference-k", type=int, default=64)
    staleness.add_argument("--perturb-pct", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0])
    staleness.add_argument("--score-mode", choices=["sum", "upgate", "product"], default="sum")
    staleness.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    staleness.add_argument("--cluster-iters", type=int, default=4)
    staleness.add_argument("--temperature", type=float, default=2.0)
    staleness.add_argument("--seed", type=int)
    staleness.add_argument("--progress", type=int, default=0)
    staleness.add_argument("--output")
    staleness.set_defaults(func=cmd_static_cluster_pool_staleness)

    continuation = sub.add_parser("static-cluster-pool-continuation")
    continuation.add_argument("--config", required=True)
    continuation.add_argument("--dense-checkpoint", required=True)
    continuation.add_argument("--steps", type=int, default=100)
    continuation.add_argument("--eval-batches", type=int)
    continuation.add_argument("--eval-steps", type=int, nargs="+")
    continuation.add_argument("--batch-size", type=int)
    continuation.add_argument("--seq-len", type=int)
    continuation.add_argument("--calibration-tokens", type=int, default=8192)
    continuation.add_argument("--rank", type=int, default=64)
    continuation.add_argument("--clusters", type=int, default=16)
    continuation.add_argument("--candidate-m", type=int, default=192)
    continuation.add_argument(
        "--sparse-ffn-kind",
        choices=["static_cluster", "active_union_indexed", "active_union_packed"],
        default="static_cluster",
        help=(
            "Sparse FFN implementation for the sparse continuation branch. "
            "active_union_packed uses packed trainable active weights and no dense-master reconcile."
        ),
    )
    continuation.add_argument(
        "--active-union-cap",
        type=int,
        help="Optional same active-neuron cap for active-union continuation layers.",
    )
    continuation.add_argument(
        "--active-union-layer-caps",
        type=int,
        nargs="*",
        default=[],
        help="Optional per-layer active-neuron caps for active-union continuation.",
    )
    continuation.add_argument("--refresh-intervals", type=int, nargs="+", default=[0])
    continuation.add_argument("--score-mode", choices=["sum", "upgate", "product"], default="sum")
    continuation.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    continuation.add_argument("--cluster-iters", type=int, default=4)
    continuation.add_argument("--lr", type=float)
    continuation.add_argument("--weight-decay", type=float)
    continuation.add_argument("--grad-clip-norm", type=float)
    continuation.add_argument("--run-dense", action=argparse.BooleanOptionalAction, default=True)
    continuation.add_argument("--run-sparse", action=argparse.BooleanOptionalAction, default=True)
    continuation.add_argument("--resume-optimizer-state", action="store_true")
    continuation.add_argument("--include-gradient-alignment", action="store_true")
    continuation.add_argument("--alignment-batches", type=int, default=1)
    continuation.add_argument("--alignment-batch-size", type=int)
    continuation.add_argument("--seed", type=int)
    continuation.add_argument("--progress", type=int, default=0)
    continuation.add_argument("--output")
    continuation.set_defaults(func=cmd_static_cluster_pool_continuation)

    grad_align = sub.add_parser("static-cluster-pool-gradient-alignment")
    grad_align.add_argument("--config", required=True)
    grad_align.add_argument("--dense-checkpoint", required=True)
    grad_align.add_argument("--eval-batches", type=int, default=1)
    grad_align.add_argument("--batch-size", type=int)
    grad_align.add_argument("--calibration-tokens", type=int, default=8192)
    grad_align.add_argument("--rank", type=int, default=64)
    grad_align.add_argument("--clusters", type=int, default=16)
    grad_align.add_argument("--candidate-m", type=int, default=192)
    grad_align.add_argument("--score-mode", choices=["sum", "upgate", "product"], default="sum")
    grad_align.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    grad_align.add_argument("--cluster-iters", type=int, default=4)
    grad_align.add_argument("--seed", type=int)
    grad_align.add_argument("--progress", type=int, default=0)
    grad_align.add_argument("--output")
    grad_align.set_defaults(func=cmd_static_cluster_pool_gradient_alignment)

    svd_hot = sub.add_parser("deferred-neuron-svd-hot-eval")
    svd_hot.add_argument("--config", required=True)
    svd_hot.add_argument("--dense-checkpoint", required=True)
    svd_hot.add_argument("--eval-batches", type=int)
    svd_hot.add_argument("--batch-size", type=int)
    svd_hot.add_argument("--k", type=int, default=64)
    svd_hot.add_argument("--rank", type=int, default=48)
    svd_hot.add_argument("--factor-m", type=int, default=64)
    svd_hot.add_argument("--product-factor-m", type=int, default=0)
    svd_hot.add_argument("--candidate-mode", choices=["mask", "slots", "triton"], default="mask")
    svd_hot.add_argument("--profile", action="store_true")
    svd_hot.add_argument("--temperature", type=float, default=2.0)
    svd_hot.add_argument("--seed", type=int)
    svd_hot.add_argument("--progress", type=int, default=0)
    svd_hot.add_argument("--output")
    svd_hot.set_defaults(func=cmd_deferred_neuron_svd_hot_eval)

    svd_bench = sub.add_parser("benchmark-svd-sparse-ffn")
    svd_bench.add_argument(
        "--size",
        type=_parse_ffn_bench_size,
        action="append",
        default=None,
        help="Benchmark size as d_modelxd_ffxtokens, e.g. 512x2048x128.",
    )
    svd_bench.add_argument("--rank", type=int, default=48)
    svd_bench.add_argument("--factor-m", type=int, default=64)
    svd_bench.add_argument("--product-factor-m", type=int, default=64)
    svd_bench.add_argument("--candidate-mode", choices=["mask", "slots", "triton"], default="mask")
    svd_bench.add_argument("--k", type=int, default=64)
    svd_bench.add_argument("--iters", type=int, default=5)
    svd_bench.add_argument("--warmup", type=int, default=1)
    svd_bench.add_argument("--profile", action="store_true")
    svd_bench.add_argument("--output")
    svd_bench.set_defaults(func=cmd_benchmark_svd_sparse_ffn)

    cluster_pool_bench = sub.add_parser("benchmark-cluster-pool-ffn")
    cluster_pool_bench.add_argument(
        "--size",
        type=_parse_ffn_bench_size,
        action="append",
        default=None,
        help="Benchmark size as d_modelxd_ffxtokens, e.g. 2048x8192x4096.",
    )
    cluster_pool_bench.add_argument("--rank", type=int, default=64)
    cluster_pool_bench.add_argument("--clusters", type=int, nargs="+", default=[8, 16, 32])
    cluster_pool_bench.add_argument("--candidate-m", type=int, nargs="+", default=[96, 128, 192])
    cluster_pool_bench.add_argument("--capacity-factor", type=float, default=1.25)
    cluster_pool_bench.add_argument("--max-tokens-per-cluster", type=int)
    cluster_pool_bench.add_argument("--block-d", type=int, default=64)
    cluster_pool_bench.add_argument("--iters", type=int, default=10)
    cluster_pool_bench.add_argument("--warmup", type=int, default=3)
    cluster_pool_bench.add_argument("--output")
    cluster_pool_bench.set_defaults(func=cmd_benchmark_cluster_pool_ffn)

    cluster_pool_train_bench = sub.add_parser("benchmark-static-cluster-pool-ffn-train")
    cluster_pool_train_bench.add_argument(
        "--size",
        type=_parse_ffn_bench_size,
        action="append",
        default=None,
        help="Benchmark size as d_modelxd_ffxtokens, e.g. 2048x8192x1024.",
    )
    cluster_pool_train_bench.add_argument("--clusters", type=int, default=16)
    cluster_pool_train_bench.add_argument("--candidate-m", type=int, default=192)
    cluster_pool_train_bench.add_argument("--iters", type=int, default=5)
    cluster_pool_train_bench.add_argument("--warmup", type=int, default=1)
    cluster_pool_train_bench.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    cluster_pool_train_bench.add_argument("--cuda-graph-warmup", type=int, default=3)
    cluster_pool_train_bench.add_argument("--seed", type=int, default=1234)
    cluster_pool_train_bench.add_argument("--grad-check", action=argparse.BooleanOptionalAction, default=True)
    cluster_pool_train_bench.add_argument("--grad-check-d-model", type=int, default=64)
    cluster_pool_train_bench.add_argument("--grad-check-d-ff", type=int, default=128)
    cluster_pool_train_bench.add_argument("--grad-check-tokens", type=int, default=32)
    cluster_pool_train_bench.add_argument("--output")
    cluster_pool_train_bench.set_defaults(func=cmd_benchmark_static_cluster_pool_ffn_train)

    active_union_train_bench = sub.add_parser("benchmark-active-union-ffn-train")
    active_union_train_bench.add_argument(
        "--size",
        type=_parse_ffn_bench_size,
        action="append",
        default=None,
        help="Benchmark size as d_modelxd_ffxtokens, e.g. 2048x8192x256.",
    )
    active_union_train_bench.add_argument("--active-m", type=int, nargs="+", default=[320])
    active_union_train_bench.add_argument("--iters", type=int, default=5)
    active_union_train_bench.add_argument("--warmup", type=int, default=1)
    active_union_train_bench.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    active_union_train_bench.add_argument("--cuda-graph-warmup", type=int, default=3)
    active_union_train_bench.add_argument(
        "--triton-swiglu-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Triton packed SwiGLU custom autograd path for dense fused WUG "
            "and active-union packed/indexed WUG variants when CUDA is available."
        ),
    )
    active_union_train_bench.add_argument("--seed", type=int, default=1234)
    active_union_train_bench.add_argument("--output")
    active_union_train_bench.set_defaults(func=cmd_benchmark_active_union_ffn_train)

    active_union_model_bench = sub.add_parser("benchmark-active-union-model-train-step")
    active_union_model_bench.add_argument("--config", required=True)
    active_union_model_bench.add_argument("--dense-checkpoint")
    active_union_model_bench.add_argument("--random-init", action="store_true")
    active_union_model_bench.add_argument("--d-model", type=int)
    active_union_model_bench.add_argument("--d-ff", type=int)
    active_union_model_bench.add_argument("--layers", type=int)
    active_union_model_bench.add_argument("--heads", type=int)
    active_union_model_bench.add_argument("--vocab-size", type=int)
    active_union_model_bench.add_argument("--active-union-cap", type=int, nargs="+", default=[0, 320])
    active_union_model_bench.add_argument("--batch-size", type=int)
    active_union_model_bench.add_argument("--seq-len", type=int)
    active_union_model_bench.add_argument("--calibration-tokens", type=int, default=8192)
    active_union_model_bench.add_argument("--rank", type=int, default=64)
    active_union_model_bench.add_argument("--clusters", type=int, default=16)
    active_union_model_bench.add_argument("--candidate-m", type=int, default=192)
    active_union_model_bench.add_argument("--score-mode", choices=["sum", "upgate", "product"], default="sum")
    active_union_model_bench.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    active_union_model_bench.add_argument("--cluster-iters", type=int, default=4)
    active_union_model_bench.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    active_union_model_bench.add_argument("--lr", type=float)
    active_union_model_bench.add_argument("--weight-decay", type=float)
    active_union_model_bench.add_argument("--iters", type=int, default=5)
    active_union_model_bench.add_argument("--warmup", type=int, default=1)
    active_union_model_bench.add_argument("--component-iters", type=int, default=3)
    active_union_model_bench.add_argument("--component-warmup", type=int, default=1)
    active_union_model_bench.add_argument("--event-profile", action=argparse.BooleanOptionalAction, default=True)
    active_union_model_bench.add_argument("--event-iters", type=int, default=3)
    active_union_model_bench.add_argument("--event-warmup", type=int, default=1)
    active_union_model_bench.add_argument("--pack-dense-ffn", action=argparse.BooleanOptionalAction, default=True)
    active_union_model_bench.add_argument(
        "--triton-swiglu-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the same Triton fused SwiGLU activation/backward helper to packed dense FFNs and packed active-union FFNs.",
    )
    active_union_model_bench.add_argument("--seed", type=int)
    active_union_model_bench.add_argument("--output")
    active_union_model_bench.set_defaults(func=cmd_benchmark_active_union_model_train_step)

    active_union_block_bench = sub.add_parser("benchmark-active-union-block-train-step")
    active_union_block_bench.add_argument("--config", required=True)
    active_union_block_bench.add_argument("--dense-checkpoint", required=True)
    active_union_block_bench.add_argument("--active-union-cap", type=int, default=192)
    active_union_block_bench.add_argument("--layers", type=int, nargs="+", default=[0])
    active_union_block_bench.add_argument("--batch-size", type=int)
    active_union_block_bench.add_argument("--seq-len", type=int)
    active_union_block_bench.add_argument("--calibration-tokens", type=int, default=8192)
    active_union_block_bench.add_argument("--rank", type=int, default=64)
    active_union_block_bench.add_argument("--clusters", type=int, default=16)
    active_union_block_bench.add_argument("--candidate-m", type=int, default=192)
    active_union_block_bench.add_argument("--score-mode", choices=["sum", "upgate", "product"], default="sum")
    active_union_block_bench.add_argument("--aggregation", choices=["mean", "max"], default="mean")
    active_union_block_bench.add_argument("--cluster-iters", type=int, default=4)
    active_union_block_bench.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    active_union_block_bench.add_argument("--iters", type=int, default=10)
    active_union_block_bench.add_argument("--warmup", type=int, default=3)
    active_union_block_bench.add_argument(
        "--triton-swiglu-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    active_union_block_bench.add_argument("--seed", type=int)
    active_union_block_bench.add_argument("--output")
    active_union_block_bench.set_defaults(func=cmd_benchmark_active_union_block_train_step)

    mlx_svd_bench = sub.add_parser("benchmark-mlx-svd-sparse-ffn")
    mlx_svd_bench.add_argument(
        "--size",
        type=_parse_ffn_bench_size,
        action="append",
        default=None,
        help="Benchmark size as d_modelxd_ffxtokens, e.g. 512x2048x128.",
    )
    mlx_svd_bench.add_argument("--rank", type=int, default=48)
    mlx_svd_bench.add_argument("--factor-m", type=int, default=64)
    mlx_svd_bench.add_argument("--product-factor-m", type=int, default=64)
    mlx_svd_bench.add_argument("--k", type=int, default=64)
    mlx_svd_bench.add_argument(
        "--backend",
        choices=["graph", "metal", "hybrid", "parallel"],
        default="graph",
        help=(
            "MLX sparse FFN backend. 'graph' uses MLX ops, 'metal' is the serial "
            "whole-path custom kernel, and 'hybrid'/'parallel' uses MLX selector "
            "slots plus a parallel Metal candidate/downsum kernel."
        ),
    )
    mlx_svd_bench.add_argument("--iters", type=int, default=5)
    mlx_svd_bench.add_argument("--warmup", type=int, default=1)
    mlx_svd_bench.add_argument("--seed", type=int, default=0)
    mlx_svd_bench.add_argument("--output")
    mlx_svd_bench.set_defaults(func=cmd_benchmark_mlx_svd_sparse_ffn)

    bench = sub.add_parser("benchmark-kernels")
    bench.add_argument("--config", required=True)
    bench.add_argument("--iters", type=int, default=10)
    bench.set_defaults(func=cmd_benchmark_kernels)

    teacher = sub.add_parser("train-macro-teacher")
    teacher.add_argument("--config", required=True)
    teacher.add_argument("--steps", type=int, default=100)
    teacher.add_argument("--teacher-checkpoint")
    teacher.add_argument("--teacher-mode", choices=["exact", "deferred_grouped"], default="exact")
    teacher.add_argument("--allow-random-teacher", action="store_true")
    teacher.add_argument("--boundary-cache")
    teacher.add_argument("--run-dir")
    teacher.add_argument("--fixed-recipe", type=int)
    teacher.add_argument("--fixed-depth", type=int)
    teacher.add_argument("--lr", type=float)
    teacher.add_argument("--lambda-hid", type=float, default=1.0)
    teacher.add_argument("--lambda-cos", type=float, default=0.5)
    teacher.add_argument("--lambda-kl", type=float, default=0.25)
    teacher.add_argument("--lambda-norm", type=float, default=0.05)
    teacher.add_argument("--temperature", type=float, default=2.0)
    teacher.add_argument("--log-every", type=int, default=10)
    teacher.add_argument("--save-checkpoint")
    teacher.set_defaults(func=cmd_train_macro_teacher)

    collect = sub.add_parser("collect-boundaries")
    collect.add_argument("--config", required=True)
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--num-batches", type=int, default=256)
    collect.set_defaults(func=cmd_collect_boundaries)

    diagnose = sub.add_parser("diagnose-coda-collusion")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--checkpoint", required=True)
    diagnose.add_argument("--teacher-checkpoint")
    diagnose.set_defaults(func=cmd_diagnose_coda_collusion)

    macro_range = sub.add_parser("diagnose-macro-range")
    macro_range.add_argument("--config", required=True)
    macro_range.add_argument("--teacher-checkpoint", required=True)
    macro_range.add_argument("--fixed-depth", type=int)
    macro_range.add_argument("--fixed-recipe", type=int)
    macro_range.add_argument("--batches", type=int, default=32)
    macro_range.set_defaults(func=cmd_diagnose_macro_range)

    ladder = sub.add_parser("run-macro-capacity-ladder")
    ladder.add_argument("--config", required=True)
    ladder.add_argument("--teacher-checkpoint", required=True)
    ladder.add_argument("--fixed-depth", type=int)
    ladder.add_argument("--fixed-recipe", type=int)
    ladder.add_argument("--steps", type=int, default=1000)
    ladder.set_defaults(func=cmd_run_macro_capacity_ladder)

    compare_configs = sub.add_parser("compare-configs")
    compare_configs.add_argument("old_config")
    compare_configs.add_argument("new_config")
    compare_configs.set_defaults(func=cmd_compare_configs)

    ablate = sub.add_parser("run-ablations")
    ablate.add_argument("--config", required=True)
    ablate.add_argument("--steps", type=int, default=1)
    ablate.add_argument("--max-ablations", type=int)
    ablate.set_defaults(func=cmd_run_ablations)

    ttq = sub.add_parser("compare-ttq")
    ttq.add_argument("--runs-dir", default="runs")
    ttq.add_argument("--target-loss", type=float, required=True)
    ttq.set_defaults(func=cmd_compare_ttq)

    summarize = sub.add_parser("summarize-run")
    summarize.add_argument("run")
    summarize.set_defaults(func=cmd_summarize_run)

    summarize_alignment = sub.add_parser("summarize-alignment")
    summarize_alignment.add_argument("run")
    summarize_alignment.set_defaults(func=cmd_summarize_alignment)

    summarize_cmp = sub.add_parser("summarize-comparison")
    summarize_cmp.add_argument("runs", nargs="+")
    summarize_cmp.set_defaults(func=cmd_summarize_comparison)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
