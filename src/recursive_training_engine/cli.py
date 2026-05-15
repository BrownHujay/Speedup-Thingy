from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from recursive_training_engine.artifacts import build_manifest, summarize_metrics, write_json
from recursive_training_engine.config import ExperimentConfig, ModelConfig, load_config, save_config
from recursive_training_engine.ablations import build_ablation_configs
from recursive_training_engine.data import load_token_streams
from recursive_training_engine.kernels import optimized, reference
from recursive_training_engine.layers import SVDFactorSparseFFN
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
