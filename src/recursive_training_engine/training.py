from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from recursive_training_engine.artifacts import build_manifest, write_json
from recursive_training_engine.audit import AuditEngine, AuditResult
from recursive_training_engine.config import ExperimentConfig, save_config
from recursive_training_engine.kernels import optimized
from recursive_training_engine.metrics import (
    dense_aligned_param_count,
    dense_param_count,
    estimate_hot_active_param_equiv_per_token,
    estimate_hot_flops_per_token,
    recursive_param_count,
    router_aux_losses,
)
from recursive_training_engine.models import DenseModel, ModelOutput, RecursiveModel
from recursive_training_engine.reporting import JsonlLogger
from recursive_training_engine.utils import WallTimer, default_device, maybe_peak_memory, set_seed


@dataclass(slots=True)
class TrainStepResult:
    loss: torch.Tensor
    model_output: ModelOutput
    metrics: dict[str, Any]
    audit: AuditResult | None = None


class TrainEngine:
    def __init__(
        self,
        config: ExperimentConfig,
        dense_model: DenseModel | None = None,
        recursive_model: RecursiveModel | None = None,
        *,
        device: torch.device | None = None,
    ):
        self.config = config
        self.device = device or default_device()
        set_seed(config.training.seed)
        if config.training.strict_cuda and self.device.type != "cuda":
            raise RuntimeError("strict_cuda requires a CUDA device")
        if config.training.strict_cuda and config.training.require_triton and not optimized.triton_available():
            raise RuntimeError("strict_cuda with require_triton needs CUDA + Triton")
        optimized.set_strict_cuda(
            config.training.strict_cuda,
            require_flash=config.training.require_flash_attention,
        )
        if config.training.allow_tf32:
            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        self.dense_model = dense_model
        self.recursive_model = recursive_model
        if dense_model is None and config.model.topology == "dense":
            self.dense_model = DenseModel(config.model)
        if recursive_model is None and config.model.topology == "recursive":
            self.recursive_model = RecursiveModel(config.model, config.output)
        if self.dense_model is not None:
            self.dense_model.to(self.device)
        if self.recursive_model is not None:
            self.recursive_model.to(self.device)
        if config.training.compile_model and hasattr(torch, "compile") and self.device.type != "mps":
            if self.dense_model is not None:
                self.dense_model = torch.compile(
                    self.dense_model, mode=config.training.compile_mode
                )
            if self.recursive_model is not None:
                self.recursive_model.compile_hot_paths(mode=config.training.compile_mode)
        params = []
        if self.dense_model is not None:
            params.extend(self.dense_model.parameters())
        if self.recursive_model is not None:
            params.extend(self.recursive_model.parameters())
        self.optimizer = self._build_optimizer(params)
        self.optimizer.zero_grad(set_to_none=True)
        self.audit_engine = AuditEngine(config.training)
        self.run_dir = Path(config.output_dir) / config.run_name
        save_config(config, self.run_dir / "resolved_config.yaml")
        write_json(self.run_dir / "manifest.json", build_manifest(config))
        self.logger = JsonlLogger(self.run_dir / "metrics.jsonl")
        self.global_step = 0
        self.global_micro_step = 0
        self.tokens_seen = 0
        self.last_metrics: dict[str, Any] | None = None

    def _maybe_log(self, metrics: dict[str, Any]) -> None:
        self.last_metrics = metrics
        log_every = max(1, self.config.training.log_every)
        logical_step = max(self.global_step, self.global_micro_step)
        if logical_step % log_every == 0:
            self.logger.write({"step": logical_step, "optimizer_step": self.global_step, **metrics})

    def write_run_manifest(self, extra: dict[str, Any] | None = None) -> None:
        write_json(self.run_dir / "manifest.json", build_manifest(self.config, extra=extra))

    def _build_optimizer(self, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        kwargs: dict[str, Any] = {
            "lr": self.config.training.lr,
            "weight_decay": self.config.training.weight_decay,
        }
        if self.device.type == "cuda" and self.config.training.fused_optimizer:
            kwargs["fused"] = True
        elif self.config.training.foreach_optimizer and self.device.type in {"cuda", "cpu"}:
            kwargs["foreach"] = True
        try:
            return torch.optim.AdamW(params, **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            kwargs.pop("foreach", None)
            return torch.optim.AdamW(params, **kwargs)

    def _autocast(self):
        if self.config.training.precision != "bf16":
            return nullcontext()
        if self.device.type == "mps":
            return nullcontext()
        if self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def close(self) -> None:
        if self.logger.rows_written == 0 and self.last_metrics is not None:
            logical_step = max(self.global_step, self.global_micro_step)
            self.logger.write(
                {
                    "event": "final",
                    "step": logical_step,
                    "optimizer_step": self.global_step,
                    **self.last_metrics,
                }
            )
        self.logger.close()

    def _move_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return batch[0].to(self.device), batch[1].to(self.device)

    def _loss_scale(self, targets: torch.Tensor) -> float:
        if self.config.training.loss_normalization == "token_mean":
            return float(targets.shape[1])
        return 1.0

    def _finish_step(self, loss: torch.Tensor) -> bool:
        accum = self.config.training.grad_accum_steps
        (loss / accum).backward()
        self.global_micro_step += 1
        should_step = self.global_micro_step % accum == 0
        if should_step and self.config.training.grad_clip_norm is not None:
            params = []
            if self.dense_model is not None:
                params.extend(self.dense_model.parameters())
            if self.recursive_model is not None:
                params.extend(self.recursive_model.parameters())
            torch.nn.utils.clip_grad_norm_(params, self.config.training.grad_clip_norm)
        if should_step:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        return should_step

    def _lr_scale(self) -> float:
        tr = self.config.training
        if tr.lr_schedule == "constant":
            return 1.0
        if tr.lr_schedule == "linear_decay_after":
            assert tr.lr_decay_start_tokens is not None and tr.lr_decay_end_tokens is not None
            if self.tokens_seen <= tr.lr_decay_start_tokens:
                return 1.0
            if self.tokens_seen >= tr.lr_decay_end_tokens:
                return tr.lr_final_scale
            progress = (self.tokens_seen - tr.lr_decay_start_tokens) / (
                tr.lr_decay_end_tokens - tr.lr_decay_start_tokens
            )
            return 1.0 + progress * (tr.lr_final_scale - 1.0)
        raise ValueError(tr.lr_schedule)

    def _apply_lr_schedule(self) -> float:
        lr = self.config.training.lr * self._lr_scale()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _accum_metrics(self, tokens: torch.Tensor, optimizer_step: bool) -> dict[str, Any]:
        tr = self.config.training
        return {
            "did_optimizer_step": optimizer_step,
            "grad_accum_index": (self.global_micro_step - 1) % tr.grad_accum_steps + 1,
            "grad_accum_steps": tr.grad_accum_steps,
            "effective_batch_size": tr.batch_size * tr.grad_accum_steps,
            "accumulated_tokens": tokens.numel() * tr.grad_accum_steps,
        }

    def _enforce_hot_budget(
        self,
        *,
        shortlist_size: torch.Tensor | float | int | None,
        include_output: bool,
    ) -> dict[str, Any]:
        tr = self.config.training
        active = estimate_hot_active_param_equiv_per_token(
            self.config.model,
            shortlist_size=shortlist_size,
            include_output=include_output,
        )
        flops = estimate_hot_flops_per_token(active)
        target_limit = None
        if tr.target_speedup_vs_dense is not None:
            target_limit = dense_aligned_param_count(self.config.model) / tr.target_speedup_vs_dense
        active_limit = tr.max_active_param_equiv_per_token
        if target_limit is not None:
            active_limit = target_limit if active_limit is None else min(active_limit, target_limit)
        if active_limit is not None and active > active_limit:
            raise RuntimeError(
                "recursive hot path exceeds active parameter budget: "
                f"{active:.1f} > {active_limit:.1f}"
            )
        if tr.max_hotpath_flops_per_token is not None and flops > tr.max_hotpath_flops_per_token:
            raise RuntimeError(
                "recursive hot path exceeds FLOPs/token budget: "
                f"{flops:.1f} > {tr.max_hotpath_flops_per_token:.1f}"
            )
        return {
            "active_param_equiv_per_token": active,
            "hotpath_flops_per_token_est": flops,
            "active_param_budget": active_limit if active_limit is not None else 0.0,
        }

    def train_step_dense(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        if self.dense_model is None:
            raise ValueError("dense model is not available")
        tokens, targets = self._move_batch(batch)
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                out = self.dense_model(tokens, targets, return_loss_per_sample=True)
            assert out.loss_per_sample is not None
            loss = out.loss_per_sample.mean() / self._loss_scale(targets)
            stepped = self._finish_step(loss)
        metrics = {
            "mode": "dense_exact",
            "loss": loss,
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": dense_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            **self._accum_metrics(tokens, stepped),
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_exact(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                out = self.recursive_model.forward_exact(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_depth=tr.fixed_depth,
                )
            assert out.loss_per_sample is not None and out.meta.router is not None
            aux = router_aux_losses(
                out.meta.router.recipe_probs,
                out.meta.router.expected_depth,
                self.recursive_model.recipe_bank.usage_ema,
                tr,
            )
            lm_loss = out.loss_per_sample.mean() / self._loss_scale(targets)
            loss = lm_loss + sum(aux.values())
            stepped = self._finish_step(loss)
        if out.meta.recipe_ids is not None:
            self.recursive_model.recipe_bank.update_usage(
                out.meta.recipe_ids, beta=tr.coverage_beta
            )
        metrics = {
            "mode": "recursive_exact",
            "loss": loss,
            "lm_loss": lm_loss,
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "active_touches_per_token": out.meta.active_touches.mean() if out.meta.active_touches is not None else 0.0,
            "avg_depth": out.meta.depths.float().mean() if out.meta.depths is not None else 0.0,
            **self._accum_metrics(tokens, stepped),
            **{f"aux_{k}": v for k, v in aux.items()},
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_macro(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        return self._train_step_recursive_macro_impl(batch, shortlist=False)

    def train_step_recursive_macro_shortlist(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        return self._train_step_recursive_macro_impl(batch, shortlist=True)

    def _train_step_recursive_macro_impl(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        *,
        shortlist: bool,
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        mode = "recursive_macro_shortlist" if shortlist else "recursive_macro"
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                hot = self.recursive_model.forward_macro(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    shortlist=shortlist,
                    seed=tr.seed + self.global_step,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_depth=tr.fixed_depth,
                )
            assert hot.loss_per_sample is not None and hot.meta.router is not None
            with self._autocast():
                audit = self.audit_engine.correction(
                    self.recursive_model,
                    tokens,
                    targets,
                    hot,
                    seed=tr.seed + 10_000 + self.global_step,
                )
            aux_router = router_aux_losses(
                hot.meta.router.recipe_probs,
                hot.meta.router.expected_depth,
                self.recursive_model.recipe_bank.usage_ema,
                tr,
            )
            loss = (
                audit.corrected_loss_per_sample.mean() / self._loss_scale(targets)
                + sum(audit.macro_aux.values())
                + sum(aux_router.values())
            )
            shortlist_size = None
            if hot.meta.shortlist is not None:
                shortlist_size = hot.meta.shortlist.shortlist.shape[-1]
            budget_metrics = self._enforce_hot_budget(
                shortlist_size=shortlist_size,
                include_output=True,
            )
            stepped = self._finish_step(loss)
        if hot.meta.recipe_ids is not None:
            self.recursive_model.recipe_bank.update_usage(
                hot.meta.recipe_ids, beta=tr.coverage_beta
            )
        metrics = {
            "mode": mode,
            "loss": loss,
            "hot_lm_loss": hot.loss_per_sample.mean() / self._loss_scale(targets),
            "corrected_lm_loss": audit.corrected_loss_per_sample.mean() / self._loss_scale(targets),
            "nll_per_token": audit.corrected_loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "active_touches_per_token": hot.meta.active_touches.mean() if hot.meta.active_touches is not None else 0.0,
            "avg_depth": hot.meta.depths.float().mean() if hot.meta.depths is not None else 0.0,
            "physical_passes": hot.meta.macro_trace.physical_passes.mean() if hot.meta.macro_trace is not None else 0.0,
            "macro_decomposition_error": hot.meta.macro_trace.decomposition_error if hot.meta.macro_trace is not None and hot.meta.macro_trace.decomposition_error is not None else 0.0,
            "shortlist_duplicate_count": hot.meta.shortlist.duplicate_count if hot.meta.shortlist is not None else 0.0,
            "shortlist_size": hot.meta.shortlist.shortlist.shape[-1] if hot.meta.shortlist is not None else self.config.model.vocab_size,
            **budget_metrics,
            **self._accum_metrics(tokens, stepped),
            **{f"audit_{k}": v for k, v in audit.metrics.items()},
            **{f"macro_{k}": v for k, v in audit.macro_aux.items()},
            **{f"aux_{k}": v for k, v in aux_router.items()},
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=hot, metrics=metrics, audit=audit)

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        mode = self.config.training.mode
        if mode == "dense_exact":
            return self.train_step_dense(batch)
        if mode == "recursive_exact":
            return self.train_step_recursive_exact(batch)
        if mode == "recursive_macro":
            return self.train_step_recursive_macro(batch)
        if mode == "recursive_macro_shortlist":
            return self.train_step_recursive_macro_shortlist(batch)
        raise ValueError(mode)

    def save_checkpoint(self, path: str | Path, *, data_cursor: int | None = None) -> None:
        model = self.dense_model if self.dense_model is not None else self.recursive_model
        payload = {
            "config": self.config,
            "model": model.state_dict() if model is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "global_micro_step": self.global_micro_step,
            "tokens_seen": self.tokens_seen,
            "audit": self.audit_engine.state_dict(),
            "data_cursor": data_cursor,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)

    def load_checkpoint(self, path: str | Path) -> int | None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        model = self.dense_model if self.dense_model is not None else self.recursive_model
        if model is not None and payload.get("model") is not None:
            model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
        self.global_micro_step = int(payload.get("global_micro_step", 0))
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        self.audit_engine.load_state_dict(payload.get("audit", {}))
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].detach().cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        return payload.get("data_cursor")
