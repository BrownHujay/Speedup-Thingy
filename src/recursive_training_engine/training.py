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
from recursive_training_engine.macro import (
    macro_alignment_metrics,
    macro_distill_loss,
)
from recursive_training_engine.models import DenseModel, ModelOutput, RecursiveModel, load_compatible_state_dict
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
        self.teacher_model: RecursiveModel | None = None
        if dense_model is None and config.model.topology == "dense":
            self.dense_model = DenseModel(config.model)
        if recursive_model is None and config.model.topology == "recursive":
            self.recursive_model = RecursiveModel(config.model, config.output)
        if self.dense_model is not None:
            self.dense_model.to(self.device)
        if self.recursive_model is not None:
            self.recursive_model.to(self.device)
        if (
            config.training.mode == "recursive_macro_lm_aligned"
            and config.training.aligned_lm_freeze_teacher
            and config.training.aligned_lm_teacher_checkpoint is not None
        ):
            self.teacher_model = RecursiveModel(config.model, config.output).to(self.device)
            payload = torch.load(
                config.training.aligned_lm_teacher_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            if payload.get("model") is None:
                raise RuntimeError(
                    f"teacher checkpoint has no model state: {config.training.aligned_lm_teacher_checkpoint}"
                )
            load_compatible_state_dict(self.teacher_model, payload["model"], skip_prefixes=("macro.",))
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)
        if config.training.compile_model and hasattr(torch, "compile") and self.device.type != "mps":
            if self.dense_model is not None:
                self.dense_model = torch.compile(
                    self.dense_model, mode=config.training.compile_mode
                )
            if self.recursive_model is not None:
                self.recursive_model.compile_hot_paths(mode=config.training.compile_mode)
        self.aligned_lm_phase = "A"
        self._apply_trainable_policy()
        self.optimizer = self._build_optimizer(self._optimizer_param_groups())
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
        self.last_grad_metrics: dict[str, Any] = {}

    def _maybe_log(self, metrics: dict[str, Any]) -> None:
        self.last_metrics = metrics
        log_every = max(1, self.config.training.log_every)
        logical_step = max(self.global_step, self.global_micro_step)
        if logical_step % log_every == 0:
            self.logger.write({"step": logical_step, "optimizer_step": self.global_step, **metrics})

    def write_run_manifest(self, extra: dict[str, Any] | None = None) -> None:
        write_json(self.run_dir / "manifest.json", build_manifest(self.config, extra=extra))

    def _apply_trainable_policy(self) -> None:
        if self.recursive_model is None:
            return
        if self.config.training.mode == "recursive_macro_distill_only":
            for param in self.recursive_model.parameters():
                param.requires_grad_(False)
            for param in self.recursive_model.macro.parameters():
                param.requires_grad_(True)
        if self.config.training.mode == "recursive_macro_lm_aligned":
            self._set_recursive_trainable_modules(self.config.training.aligned_lm_phase_a_train)

    def _set_recursive_trainable_modules(self, trainable_names: list[str]) -> None:
        if self.recursive_model is None:
            return
        names = set(trainable_names)
        for param in self.recursive_model.parameters():
            param.requires_grad_(False)
        module_params: dict[str, list[torch.nn.Parameter]] = {
            "embedding": list(self.recursive_model.embed.parameters()),
            "prelude": list(self.recursive_model.prelude.parameters()),
            "core": list(self.recursive_model.core.parameters()),
            "macro": list(self.recursive_model.macro.parameters()),
            "coda": [*self.recursive_model.coda.parameters(), *self.recursive_model.final_norm.parameters()],
            "router": list(self.recursive_model.router.parameters()),
            "output": list(self.recursive_model.lm_head.parameters())
            if self.recursive_model.lm_head is not None
            else [],
        }
        for name in names:
            for param in module_params.get(name, []):
                param.requires_grad_(True)

    def _metric_float(
        self,
        metrics: dict[str, Any] | None,
        *names: str,
        default: float | None = None,
    ) -> float | None:
        if metrics is None:
            return default
        for name in names:
            if name not in metrics:
                continue
            value = metrics[name]
            if isinstance(value, torch.Tensor):
                return float(value.detach().float().cpu())
            return float(value)
        return default

    def _aligned_lm_phase_a_gate(self, metrics: dict[str, Any]) -> bool:
        tr = self.config.training
        hidden_cos = self._metric_float(metrics, "hidden_cosine_exact_macro", "audit_hidden_cosine_exact_macro", default=0.0)
        gap_value = self._metric_float(metrics, "hot_exact_nll_gap", default=float("inf"))
        gap = abs(float("inf") if gap_value is None else gap_value)
        delta_ratio = self._metric_float(metrics, "delta_rms_ratio", "audit_delta_rms_ratio", default=float("inf"))
        hidden_mse = self._metric_float(metrics, "hidden_mse_exact_macro", "audit_hidden_mse_exact_macro", default=float("inf"))
        macro_norm = self._metric_float(metrics, "macro_norm", "audit_macro_norm", default=float("inf"))
        exact_norm = self._metric_float(metrics, "exact_norm", "audit_exact_norm", default=0.0) or 0.0
        prev_mse = self._metric_float(
            self.last_metrics,
            "hidden_mse_exact_macro",
            "audit_hidden_mse_exact_macro",
            default=None,
        )
        prev_norm = self._metric_float(self.last_metrics, "macro_norm", "audit_macro_norm", default=None)
        mse_ok = prev_mse is None or hidden_mse <= max(prev_mse + 0.05, 1.0)
        norm_limit = tr.audit_schedule_macro_norm_threshold
        if norm_limit is None:
            norm_limit = tr.audit_schedule_macro_norm_threshold_mult * exact_norm
        norm_ok = macro_norm <= norm_limit and (
            prev_norm is None or macro_norm <= max(prev_norm * 1.05, prev_norm + 0.05)
        )
        return (
            (0.0 if hidden_cos is None else hidden_cos) >= 0.95
            and (float("inf") if delta_ratio is None else delta_ratio) >= 0.8
            and (float("inf") if delta_ratio is None else delta_ratio) <= 1.25
            and gap <= 0.25
            and mse_ok
            and norm_ok
        )

    def _aligned_lm_phase_b_gate(self, metrics: dict[str, Any]) -> bool:
        tr = self.config.training
        hidden_cos = self._metric_float(metrics, "hidden_cosine_exact_macro", "audit_hidden_cosine_exact_macro", default=0.0)
        gap_value = self._metric_float(metrics, "hot_exact_nll_gap", default=float("inf"))
        gap = abs(float("inf") if gap_value is None else gap_value)
        delta_ratio = self._metric_float(metrics, "delta_rms_ratio", "audit_delta_rms_ratio", default=float("inf"))
        hidden_mse = self._metric_float(metrics, "hidden_mse_exact_macro", "audit_hidden_mse_exact_macro", default=float("inf"))
        macro_norm = self._metric_float(metrics, "macro_norm", "audit_macro_norm", default=float("inf"))
        exact_norm = self._metric_float(metrics, "exact_norm", "audit_exact_norm", default=0.0) or 0.0
        prev_mse = self._metric_float(
            self.last_metrics,
            "hidden_mse_exact_macro",
            "audit_hidden_mse_exact_macro",
            default=None,
        )
        norm_limit = tr.audit_schedule_macro_norm_threshold
        if norm_limit is None:
            norm_limit = tr.audit_schedule_macro_norm_threshold_mult * exact_norm
        return (
            (0.0 if hidden_cos is None else hidden_cos) >= tr.audit_schedule_hidden_cosine_threshold
            and (float("inf") if delta_ratio is None else delta_ratio) >= tr.audit_schedule_delta_rms_ratio_min
            and (float("inf") if delta_ratio is None else delta_ratio) <= tr.audit_schedule_delta_rms_ratio_max
            and gap <= tr.audit_schedule_nll_gap_threshold
            and (float("inf") if hidden_mse is None else hidden_mse) <= max(
                tr.audit_schedule_hidden_mse_threshold,
                (0.0 if prev_mse is None else prev_mse) + 0.05,
            )
            and (float("inf") if macro_norm is None else macro_norm) <= norm_limit
        )

    def _maybe_update_aligned_lm_phase(self, metrics: dict[str, Any]) -> None:
        tr = self.config.training
        if tr.mode != "recursive_macro_lm_aligned":
            return
        if self.aligned_lm_phase == "A":
            if self.global_step >= tr.coda_warmup_steps and self._aligned_lm_phase_a_gate(metrics):
                self.aligned_lm_phase = "B"
        elif self.aligned_lm_phase == "B":
            if not self._aligned_lm_phase_a_gate(metrics):
                self.aligned_lm_phase = "A"
            elif tr.unfreeze_prelude_core_after_gate and self._aligned_lm_phase_b_gate(metrics):
                self.aligned_lm_phase = "C"
        elif self.aligned_lm_phase == "C" and not self._aligned_lm_phase_b_gate(metrics):
            self.aligned_lm_phase = "B"

    def _set_coda_trainable_for_schedule(self) -> None:
        if self.recursive_model is None:
            return
        tr = self.config.training
        if tr.mode != "recursive_macro_lm_aligned":
            return
        phase_train = tr.aligned_lm_phase_a_train
        if self.aligned_lm_phase == "B":
            phase_train = tr.aligned_lm_phase_b_train
        elif self.aligned_lm_phase == "C":
            phase_train = tr.aligned_lm_phase_c_train
        self._set_recursive_trainable_modules(phase_train)

    def _dedupe_params(self, params: list[torch.nn.Parameter], seen: set[int]) -> list[torch.nn.Parameter]:
        out = []
        for param in params:
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(param)
        return out

    def _optimizer_param_groups(self) -> list[dict[str, Any]]:
        tr = self.config.training
        if self.dense_model is not None:
            return [
                {
                    "params": list(self.dense_model.parameters()),
                    "lr": tr.effective_lr_base,
                    "base_lr": tr.effective_lr_base,
                    "name": "dense",
                }
            ]
        if self.recursive_model is None:
            return []
        model = self.recursive_model
        seen: set[int] = set()
        specs: list[tuple[str, float, list[torch.nn.Parameter]]] = [
            ("embedding", tr.effective_lr_base, list(model.embed.parameters())),
            ("prelude", tr.effective_lr_base, list(model.prelude.parameters())),
            ("core", tr.effective_lr_base, list(model.core.parameters())),
            ("macro", tr.effective_lr_macro, list(model.macro.parameters())),
            (
                "coda",
                tr.effective_lr_coda,
                [*model.coda.parameters(), *model.final_norm.parameters()],
            ),
            (
                "output",
                tr.effective_lr_output,
                list(model.lm_head.parameters()) if model.lm_head is not None else [],
            ),
            ("router", tr.effective_lr_router, list(model.router.parameters())),
        ]
        groups: list[dict[str, Any]] = []
        for name, lr, params in specs:
            unique = self._dedupe_params(params, seen)
            if unique:
                groups.append({"params": unique, "lr": lr, "base_lr": lr, "name": name})
        return groups

    def _build_optimizer(self, params: list[dict[str, Any]]) -> torch.optim.Optimizer:
        kwargs: dict[str, Any] = {
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

    def _optimizer_grad_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for group in self.optimizer.param_groups:
            name = str(group.get("name", "group"))
            grad_sq = 0.0
            param_sq = 0.0
            for param in group["params"]:
                param_sq += float(param.detach().float().square().sum().cpu())
                if param.grad is not None:
                    grad_sq += float(param.grad.detach().float().square().sum().cpu())
            grad_norm = grad_sq**0.5
            param_norm = param_sq**0.5
            metrics[f"grad_norm_{name}"] = grad_norm
            metrics[f"update_ratio_{name}"] = float(group["lr"]) * grad_norm / max(param_norm, 1e-12)
        return metrics

    def _finish_step(self, loss: torch.Tensor) -> bool:
        accum = self.config.training.grad_accum_steps
        self.last_grad_metrics = {}
        if not torch.isfinite(loss.detach()):
            self.optimizer.zero_grad(set_to_none=True)
            self.global_micro_step += 1
            self.last_grad_metrics = {"skipped_nonfinite_loss": 1.0}
            return False
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
            self.last_grad_metrics = self._optimizer_grad_metrics()
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
            group["lr"] = float(group.get("base_lr", lr)) * self._lr_scale()
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

    def _router_aux_losses(
        self,
        recipe_probs: torch.Tensor,
        expected_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        tr = self.config.training
        if tr.disable_router_aux:
            zero = recipe_probs.new_zeros(())
            return {"load": zero, "cover": zero, "depth": zero}
        return router_aux_losses(
            recipe_probs,
            expected_depth,
            self.recursive_model.recipe_bank.usage_ema,
            tr,
        )

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
        self._set_coda_trainable_for_schedule()
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
            **self.last_grad_metrics,
            **self._accum_metrics(tokens, stepped),
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_exact(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._set_coda_trainable_for_schedule()
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
            aux = self._router_aux_losses(
                out.meta.router.recipe_probs,
                out.meta.router.expected_depth,
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
            **self.last_grad_metrics,
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
        if shortlist and self.config.training.debug_force_full_output:
            raise RuntimeError("debug_force_full_output forbids recursive_macro_shortlist")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        mode = "recursive_macro_shortlist" if shortlist else tr.mode
        if mode == "recursive_macro_lm_aligned":
            mode = "recursive_macro_lm_aligned"
        elif mode not in {"recursive_macro_shortlist", "recursive_macro"}:
            mode = "recursive_macro"
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
            audit_model = self.teacher_model if self.teacher_model is not None else self.recursive_model
            with self._autocast():
                audit = self.audit_engine.correction(
                    audit_model,
                    tokens,
                    targets,
                    hot,
                    seed=tr.seed + 10_000 + self.global_step,
                )
            aux_router = self._router_aux_losses(
                hot.meta.router.recipe_probs,
                hot.meta.router.expected_depth,
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
        required_audit_metrics = {
            key: audit.metrics[key]
            for key in [
                "audit_count",
                "audit_inclusion_prob",
                "audit_residual_mean",
                "audit_residual_std",
                "audit_residual_var",
                "hidden_mse_exact_macro",
                "hidden_cosine_exact_macro",
                "logit_kl_exact_macro",
            ]
            if key in audit.metrics
        }
        metrics = {
            "mode": mode,
            "loss": loss,
            "hot_lm_loss": hot.loss_per_sample.mean() / self._loss_scale(targets),
            "corrected_lm_loss": audit.corrected_loss_per_sample.mean() / self._loss_scale(targets),
            "nll_per_token": audit.corrected_loss_per_sample.mean() / float(targets.shape[1]),
            "hot_eval_nll": hot.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "macro_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "macro"),
                current_lr,
            ),
            "coda_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "coda"),
                current_lr,
            ),
            "coda_trainable": any(
                param.requires_grad for param in self.recursive_model.coda.parameters()
            ),
            "aligned_lm_phase": self.aligned_lm_phase,
            "active_touches_per_token": hot.meta.active_touches.mean() if hot.meta.active_touches is not None else 0.0,
            "avg_depth": hot.meta.depths.float().mean() if hot.meta.depths is not None else 0.0,
            "unique_recipes_per_batch": hot.meta.recipe_ids.unique().numel() if hot.meta.recipe_ids is not None else 0.0,
            "fixed_depth": tr.fixed_depth or 0,
            "fixed_recipe": tr.fixed_recipe or 0,
            "physical_passes": hot.meta.macro_trace.physical_passes.mean() if hot.meta.macro_trace is not None else 0.0,
            "macro_decomposition_error": hot.meta.macro_trace.decomposition_error if hot.meta.macro_trace is not None and hot.meta.macro_trace.decomposition_error is not None else 0.0,
            "shortlist_duplicate_count": hot.meta.shortlist.duplicate_count if hot.meta.shortlist is not None else 0.0,
            "shortlist_size": hot.meta.shortlist.shortlist.shape[-1] if hot.meta.shortlist is not None else self.config.model.vocab_size,
            **budget_metrics,
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
            **required_audit_metrics,
            **{f"audit_{k}": v for k, v in audit.metrics.items()},
            **{f"macro_{k}": v for k, v in audit.macro_aux.items()},
            "macro_distill_loss": sum(audit.macro_aux.values()),
            "macro_hidden_loss": audit.macro_aux.get("hid", hot.loss_per_sample.new_zeros(())),
            "macro_logit_kl_loss": audit.macro_aux.get("kl", hot.loss_per_sample.new_zeros(())),
            "macro_consistency_loss": audit.macro_aux.get("cons", hot.loss_per_sample.new_zeros(())),
            "macro_delta_dir_loss": audit.macro_aux.get("delta_dir", hot.loss_per_sample.new_zeros(())),
            "macro_delta_rms_loss": audit.macro_aux.get("delta_rms", hot.loss_per_sample.new_zeros(())),
            "macro_endpoint_normed_loss": audit.macro_aux.get(
                "endpoint_normed",
                hot.loss_per_sample.new_zeros(()),
            ),
            "macro_endpoint_raw_loss": audit.macro_aux.get(
                "endpoint_raw",
                hot.loss_per_sample.new_zeros(()),
            ),
            "macro_rms_trust_loss": audit.macro_aux.get("rms_trust", hot.loss_per_sample.new_zeros(())),
            **{f"aux_{k}": v for k, v in aux_router.items()},
        }
        if audit.exact is not None and audit.exact.loss_per_sample is not None:
            exact_eval_nll = audit.exact.loss_per_sample.mean() / float(targets.shape[1])
            metrics["exact_eval_nll"] = exact_eval_nll
            metrics["hot_exact_nll_gap"] = metrics["hot_eval_nll"] - exact_eval_nll
        self.tokens_seen += int(tokens.numel())
        self._maybe_update_aligned_lm_phase(metrics)
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=hot, metrics=metrics, audit=audit)

    def train_step_recursive_macro_distill_only(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._apply_trainable_policy()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast(), torch.no_grad():
                exact = self.recursive_model.forward_exact(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_depth=tr.fixed_depth,
                )
            if (
                self.config.model.macro_type == "v2_delta_radius"
                and self.config.model.macro_radius_init_from_teacher
                and exact.meta.h0 is not None
                and exact.meta.recurrent_hidden is not None
                and tr.fixed_recipe is not None
                and tr.fixed_depth is not None
            ):
                radius = (
                    exact.meta.recurrent_hidden.detach() - exact.meta.h0.detach()
                ).float().pow(2).mean(dim=(1, 2)).sqrt().mean()
                current = self.recursive_model.macro.teacher_delta_rms[
                    tr.fixed_recipe,
                    self.recursive_model.macro.stride_to_idx[tr.fixed_depth],
                ]
                if float(current.detach().cpu()) == 0.0:
                    self.recursive_model.macro.initialize_radius_from_teacher_delta(
                        tr.fixed_recipe,
                        tr.fixed_depth,
                        radius,
                    )
            with self._autocast():
                hot = self.recursive_model.forward_macro(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    shortlist=False,
                    seed=tr.seed + self.global_step,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_depth=tr.fixed_depth,
                )
            hot_endpoint = (
                hot.meta.recurrent_hidden if hot.meta.recurrent_hidden is not None else hot.meta.hidden
            )
            exact_endpoint = (
                exact.meta.recurrent_hidden if exact.meta.recurrent_hidden is not None else exact.meta.hidden
            )
            if hot_endpoint is None or exact_endpoint is None:
                raise RuntimeError("macro distillation requires hidden states")
            losses = macro_distill_loss(
                hot_endpoint,
                exact_endpoint,
                hot.meta.logits,
                exact.meta.logits,
                h0=hot.meta.h0,
                lambda_hid=tr.effective_lambda_hid,
                lambda_cos=tr.effective_lambda_cos,
                lambda_kl=tr.effective_lambda_kl,
                lambda_norm=tr.lambda_norm,
                lambda_delta_dir=tr.lambda_delta_dir,
                lambda_delta_rms=tr.lambda_delta_rms,
                lambda_endpoint_normed=tr.lambda_endpoint_normed,
                lambda_endpoint_raw=tr.lambda_endpoint_raw,
                lambda_macro_rms_trust=tr.lambda_macro_rms_trust
                if tr.macro_rms_trust_region
                else 0.0,
                temperature=tr.distill_temperature,
            )
            if tr.lambda_cons != 0.0 and hot.meta.h0 is not None and hot.meta.recipe_ids is not None:
                losses["cons"] = tr.lambda_cons * self.recursive_model.macro.consistency_loss(
                    hot.meta.h0,
                    hot.meta.h0,
                    hot.meta.recipe_ids,
                )
            else:
                losses["cons"] = hot.meta.hidden.new_zeros(())
            loss = sum(losses.values())
            stepped = self._finish_step(loss)
        assert hot.loss_per_sample is not None and exact.loss_per_sample is not None
        hot_endpoint = (
            hot.meta.recurrent_hidden if hot.meta.recurrent_hidden is not None else hot.meta.hidden
        )
        exact_endpoint = (
            exact.meta.recurrent_hidden if exact.meta.recurrent_hidden is not None else exact.meta.hidden
        )
        assert hot_endpoint is not None and exact_endpoint is not None
        exact_nll = exact.loss_per_sample.mean() / float(targets.shape[1])
        hot_nll = hot.loss_per_sample.mean() / float(targets.shape[1])
        align_metrics = macro_alignment_metrics(
            hot_endpoint,
            exact_endpoint,
            h0=hot.meta.h0,
            hot_logits=hot.meta.logits,
            exact_logits=exact.meta.logits,
            hot_nll=hot_nll,
            exact_nll=exact_nll,
        )
        metrics = {
            "mode": "recursive_macro_distill_only",
            "loss": loss,
            "macro_distill_loss": loss,
            "macro_hidden_loss": losses["hid"],
            "macro_logit_kl_loss": losses["kl"],
            "macro_consistency_loss": losses["cons"],
            "macro_norm_loss": losses["norm"],
            "macro_delta_dir_loss": losses["delta_dir"],
            "macro_delta_rms_loss": losses["delta_rms"],
            "macro_endpoint_normed_loss": losses["endpoint_normed"],
            "macro_endpoint_raw_loss": losses["endpoint_raw"],
            "macro_rms_trust_loss": losses["rms_trust"],
            "exact_eval_nll": exact_nll,
            "hot_eval_nll": hot_nll,
            **align_metrics,
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "macro_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "macro"),
                current_lr,
            ),
            "coda_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "coda"),
                current_lr,
            ),
            "coda_trainable": any(
                param.requires_grad for param in self.recursive_model.coda.parameters()
            ),
            "aligned_lm_phase": self.aligned_lm_phase,
            "unique_recipes_per_batch": hot.meta.recipe_ids.unique().numel() if hot.meta.recipe_ids is not None else 0.0,
            "fixed_depth": tr.fixed_depth or 0,
            "fixed_recipe": tr.fixed_recipe or 0,
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=hot, metrics=metrics)

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainStepResult:
        mode = self.config.training.mode
        if mode == "dense_exact":
            return self.train_step_dense(batch)
        if mode == "recursive_exact":
            return self.train_step_recursive_exact(batch)
        if mode == "recursive_macro":
            return self.train_step_recursive_macro(batch)
        if mode == "recursive_macro_lm_aligned":
            return self.train_step_recursive_macro(batch)
        if mode == "recursive_macro_shortlist":
            return self.train_step_recursive_macro_shortlist(batch)
        if mode == "recursive_macro_distill_only":
            return self.train_step_recursive_macro_distill_only(batch)
        if mode == "recursive_macro_shadow_coda":
            raise ValueError("recursive_macro_shadow_coda is diagnostic; use diagnose-coda-collusion")
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
            "aligned_lm_phase": self.aligned_lm_phase,
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
        if payload.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except ValueError:
                # Macro distillation checkpoints intentionally use a macro-only optimizer.
                # Resuming them into aligned LM training should load weights and start a
                # fresh optimizer with the current parameter groups.
                pass
        self.global_step = int(payload.get("global_step", 0))
        self.global_micro_step = int(payload.get("global_micro_step", 0))
        self.tokens_seen = int(payload.get("tokens_seen", 0))
        self.audit_engine.load_state_dict(payload.get("audit", {}))
        self.aligned_lm_phase = str(payload.get("aligned_lm_phase", self.aligned_lm_phase))
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].detach().cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        return payload.get("data_cursor")
