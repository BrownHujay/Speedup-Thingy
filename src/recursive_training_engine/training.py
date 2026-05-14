from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
from recursive_training_engine.routing import RouterOutput
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
        self.dense_teacher_model: DenseModel | None = None
        if dense_model is None and config.model.topology == "dense":
            self.dense_model = DenseModel(config.model)
        if recursive_model is None and config.model.topology == "recursive":
            self.recursive_model = RecursiveModel(config.model, config.output)
        if self.dense_model is not None:
            self.dense_model.to(self.device)
        if self.recursive_model is not None:
            self.recursive_model.to(self.device)
        if config.training.mode == "recursive_exact_dense_hidden_distill":
            if config.training.aligned_lm_teacher_checkpoint is None:
                raise RuntimeError(
                    "recursive_exact_dense_hidden_distill requires --teacher-checkpoint "
                    "pointing at a dense_exact checkpoint"
                )
            self.dense_teacher_model = DenseModel(
                replace(config.model, topology="dense")
            ).to(self.device)
            payload = torch.load(
                config.training.aligned_lm_teacher_checkpoint,
                map_location=self.device,
                weights_only=False,
            )
            if payload.get("model") is None:
                raise RuntimeError(
                    f"dense teacher checkpoint has no model state: "
                    f"{config.training.aligned_lm_teacher_checkpoint}"
                )
            self.dense_teacher_model.load_state_dict(payload["model"], strict=True)
            self.dense_teacher_model.eval()
            for param in self.dense_teacher_model.parameters():
                param.requires_grad_(False)
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
        if self.recursive_model is not None:
            core_lr = next(
                (
                    float(group["lr"])
                    for group in self.optimizer.param_groups
                    if group.get("name") == "core"
                ),
                self.config.training.effective_lr_base,
            )
            pass_groups = {
                "pass_film": (
                    self.recursive_model.core.pass_gamma1,
                    self.recursive_model.core.pass_beta1,
                    self.recursive_model.core.pass_gamma2,
                    self.recursive_model.core.pass_beta2,
                ),
                "pass_scale": (
                    self.recursive_model.core.pass_attn_scale,
                    self.recursive_model.core.pass_mlp_scale,
                ),
            }
            for name, params in pass_groups.items():
                grad_sq = 0.0
                param_sq = 0.0
                for param in params:
                    param_sq += float(param.detach().float().square().sum().cpu())
                    if param.grad is not None:
                        grad_sq += float(param.grad.detach().float().square().sum().cpu())
                grad_norm = grad_sq**0.5
                param_norm = param_sq**0.5
                metrics[f"grad_norm_{name}"] = grad_norm
                metrics[f"update_ratio_{name}"] = core_lr * grad_norm / max(param_norm, 1e-12)
            if self.recursive_model.core.global_corrector is not None:
                grad_sq = 0.0
                param_sq = 0.0
                for param in self.recursive_model.core.global_corrector.parameters():
                    param_sq += float(param.detach().float().square().sum().cpu())
                    if param.grad is not None:
                        grad_sq += float(param.grad.detach().float().square().sum().cpu())
                grad_norm = grad_sq**0.5
                param_norm = param_sq**0.5
                metrics["grad_norm_global_corrector"] = grad_norm
                metrics["update_ratio_global_corrector"] = core_lr * grad_norm / max(param_norm, 1e-12)
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

    def _route_depth(self) -> int:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        return int(self.config.training.fixed_depth or self.recursive_model.config.t_max)

    def _proposal_recipe_logits(self, route_logits: torch.Tensor) -> torch.Tensor:
        tr = self.config.training
        logits = route_logits.clone()
        if not tr.route_em_allow_dense_fallback and logits.shape[-1] > 1:
            logits[..., 0] = torch.finfo(logits.dtype).min
        return logits

    def _sample_route_candidates(
        self,
        h0: torch.Tensor,
        *,
        count: int,
        proposal: str | None = None,
    ) -> tuple[RouterOutput, torch.Tensor]:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        tr = self.config.training
        model = self.recursive_model
        batch = h0.shape[0]
        t_max = model.config.t_max
        recipe_start = 0 if tr.route_em_allow_dense_fallback else 1
        if recipe_start >= model.config.recipe_count:
            raise ValueError("route-em needs at least one selectable sparse recipe")
        route = model.router(h0.detach(), fixed_depth=self._route_depth())
        proposal = tr.route_em_proposal if proposal is None else proposal
        if proposal == "random" or route.recipe_logits_by_pass is None:
            schedules = torch.randint(
                recipe_start,
                model.config.recipe_count,
                (batch, count, t_max),
                device=h0.device,
                dtype=torch.long,
            )
            return route, schedules
        logits = self._proposal_recipe_logits(route.recipe_logits_by_pass.detach())
        probs = F.softmax(logits, dim=-1)
        top = probs.argmax(dim=-1)
        schedules = torch.empty(
            batch,
            count,
            t_max,
            device=h0.device,
            dtype=torch.long,
        )
        schedules[:, 0, :] = top
        if count > 1:
            draws = torch.multinomial(
                probs.reshape(batch * t_max, model.config.recipe_count),
                num_samples=count - 1,
                replacement=True,
            )
            schedules[:, 1:, :] = draws.view(batch, t_max, count - 1).permute(0, 2, 1)
        return route, schedules

    @torch.no_grad()
    def _score_route_candidates(
        self,
        h0: torch.Tensor,
        targets: torch.Tensor,
        schedules: torch.Tensor,
        *,
        fixed_depth: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        batch, candidates, t_max = schedules.shape
        flat_h0 = h0.detach().unsqueeze(1).expand(-1, candidates, -1, -1).reshape(
            batch * candidates,
            h0.shape[1],
            h0.shape[2],
        )
        flat_targets = targets.unsqueeze(1).expand(-1, candidates, -1).reshape(
            batch * candidates,
            targets.shape[1],
        )
        flat_schedules = schedules.reshape(batch * candidates, t_max)
        out = self.recursive_model.forward_exact_from_h0(
            flat_h0,
            flat_targets,
            return_loss_per_sample=True,
            fixed_recipe=self.config.training.fixed_recipe,
            fixed_depth=fixed_depth,
            recipe_schedule=flat_schedules,
        )
        if out.loss_per_sample is None:
            raise RuntimeError("route candidate scoring requires per-sample losses")
        scores = out.loss_per_sample.view(batch, candidates)
        return scores, scores / float(targets.shape[1])

    def _target_sparse_schedule(self) -> list[int]:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        tr = self.config.training
        schedule = list(
            tr.fixed_recipe_schedule
            or [tr.fixed_recipe if tr.fixed_recipe is not None else 1]
        )
        if not schedule:
            schedule = [1]
        while len(schedule) < self.recursive_model.config.t_max:
            schedule.append(schedule[-1])
        return schedule[: self.recursive_model.config.t_max]

    def _full_recipe_schedule(self) -> list[int]:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        return [0 for _ in range(self.recursive_model.config.t_max)]

    def _sample_sparse_schedule_tensor(self, batch: int, *, count: int) -> torch.Tensor:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        if count < 1:
            return torch.empty(
                batch,
                0,
                self.recursive_model.config.t_max,
                dtype=torch.long,
                device=self.device,
            )
        recipe_start = 1
        if recipe_start >= self.recursive_model.config.recipe_count:
            raise ValueError("sandwich supernet needs at least one sparse recipe")
        return torch.randint(
            recipe_start,
            self.recursive_model.config.recipe_count,
            (batch, count, self.recursive_model.config.t_max),
            dtype=torch.long,
            device=self.device,
        )

    def _sandwich_kl(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.config.training.sandwich_temperature
        teacher_logp = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
        student_logp = F.log_softmax(student_logits / temperature, dim=-1)
        return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1).mean() * (
            temperature * temperature
        )

    def _sandwich_hidden_loss(
        self,
        teacher: ModelOutput,
        student: ModelOutput,
        *,
        depth: int,
    ) -> torch.Tensor:
        if teacher.meta.recurrent_hidden is None or student.meta.recurrent_hidden is None:
            raise RuntimeError("sandwich hidden loss requires recurrent hidden states")
        loss = (
            self._rms_unit(teacher.meta.recurrent_hidden.detach())
            - self._rms_unit(student.meta.recurrent_hidden)
        ).square().mean()
        count = 1
        if teacher.meta.states is not None and student.meta.states is not None:
            for step in range(1, depth + 1):
                if step not in teacher.meta.states or step not in student.meta.states:
                    continue
                loss = loss + (
                    self._rms_unit(teacher.meta.states[step].detach())
                    - self._rms_unit(student.meta.states[step])
                ).square().mean()
                count += 1
        return loss / float(count)

    def _route_supervision_loss(
        self,
        route: RouterOutput,
        schedule: torch.Tensor,
        *,
        fixed_depth: int,
    ) -> torch.Tensor:
        if route.recipe_logits_by_pass is None:
            return route.recipe_logits.new_zeros(())
        t_max = schedule.shape[1]
        active = torch.arange(t_max, device=schedule.device) < fixed_depth
        logits = route.recipe_logits_by_pass[:, :t_max, :][:, active, :]
        targets = schedule[:, :t_max][:, active]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

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
                    fixed_recipe_schedule=tr.fixed_recipe_schedule,
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
            "fixed_recipe_schedule": tr.fixed_recipe_schedule or [],
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
            **{f"aux_{k}": v for k, v in aux.items()},
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_deferred_grouped_exact(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                out = self.recursive_model.forward_deferred_grouped_exact(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_recipe_schedule=tr.fixed_recipe_schedule,
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
            self.recursive_model.recipe_bank.update_usage(out.meta.recipe_ids, beta=tr.coverage_beta)
        active_touches = out.meta.active_touches.mean() if out.meta.active_touches is not None else 0.0
        depth = int(tr.fixed_depth or self.recursive_model.config.t_max)
        metrics = {
            "mode": "recursive_deferred_grouped_exact",
            "loss": loss,
            "lm_loss": lm_loss,
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "active_touches_per_token": active_touches,
            "active_touches_per_micro": active_touches / max(float(depth), 1.0),
            "avg_depth": out.meta.depths.float().mean() if out.meta.depths is not None else 0.0,
            "fixed_recipe_schedule": tr.fixed_recipe_schedule or [],
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
            **{f"aux_{k}": v for k, v in aux.items()},
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def _dense_distill_layer_map(self, depth: int) -> list[int]:
        tr = self.config.training
        if tr.dense_hidden_layer_map is not None:
            layers = [int(layer) for layer in tr.dense_hidden_layer_map]
            if len(layers) < depth:
                raise ValueError(
                    f"dense_hidden_layer_map has {len(layers)} entries but depth={depth}"
                )
            layers = layers[:depth]
        else:
            dense_layers = self.config.model.n_dense_layers
            layers = [math.ceil((idx + 1) * dense_layers / depth) for idx in range(depth)]
        max_layer = self.config.model.n_dense_layers
        if any(layer < 1 or layer > max_layer for layer in layers):
            raise ValueError(
                f"dense hidden layer map entries must be in [1, {max_layer}], got {layers}"
            )
        return layers

    def _rms_unit(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)

    def _dense_hidden_distill_losses(
        self,
        tokens: torch.Tensor,
        exact: ModelOutput,
        dense: ModelOutput,
        *,
        depth: int,
    ) -> dict[str, torch.Tensor]:
        if self.dense_teacher_model is None:
            raise ValueError("dense teacher model is not available")
        if exact.meta.states is None or dense.meta.states is None:
            raise RuntimeError("dense hidden distillation requires returned states")
        if exact.meta.h0 is None:
            raise RuntimeError("recursive exact output is missing h0")
        if exact.meta.logits is None or dense.meta.logits is None:
            raise RuntimeError("dense hidden distillation requires full logits")

        layer_map = self._dense_distill_layer_map(depth)
        hidden_loss = exact.meta.h0.new_zeros(())
        delta_loss = exact.meta.h0.new_zeros(())
        cosine_sum = exact.meta.h0.new_zeros(())
        rec_prev = exact.meta.h0
        dense_prev = self.dense_teacher_model.embed(tokens).detach()
        for rec_depth, dense_layer in zip(range(1, depth + 1), layer_map, strict=True):
            if rec_depth not in exact.meta.states:
                raise RuntimeError(
                    f"recursive state {rec_depth} is missing; include it in model.depth_choices"
                )
            if dense_layer not in dense.meta.states:
                raise RuntimeError(f"dense teacher state {dense_layer} is missing")
            rec_state = exact.meta.states[rec_depth]
            dense_state = dense.meta.states[dense_layer].detach()
            rec_normed = self._rms_unit(rec_state)
            dense_normed = self._rms_unit(dense_state)
            hidden_loss = hidden_loss + (rec_normed - dense_normed).square().mean()
            cosine_sum = cosine_sum + F.cosine_similarity(
                rec_normed.flatten(1),
                dense_normed.flatten(1),
                dim=-1,
            ).mean()
            delta_loss = delta_loss + (rec_state.float() - rec_prev.float() - (dense_state.float() - dense_prev.float())).square().mean()
            rec_prev = rec_state
            dense_prev = dense_state

        denom = float(depth)
        hidden_loss = hidden_loss / denom
        delta_loss = delta_loss / denom
        hidden_cosine = cosine_sum / denom

        temperature = self.config.training.dense_distill_temperature
        dense_logp = F.log_softmax(dense.meta.logits.detach() / temperature, dim=-1)
        exact_logp = F.log_softmax(exact.meta.logits / temperature, dim=-1)
        dense_probs = dense_logp.exp()
        kl = (dense_probs * (dense_logp - exact_logp)).sum(dim=-1).mean() * (
            temperature * temperature
        )
        return {
            "hidden": hidden_loss,
            "delta": delta_loss,
            "kl": kl,
            "hidden_cosine": hidden_cosine,
        }

    def train_step_recursive_exact_dense_hidden_distill(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        if self.dense_teacher_model is None:
            raise ValueError("dense teacher model is not available")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        depth = self._route_depth()
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast(), torch.no_grad():
                dense = self.dense_teacher_model(
                    tokens,
                    targets,
                    return_loss_per_sample=True,
                    return_states=True,
                )
            with self._autocast():
                out = self.recursive_model.forward_exact(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_recipe_schedule=tr.fixed_recipe_schedule,
                    fixed_depth=tr.fixed_depth,
                    state_depths=list(range(1, depth + 1)),
                )
                assert out.loss_per_sample is not None and dense.loss_per_sample is not None
                losses = self._dense_hidden_distill_losses(
                    tokens,
                    out,
                    dense,
                    depth=depth,
                )
                lm_loss = out.loss_per_sample.mean() / self._loss_scale(targets)
                hidden_loss = tr.lambda_dense_hidden * losses["hidden"]
                delta_loss = tr.lambda_dense_delta * losses["delta"]
                kl_loss = tr.lambda_dense_kl * losses["kl"]
                loss = lm_loss + hidden_loss + delta_loss + kl_loss
            stepped = self._finish_step(loss)
        if out.meta.recipe_ids is not None:
            self.recursive_model.recipe_bank.update_usage(
                out.meta.recipe_ids, beta=tr.coverage_beta
            )
        sparse_touches = out.meta.active_touches.mean() if out.meta.active_touches is not None else 0.0
        global_touches = (
            3.0 * self.config.model.d_model * self.config.model.global_corrector_rank
            if self.config.model.use_global_lowrank_corrector
            else 0.0
        )
        metrics = {
            "mode": "recursive_exact_dense_hidden_distill",
            "loss": loss,
            "lm_loss": lm_loss,
            "dense_hidden_loss": losses["hidden"],
            "dense_delta_loss": losses["delta"],
            "dense_kl_loss": losses["kl"],
            "weighted_dense_hidden_loss": hidden_loss,
            "weighted_dense_delta_loss": delta_loss,
            "weighted_dense_kl_loss": kl_loss,
            "dense_hidden_cosine": losses["hidden_cosine"],
            "teacher_nll_per_token": dense.loss_per_sample.mean() / float(targets.shape[1]),
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "active_touches_per_token": sparse_touches,
            "global_corrector_touches_per_pass": global_touches,
            "active_touches_with_global_per_token": sparse_touches + global_touches,
            "avg_depth": out.meta.depths.float().mean() if out.meta.depths is not None else 0.0,
            "fixed_recipe_schedule": tr.fixed_recipe_schedule or [],
            "dense_hidden_layer_map": self._dense_distill_layer_map(depth),
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_exact_route_em(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        fixed_depth = self._route_depth()
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                h0 = self.recursive_model._prelude(tokens)
            with torch.no_grad():
                _, candidates = self._sample_route_candidates(h0, count=tr.route_em_candidates)
                candidate_scores, candidate_nll = self._score_route_candidates(
                    h0,
                    targets,
                    candidates,
                    fixed_depth=fixed_depth,
                )
                best_idx = candidate_scores.argmin(dim=1)
                batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
                best_schedule = candidates[batch_idx, best_idx]
                best_candidate_nll = candidate_nll[batch_idx, best_idx]
            with self._autocast():
                route = self.recursive_model.router(h0.detach(), fixed_depth=fixed_depth)
                out = self.recursive_model.forward_exact_from_h0(
                    h0,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=tr.fixed_recipe,
                    fixed_depth=fixed_depth,
                    recipe_schedule=best_schedule,
                )
                assert out.loss_per_sample is not None
                lm_loss = out.loss_per_sample.mean() / self._loss_scale(targets)
                router_loss = self._route_supervision_loss(
                    route,
                    best_schedule,
                    fixed_depth=fixed_depth,
                )
                loss = lm_loss + tr.lambda_router * router_loss
            stepped = self._finish_step(loss)
        self.recursive_model.recipe_bank.update_usage(best_schedule, beta=tr.coverage_beta)
        metrics = {
            "mode": "recursive_exact_route_em",
            "loss": loss,
            "lm_loss": lm_loss,
            "router_loss": router_loss,
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "route_oracle_nll_per_token": best_candidate_nll.mean(),
            "route_candidate_mean_nll": candidate_nll.mean(),
            "route_candidate_best_idx_mean": best_idx.float().mean(),
            "route_em_candidates": tr.route_em_candidates,
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "router_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "router"),
                current_lr,
            ),
            "active_touches_per_token": out.meta.active_touches.mean()
            if out.meta.active_touches is not None
            else 0.0,
            "avg_depth": out.meta.depths.float().mean() if out.meta.depths is not None else 0.0,
            "unique_recipes_per_batch": best_schedule.unique().numel(),
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_exact_factorized_soft(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        current_lr = self._apply_lr_schedule()
        with WallTimer() as timer:
            with self._autocast():
                out = self.recursive_model.forward_exact_factorized_soft(
                    tokens,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    top_k=tr.factorized_route_top_k,
                    temperature=tr.factorized_route_temperature,
                    fixed_depth=tr.fixed_depth,
                )
            assert out.loss_per_sample is not None and out.meta.router is not None
            lm_loss = out.loss_per_sample.mean() / self._loss_scale(targets)
            loss = lm_loss
            stepped = self._finish_step(loss)
        metrics = {
            "mode": "recursive_exact_factorized_soft",
            "loss": loss,
            "lm_loss": lm_loss,
            "nll_per_token": out.loss_per_sample.mean() / float(targets.shape[1]),
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "router_lr": next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name") == "router"),
                current_lr,
            ),
            "active_touches_per_token": out.meta.active_touches.mean()
            if out.meta.active_touches is not None
            else 0.0,
            "avg_depth": out.meta.depths.float().mean() if out.meta.depths is not None else 0.0,
            "factorized_route_top_k": tr.factorized_route_top_k,
            "factorized_route_temperature": tr.factorized_route_temperature,
            "factor_entropy": out.meta.router.factor_entropy.mean()
            if out.meta.router.factor_entropy is not None
            else 0.0,
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=out, metrics=metrics)

    def train_step_recursive_sandwich_supernet(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainStepResult:
        if self.recursive_model is None:
            raise ValueError("recursive model is not available")
        self._set_coda_trainable_for_schedule()
        tokens, targets = self._move_batch(batch)
        tr = self.config.training
        depth = self._route_depth()
        full_schedule = self._full_recipe_schedule()
        target_schedule = self._target_sparse_schedule()
        random_schedules = self._sample_sparse_schedule_tensor(
            tokens.shape[0],
            count=tr.sandwich_random_paths,
        )
        current_lr = self._apply_lr_schedule()
        state_depths = list(range(1, depth + 1))
        with WallTimer() as timer:
            with self._autocast():
                h0 = self.recursive_model._prelude(tokens)
                full = self.recursive_model.forward_exact_from_h0(
                    h0,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=0,
                    fixed_recipe_schedule=full_schedule,
                    fixed_depth=depth,
                    state_depths=state_depths,
                )
                thin = self.recursive_model.forward_exact_from_h0(
                    h0,
                    targets,
                    return_states=True,
                    return_loss_per_sample=True,
                    fixed_recipe=target_schedule[0],
                    fixed_recipe_schedule=target_schedule,
                    fixed_depth=depth,
                    state_depths=state_depths,
                )
                assert full.loss_per_sample is not None and thin.loss_per_sample is not None
                full_ce = full.loss_per_sample.mean() / self._loss_scale(targets)
                thin_ce = thin.loss_per_sample.mean() / self._loss_scale(targets)
                kd_thin = self._sandwich_kl(full.meta.logits, thin.meta.logits)
                hidden = self._sandwich_hidden_loss(full, thin, depth=depth)
                rand_ce = full_ce.new_zeros(())
                rand_kd = full_ce.new_zeros(())
                rand_nll = full_ce.new_zeros(())
                if tr.sandwich_random_paths > 0:
                    batch_size, rand_count, t_max = random_schedules.shape
                    flat_h0 = h0.unsqueeze(1).expand(-1, rand_count, -1, -1).reshape(
                        batch_size * rand_count,
                        h0.shape[1],
                        h0.shape[2],
                    )
                    flat_targets = targets.unsqueeze(1).expand(-1, rand_count, -1).reshape(
                        batch_size * rand_count,
                        targets.shape[1],
                    )
                    flat_schedule = random_schedules.reshape(batch_size * rand_count, t_max)
                    rand = self.recursive_model.forward_exact_from_h0(
                        flat_h0,
                        flat_targets,
                        return_loss_per_sample=True,
                        recipe_schedule=flat_schedule,
                        fixed_depth=depth,
                    )
                    assert rand.loss_per_sample is not None
                    expanded_full_logits = full.meta.logits.detach().unsqueeze(1).expand(
                        -1,
                        rand_count,
                        -1,
                        -1,
                    ).reshape_as(rand.meta.logits)
                    rand_ce = (
                        rand.loss_per_sample.mean()
                        / self._loss_scale(flat_targets)
                        * float(rand_count)
                    )
                    rand_kd = self._sandwich_kl(expanded_full_logits, rand.meta.logits) * float(
                        rand_count
                    )
                    rand_nll = rand.loss_per_sample.mean() / float(targets.shape[1])
                loss = (
                    full_ce
                    + thin_ce
                    + tr.lambda_sandwich_rand_ce * rand_ce
                    + tr.lambda_sandwich_kd * kd_thin
                    + tr.lambda_sandwich_rand_kd * rand_kd
                    + tr.lambda_sandwich_hidden * hidden
                )
            stepped = self._finish_step(loss)
        if thin.meta.recipe_ids is not None:
            self.recursive_model.recipe_bank.update_usage(thin.meta.recipe_ids, beta=tr.coverage_beta)
        if tr.sandwich_random_paths > 0:
            self.recursive_model.recipe_bank.update_usage(
                random_schedules.reshape(-1, random_schedules.shape[-1]),
                beta=tr.coverage_beta,
            )
        full_nll = full.loss_per_sample.mean() / float(targets.shape[1])
        thin_nll = thin.loss_per_sample.mean() / float(targets.shape[1])
        metrics = {
            "mode": "recursive_sandwich_supernet",
            "loss": loss,
            "full_ce_loss": full_ce,
            "thin_ce_loss": thin_ce,
            "rand_ce_loss": rand_ce,
            "thin_kd_loss": kd_thin,
            "rand_kd_loss": rand_kd,
            "hidden_loss": hidden,
            "full_nll_per_token": full_nll,
            "thin_nll_per_token": thin_nll,
            "rand_nll_per_token": rand_nll,
            "full_thin_nll_gap": thin_nll - full_nll,
            "nll_per_token": thin_nll,
            "step_time": timer.elapsed,
            "tokens_per_sec": tokens.numel() / max(timer.elapsed, 1e-9),
            "peak_vram": maybe_peak_memory(self.device),
            "stored_params": recursive_param_count(self.config.model),
            "tokens_seen": self.tokens_seen,
            "lr": current_lr,
            "full_active_touches_per_token": full.meta.active_touches.mean()
            if full.meta.active_touches is not None
            else 0.0,
            "thin_active_touches_per_token": thin.meta.active_touches.mean()
            if thin.meta.active_touches is not None
            else 0.0,
            "avg_depth": thin.meta.depths.float().mean() if thin.meta.depths is not None else 0.0,
            "full_recipe_schedule": full_schedule,
            "thin_recipe_schedule": target_schedule,
            "sandwich_random_paths": tr.sandwich_random_paths,
            "unique_random_recipes_per_batch": random_schedules.unique().numel()
            if random_schedules.numel()
            else 0.0,
            "sandwich_temperature": tr.sandwich_temperature,
            **self._accum_metrics(tokens, stepped),
            **self.last_grad_metrics,
        }
        self.tokens_seen += int(tokens.numel())
        self._maybe_log(metrics)
        return TrainStepResult(loss=loss, model_output=thin, metrics=metrics)

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
        if mode == "recursive_deferred_grouped_exact":
            return self.train_step_recursive_deferred_grouped_exact(batch)
        if mode == "recursive_exact_route_em":
            return self.train_step_recursive_exact_route_em(batch)
        if mode == "recursive_exact_factorized_soft":
            return self.train_step_recursive_exact_factorized_soft(batch)
        if mode == "recursive_exact_dense_hidden_distill":
            return self.train_step_recursive_exact_dense_hidden_distill(batch)
        if mode == "recursive_sandwich_supernet":
            return self.train_step_recursive_sandwich_supernet(batch)
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
