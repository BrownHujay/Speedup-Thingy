from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from recursive_training_engine.config import TrainingConfig
from recursive_training_engine.macro import macro_distill_loss
from recursive_training_engine.metrics import hidden_cosine, logit_kl
from recursive_training_engine.models import ModelMeta, ModelOutput, RecursiveModel
from recursive_training_engine.routing import RouterOutput


@dataclass(slots=True)
class AuditResult:
    audit_mask: torch.Tensor
    p_audit: torch.Tensor
    exact: ModelOutput | None
    corrected_loss_per_sample: torch.Tensor
    macro_aux: dict[str, torch.Tensor]
    metrics: dict[str, torch.Tensor]


class AuditEngine:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.recent_residual = 0.0
        self.recipe_audit_ema: torch.Tensor | None = None
        self.last_candidate_count = 0
        self.last_capped_fraction = 0.0
        self.last_sample_mode = "bernoulli"

    def state_dict(self) -> dict[str, object]:
        return {
            "recent_residual": self.recent_residual,
            "recipe_audit_ema": self.recipe_audit_ema,
            "last_candidate_count": self.last_candidate_count,
            "last_capped_fraction": self.last_capped_fraction,
            "last_sample_mode": self.last_sample_mode,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.recent_residual = float(state.get("recent_residual", 0.0))
        ema = state.get("recipe_audit_ema")
        self.recipe_audit_ema = ema if isinstance(ema, torch.Tensor) else None
        self.last_candidate_count = int(state.get("last_candidate_count", 0))
        self.last_capped_fraction = float(state.get("last_capped_fraction", 0.0))
        self.last_sample_mode = str(state.get("last_sample_mode", "bernoulli"))

    def _recipe_coverage_deficit(self, recipe_ids: torch.Tensor, recipe_count: int) -> torch.Tensor:
        if self.recipe_audit_ema is None or self.recipe_audit_ema.numel() != recipe_count:
            self.recipe_audit_ema = torch.zeros(recipe_count, dtype=torch.float32, device=recipe_ids.device)
        ema = self.recipe_audit_ema.to(recipe_ids.device)
        deficit = (self.config.coverage_min - ema[recipe_ids]).clamp_min(0.0)
        return deficit / max(self.config.coverage_min, 1e-12)

    def _update_audit_coverage(self, recipe_ids: torch.Tensor, mask: torch.Tensor) -> None:
        if self.recipe_audit_ema is None:
            return
        ema = self.recipe_audit_ema.to(recipe_ids.device)
        audited = recipe_ids[mask].detach().long()
        counts = torch.bincount(audited, minlength=ema.numel()).float() if audited.numel() else torch.zeros_like(ema)
        probs = counts / counts.sum().clamp_min(1.0)
        ema.mul_(self.config.coverage_beta).add_(probs, alpha=1.0 - self.config.coverage_beta)
        self.recipe_audit_ema = ema.detach()

    def compute_audit_prob(self, meta: ModelMeta) -> torch.Tensor:
        if meta.router is None or meta.recipe_ids is None:
            raise ValueError("hot meta must contain router outputs")
        router = meta.router
        uncertainty = router.router_entropy / (
            router.recipe_logits.shape[-1] + router.depth_logits.shape[-1]
        )
        residual = torch.full_like(uncertainty, self.recent_residual)
        coverage_deficit = self._recipe_coverage_deficit(
            meta.recipe_ids,
            router.recipe_logits.shape[-1],
        )
        p = (
            self.config.audit_p_min
            + self.config.audit_alpha * uncertainty
            + self.config.audit_beta * residual
            + self.config.audit_gamma * coverage_deficit
        )
        return p.clamp(self.config.audit_p_min, self.config.audit_p_max)

    def sample_mask_with_prob(self, p: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=p.device)
        gen.manual_seed(seed)
        fixed_count = self.config.audit_fixed_count_per_batch
        if fixed_count is not None:
            n = p.numel()
            k = min(int(fixed_count), n)
            mask = torch.zeros_like(p, dtype=torch.bool)
            inclusion_p = torch.zeros_like(p)
            self.last_sample_mode = "fixed_count"
            self.last_candidate_count = k
            self.last_capped_fraction = 0.0
            if k > 0 and n > 0:
                chosen = torch.randperm(n, device=p.device, generator=gen)[:k]
                mask[chosen] = True
                inclusion_p.fill_(float(k) / float(n))
            return mask, inclusion_p

        self.last_sample_mode = "bernoulli"
        mask = torch.bernoulli(p, generator=gen).bool()
        inclusion_p = p.clone()
        self.last_candidate_count = int(mask.sum().detach().cpu())
        self.last_capped_fraction = 0.0
        if self.config.audit_cap is not None and int(mask.sum()) > self.config.audit_cap:
            selected = mask.nonzero(as_tuple=False).flatten()
            perm = torch.randperm(selected.numel(), device=p.device, generator=gen)
            chosen = selected[perm[: self.config.audit_cap]]
            capped = torch.zeros_like(mask)
            capped[chosen] = True
            dropped = selected.numel() - self.config.audit_cap
            self.last_capped_fraction = float(dropped / max(selected.numel(), 1))
            inclusion_p = p * (float(self.config.audit_cap) / float(selected.numel()))
            mask = capped
        return mask, inclusion_p

    def sample_mask(self, p: torch.Tensor, seed: int) -> torch.Tensor:
        return self.sample_mask_with_prob(p, seed=seed)[0]

    def maybe_update_fixed_count_schedule(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        if not self.config.audit_schedule_enabled:
            return
        if self.config.audit_fixed_count_per_batch is None:
            return
        current = int(self.config.audit_fixed_count_per_batch)
        if current <= 0:
            return
        min_count = self.config.audit_schedule_min_count
        if min_count is None:
            min_count = max(1, batch_size // 128)
        hidden_cos = float(metrics.get("hidden_cosine_exact_macro", torch.tensor(0.0)).detach().cpu())
        residual_var = float(metrics.get("audit_residual_var", torch.tensor(float("inf"))).detach().cpu())
        loss_gap = abs(float(metrics.get("loss_residual", torch.tensor(float("inf"))).detach().cpu()))
        good = (
            loss_gap <= self.config.audit_schedule_gap_threshold
            and hidden_cos >= self.config.audit_schedule_hidden_cosine_threshold
            and residual_var <= self.config.audit_schedule_residual_var_threshold
        )
        if good:
            next_count = max(min_count, current // 2)
        else:
            next_count = min(batch_size, max(current, min(batch_size, current * 2)))
        self.config.audit_fixed_count_per_batch = next_count
        self.config.audit_fixed_count = next_count

    def run_exact_subset(
        self,
        model: RecursiveModel,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        cached_hot_meta: ModelMeta,
    ) -> ModelOutput:
        if cached_hot_meta.router is None:
            raise ValueError("cached hot meta must contain router decisions")
        return model.forward_exact_subset(
            tokens,
            targets,
            mask,
            reuse_router_decisions=cached_hot_meta.router,
            cached_h0=cached_hot_meta.h0,
            return_states=True,
            return_loss_per_sample=True,
        )

    def compute_residual_metrics(self, hot_meta: ModelMeta, exact_meta: ModelMeta) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        hot_endpoint = hot_meta.recurrent_hidden if hot_meta.recurrent_hidden is not None else hot_meta.hidden
        exact_endpoint = (
            exact_meta.recurrent_hidden if exact_meta.recurrent_hidden is not None else exact_meta.hidden
        )
        if hot_endpoint is not None and exact_endpoint is not None:
            hot_hidden = hot_endpoint
            exact_hidden = exact_endpoint
            if hot_hidden.shape[0] != exact_hidden.shape[0]:
                hot_hidden = hot_hidden[: exact_hidden.shape[0]]
            cos = hidden_cosine(hot_hidden, exact_hidden)
            mse = (hot_hidden - exact_hidden).square().mean()
            metrics["hidden_cosine"] = cos.mean()
            metrics["hidden_cosine_exact_macro"] = cos.mean()
            metrics["hidden_mse_exact_macro"] = mse
            metrics["hidden_residual_l2"] = mse.sqrt()
            self.recent_residual = float((1.0 - cos.mean()).detach().clamp_min(0).cpu())
        if hot_meta.logits is not None and exact_meta.logits is not None:
            hot_logits = hot_meta.logits
            exact_logits = exact_meta.logits
            if hot_logits.shape[0] != exact_logits.shape[0]:
                hot_logits = hot_logits[: exact_logits.shape[0]]
            if hot_logits.shape == exact_logits.shape:
                kl = logit_kl(exact_logits, hot_logits).mean()
                metrics["logit_kl"] = kl
                metrics["logit_kl_exact_macro"] = kl
        return metrics

    def correction(
        self,
        model: RecursiveModel,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        hot: ModelOutput,
        *,
        seed: int,
    ) -> AuditResult:
        if hot.loss_per_sample is None:
            raise ValueError("hot output must include per-sample losses")
        requested_p_audit = self.compute_audit_prob(hot.meta)
        mask, p_audit = self.sample_mask_with_prob(requested_p_audit, seed=seed)
        corrected = hot.loss_per_sample.clone()
        macro_aux: dict[str, torch.Tensor] = {
            "hid": hot.loss_per_sample.new_zeros(()),
            "cos": hot.loss_per_sample.new_zeros(()),
            "kl": hot.loss_per_sample.new_zeros(()),
            "norm": hot.loss_per_sample.new_zeros(()),
            "cons": hot.loss_per_sample.new_zeros(()),
        }
        metrics: dict[str, torch.Tensor] = {
            "audit_probability": p_audit.mean(),
            "audit_requested_probability": requested_p_audit.mean(),
            "audit_rate": mask.float().mean(),
            "audit_candidate_count": p_audit.new_tensor(float(self.last_candidate_count)),
            "audit_capped_fraction": p_audit.new_tensor(self.last_capped_fraction),
            "audit_fixed_count": p_audit.new_tensor(
                float(self.config.audit_fixed_count_per_batch or 0)
            ),
            "audit_count": p_audit.new_tensor(float(mask.sum().detach().cpu())),
            "audit_inclusion_prob": p_audit.mean(),
            "audit_sampler_fixed_count": p_audit.new_tensor(
                1.0 if self.last_sample_mode == "fixed_count" else 0.0
            ),
            "audit_unbiased_correction_known": p_audit.new_tensor(
                0.0
                if self.config.audit_cap is not None and self.last_sample_mode != "fixed_count"
                else 1.0
            ),
            "audit_is_gradient_corrected": p_audit.new_tensor(
                1.0 if self.config.audit_mode == "gradient_corrected" else 0.0
            ),
        }
        if self.recipe_audit_ema is not None:
            ema = self.recipe_audit_ema.to(p_audit.device)
            metrics["coverage_min_recipe_audit_ema"] = ema.min()
            metrics["coverage_max_recipe_audit_ema"] = ema.max()
        exact = None
        if mask.any():
            gradient_corrected = self.config.audit_mode == "gradient_corrected"
            context = nullcontext() if gradient_corrected else torch.no_grad()
            with context:
                exact = self.run_exact_subset(model, tokens, targets, mask, hot.meta)
            if exact.loss_per_sample is None:
                raise ValueError("exact subset must include per-sample losses")
            hot_subset = hot.loss_per_sample[mask]
            residual = exact.loss_per_sample - hot_subset
            metrics["audit_residual_mean"] = residual.mean()
            metrics["audit_residual_std"] = residual.float().std(unbiased=False).to(residual.dtype)
            metrics["audit_residual_var"] = residual.float().var(unbiased=False).to(residual.dtype)
            if self.config.audit_residual_clip is not None:
                residual = residual.clamp(
                    -self.config.audit_residual_clip,
                    self.config.audit_residual_clip,
                )
            if not gradient_corrected:
                residual = residual.detach()
            corrected_metric = hot_subset + residual.detach() / p_audit[mask].clamp_min(1e-12)
            metrics["corrected_metric_loss"] = corrected_metric.mean()
            if self.config.audit_mode == "gradient_corrected":
                corrected[mask] = hot_subset + residual / p_audit[mask].clamp_min(1e-12)
            hot_endpoint = (
                hot.meta.recurrent_hidden[mask]
                if hot.meta.recurrent_hidden is not None
                else hot.meta.hidden[mask] if hot.meta.hidden is not None else None
            )
            exact_endpoint = (
                exact.meta.recurrent_hidden
                if exact.meta.recurrent_hidden is not None
                else exact.meta.hidden
            )
            if hot_endpoint is not None and exact_endpoint is not None:
                losses = macro_distill_loss(
                    hot_endpoint,
                    exact_endpoint,
                    hot.meta.logits[mask] if hot.meta.logits is not None else None,
                    exact.meta.logits,
                    lambda_hid=self.config.effective_lambda_hid,
                    lambda_cos=self.config.effective_lambda_cos,
                    lambda_kl=self.config.effective_lambda_kl,
                    lambda_norm=self.config.lambda_norm,
                    temperature=self.config.distill_temperature,
                )
                macro_aux.update(losses)
            metrics.update(
                self.compute_residual_metrics(
                    ModelMeta(
                        recurrent_hidden=hot_endpoint,
                        logits=hot.meta.logits[mask] if hot.meta.logits is not None else None,
                    ),
                    exact.meta,
                )
            )
            metrics["loss_residual"] = (exact.loss_per_sample - hot_subset).mean()
            metrics["clipped_loss_residual"] = residual.mean()
            self.maybe_update_fixed_count_schedule(metrics, hot.loss_per_sample.shape[0])
            metrics["audit_schedule_next_count"] = p_audit.new_tensor(
                float(self.config.audit_fixed_count_per_batch or 0)
            )
        if hot.meta.recipe_ids is not None:
            self._update_audit_coverage(hot.meta.recipe_ids, mask)
        if (
            self.config.lambda_cons != 0.0
            and hot.meta.macro_trace is not None
            and hot.meta.h0 is not None
            and hot.meta.recipe_ids is not None
        ):
            macro_aux["cons"] = self.config.lambda_cons * model.macro.consistency_loss(
                hot.meta.h0, hot.meta.h0, hot.meta.recipe_ids
            )
        return AuditResult(
            audit_mask=mask,
            p_audit=p_audit,
            exact=exact,
            corrected_loss_per_sample=corrected,
            macro_aux=macro_aux,
            metrics=metrics,
        )
