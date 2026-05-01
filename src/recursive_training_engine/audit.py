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

    def state_dict(self) -> dict[str, object]:
        return {
            "recent_residual": self.recent_residual,
            "recipe_audit_ema": self.recipe_audit_ema,
            "last_candidate_count": self.last_candidate_count,
            "last_capped_fraction": self.last_capped_fraction,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.recent_residual = float(state.get("recent_residual", 0.0))
        ema = state.get("recipe_audit_ema")
        self.recipe_audit_ema = ema if isinstance(ema, torch.Tensor) else None
        self.last_candidate_count = int(state.get("last_candidate_count", 0))
        self.last_capped_fraction = float(state.get("last_capped_fraction", 0.0))

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

    def sample_mask(self, p: torch.Tensor, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed(seed)
        mask = torch.bernoulli(p, generator=gen).bool()
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
            mask = capped
        return mask

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
            return_states=True,
            return_loss_per_sample=True,
        )

    def compute_residual_metrics(self, hot_meta: ModelMeta, exact_meta: ModelMeta) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        if hot_meta.hidden is not None and exact_meta.hidden is not None:
            hot_hidden = hot_meta.hidden
            exact_hidden = exact_meta.hidden
            if hot_hidden.shape[0] != exact_hidden.shape[0]:
                hot_hidden = hot_hidden[: exact_hidden.shape[0]]
            cos = hidden_cosine(hot_hidden, exact_hidden)
            metrics["hidden_cosine"] = cos.mean()
            metrics["hidden_residual_l2"] = (hot_hidden - exact_hidden).square().mean().sqrt()
            self.recent_residual = float((1.0 - cos.mean()).detach().clamp_min(0).cpu())
        if hot_meta.logits is not None and exact_meta.logits is not None:
            hot_logits = hot_meta.logits
            exact_logits = exact_meta.logits
            if hot_logits.shape[0] != exact_logits.shape[0]:
                hot_logits = hot_logits[: exact_logits.shape[0]]
            if hot_logits.shape == exact_logits.shape:
                metrics["logit_kl"] = logit_kl(exact_logits, hot_logits).mean()
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
        p_audit = self.compute_audit_prob(hot.meta)
        mask = self.sample_mask(p_audit, seed=seed)
        corrected = hot.loss_per_sample.clone()
        macro_aux: dict[str, torch.Tensor] = {
            "hid": hot.loss_per_sample.new_zeros(()),
            "cos": hot.loss_per_sample.new_zeros(()),
            "kl": hot.loss_per_sample.new_zeros(()),
            "cons": hot.loss_per_sample.new_zeros(()),
        }
        metrics: dict[str, torch.Tensor] = {
            "audit_probability": p_audit.mean(),
            "audit_rate": mask.float().mean(),
            "audit_candidate_count": p_audit.new_tensor(float(self.last_candidate_count)),
            "audit_capped_fraction": p_audit.new_tensor(self.last_capped_fraction),
        }
        if self.recipe_audit_ema is not None:
            ema = self.recipe_audit_ema.to(p_audit.device)
            metrics["coverage_min_recipe_audit_ema"] = ema.min()
            metrics["coverage_max_recipe_audit_ema"] = ema.max()
        exact = None
        if mask.any():
            context = nullcontext() if self.config.audit_gradient_correction else torch.no_grad()
            with context:
                exact = self.run_exact_subset(model, tokens, targets, mask, hot.meta)
            if exact.loss_per_sample is None:
                raise ValueError("exact subset must include per-sample losses")
            hot_subset = hot.loss_per_sample[mask]
            residual = exact.loss_per_sample - hot_subset
            if self.config.audit_residual_clip is not None:
                residual = residual.clamp(
                    -self.config.audit_residual_clip,
                    self.config.audit_residual_clip,
                )
            if not self.config.audit_gradient_correction:
                residual = residual.detach()
            corrected[mask] = hot_subset + residual / p_audit[mask]
            hot_hidden = hot.meta.hidden[mask] if hot.meta.hidden is not None else None
            if hot_hidden is not None and exact.meta.hidden is not None:
                losses = macro_distill_loss(
                    hot_hidden,
                    exact.meta.hidden,
                    hot.meta.logits[mask] if hot.meta.logits is not None else None,
                    exact.meta.logits,
                    lambda_hid=self.config.lambda_hid,
                    lambda_cos=self.config.lambda_cos,
                    lambda_kl=self.config.lambda_kl,
                )
                macro_aux.update(losses)
            metrics.update(self.compute_residual_metrics(ModelMeta(hidden=hot_hidden, logits=hot.meta.logits[mask] if hot.meta.logits is not None else None), exact.meta))
            metrics["loss_residual"] = (exact.loss_per_sample - hot_subset).mean()
            metrics["clipped_loss_residual"] = residual.mean()
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
