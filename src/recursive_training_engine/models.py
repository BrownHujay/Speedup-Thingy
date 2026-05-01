from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig, OutputConfig
from recursive_training_engine.layers import BankedRecursiveCore, RMSNorm, TransformerBlock
from recursive_training_engine.macro import MacroOperators, MacroTrace
from recursive_training_engine.output import ShortlistHead, ShortlistResult
from recursive_training_engine.recipes import RecipeBank
from recursive_training_engine.routing import Router, RouterOutput


@dataclass(slots=True)
class ModelMeta:
    router: RouterOutput | None = None
    recipe_ids: torch.Tensor | None = None
    depths: torch.Tensor | None = None
    h0: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    states: dict[int, torch.Tensor] | None = None
    macro_trace: MacroTrace | None = None
    shortlist: ShortlistResult | None = None
    active_touches: torch.Tensor | None = None


@dataclass(slots=True)
class ModelOutput:
    loss: torch.Tensor | None
    loss_per_sample: torch.Tensor | None
    logits: torch.Tensor | None
    meta: ModelMeta


def lm_loss_per_sample(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none")
    return token_loss.view_as(targets).sum(dim=-1)


class DenseModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.topology != "dense":
            raise ValueError("DenseModel requires topology='dense'")
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_dense_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None

    @property
    def vocab_weight(self) -> torch.Tensor:
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_loss_per_sample: bool = False,
    ) -> ModelOutput:
        x = self.blocks(self.embed(tokens))
        hidden = self.final_norm(x)
        logits = hidden @ self.vocab_weight.t()
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=ModelMeta(hidden=hidden, logits=logits))


class RecursiveModel(nn.Module):
    def __init__(self, config: ModelConfig, output_config: OutputConfig | None = None):
        super().__init__()
        if config.topology != "recursive":
            raise ValueError("RecursiveModel requires topology='recursive'")
        self.config = config
        self.output_config = output_config or OutputConfig()
        self.recipe_bank = RecipeBank(config)
        self.register_buffer(
            "active_touch_table", self.recipe_bank.active_touch_table, persistent=False
        )
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.prelude = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_prelude)])
        self.router = Router(config)
        self.core = BankedRecursiveCore(config, self.recipe_bank)
        self.macro = MacroOperators(config)
        self.coda = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_coda)])
        self.final_norm = RMSNorm(config.d_model)
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None
        self.shortlist_head = ShortlistHead(config.d_model, config.vocab_size, self.output_config)
        self._prelude_fast = None
        self._coda_logits_fast = None

    @property
    def vocab_weight(self) -> torch.Tensor:
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def compile_hot_paths(self, *, mode: str = "reduce-overhead") -> None:
        if not hasattr(torch, "compile"):
            return
        self._prelude_fast = torch.compile(self._prelude_impl, mode=mode)
        self._coda_logits_fast = torch.compile(self._coda_logits_impl, mode=mode)

    def _prelude_impl(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.prelude(self.embed(tokens))

    def _prelude(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._prelude_fast is not None:
            return self._prelude_fast(tokens)
        return self._prelude_impl(tokens)

    def _route(
        self,
        h0: torch.Tensor,
        *,
        fixed_recipe: int | None,
        fixed_depth: int | None,
        router_decisions: RouterOutput | None = None,
    ) -> RouterOutput:
        if router_decisions is not None:
            return router_decisions
        return self.router(h0, fixed_recipe=fixed_recipe, fixed_depth=fixed_depth)

    def _coda_logits_impl(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.coda(h)
        hidden = self.final_norm(z)
        logits = hidden @ self.vocab_weight.t()
        return hidden, logits

    def _coda_logits(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._coda_logits_fast is not None:
            return self._coda_logits_fast(h)
        return self._coda_logits_impl(h)

    def forward_exact(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        router_decisions: RouterOutput | None = None,
        fixed_recipe: int | None = None,
        fixed_depth: int | None = None,
    ) -> ModelOutput:
        h0 = self._prelude(tokens)
        route = self._route(
            h0,
            fixed_recipe=fixed_recipe,
            fixed_depth=fixed_depth,
            router_decisions=router_decisions,
        )
        h = h0
        states: dict[int, torch.Tensor] = {}
        boundaries = set(self.config.depth_choices)
        for t in range(self.config.t_max):
            active = route.depth > t
            h = self.core.forward_step(h, h0, route.recipe_id, active)
            boundary = t + 1
            if return_states and boundary in boundaries:
                states[boundary] = h.clone()
        hidden, logits = self._coda_logits(h)
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        meta = ModelMeta(
            router=route,
            recipe_ids=route.recipe_id,
            depths=route.depth,
            h0=h0,
            hidden=hidden,
            logits=logits,
            states=states if return_states else None,
            active_touches=self.active_touch_table[route.recipe_id],
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)

    def forward_exact_subset(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        audit_mask: torch.Tensor,
        *,
        reuse_router_decisions: RouterOutput,
        return_states: bool = True,
        return_loss_per_sample: bool = True,
    ) -> ModelOutput:
        if not audit_mask.any():
            empty = tokens.new_zeros((0,), dtype=torch.float32)
            return ModelOutput(loss=empty.sum(), loss_per_sample=empty, logits=None, meta=ModelMeta())
        subset_tokens = tokens[audit_mask]
        subset_targets = targets[audit_mask]
        subset_route = RouterOutput(
            recipe_logits=reuse_router_decisions.recipe_logits[audit_mask],
            recipe_probs=reuse_router_decisions.recipe_probs[audit_mask],
            recipe_id=reuse_router_decisions.recipe_id[audit_mask],
            recipe_onehot_st=reuse_router_decisions.recipe_onehot_st[audit_mask],
            depth_logits=reuse_router_decisions.depth_logits[audit_mask],
            depth_probs=reuse_router_decisions.depth_probs[audit_mask],
            depth_id=reuse_router_decisions.depth_id[audit_mask],
            depth=reuse_router_decisions.depth[audit_mask],
            depth_onehot_st=reuse_router_decisions.depth_onehot_st[audit_mask],
            router_entropy=reuse_router_decisions.router_entropy[audit_mask],
            recipe_entropy=reuse_router_decisions.recipe_entropy[audit_mask],
            depth_entropy=reuse_router_decisions.depth_entropy[audit_mask],
            expected_depth=reuse_router_decisions.expected_depth[audit_mask],
        )
        return self.forward_exact(
            subset_tokens,
            subset_targets,
            return_states=return_states,
            return_loss_per_sample=return_loss_per_sample,
            router_decisions=subset_route,
        )

    def forward_macro(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        shortlist: bool = False,
        seed: int = 0,
        fixed_recipe: int | None = None,
        fixed_depth: int | None = None,
    ) -> ModelOutput:
        h0 = self._prelude(tokens)
        route = self.router(h0, fixed_recipe=fixed_recipe, fixed_depth=fixed_depth)
        h, trace = self.macro(h0, h0, route.recipe_id, route.depth, return_states=return_states)
        hidden, full_logits = self._coda_logits(h)
        shortlist_result = None
        logits = full_logits
        if shortlist and targets is not None:
            shortlist_result = self.shortlist_head.loss(hidden, targets, self.vocab_weight, seed=seed)
            loss_per_sample = shortlist_result.loss_per_sample
            logits = shortlist_result.logits
        else:
            loss_per_sample = lm_loss_per_sample(full_logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        meta = ModelMeta(
            router=route,
            recipe_ids=route.recipe_id,
            depths=route.depth,
            h0=h0,
            hidden=hidden,
            logits=full_logits,
            states=trace.states if return_states else None,
            macro_trace=trace,
            shortlist=shortlist_result,
            active_touches=self.active_touch_table[route.recipe_id],
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)
