from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig, OutputConfig
from recursive_training_engine.layers import BankedRecursiveCore, RMSNorm, TransformerBlock
from recursive_training_engine.macro import MacroOperators, MacroTrace
from recursive_training_engine.output import ShortlistHead, ShortlistResult
from recursive_training_engine.recipes import RecipeBank, factor_slot_counts
from recursive_training_engine.routing import Router, RouterOutput


@dataclass(slots=True)
class ModelMeta:
    router: RouterOutput | None = None
    recipe_ids: torch.Tensor | None = None
    recipe_schedule: torch.Tensor | None = None
    depths: torch.Tensor | None = None
    h0: torch.Tensor | None = None
    recurrent_hidden: torch.Tensor | None = None
    hidden: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    states: dict[int, torch.Tensor] | None = None
    macro_trace: MacroTrace | None = None
    shortlist: ShortlistResult | None = None
    active_touches: torch.Tensor | None = None
    factor_route_tuples: torch.Tensor | None = None
    factor_route_weights: torch.Tensor | None = None


@dataclass(slots=True)
class ModelOutput:
    loss: torch.Tensor | None
    loss_per_sample: torch.Tensor | None
    logits: torch.Tensor | None
    meta: ModelMeta


def load_compatible_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    current = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    mismatched: list[str] = []
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(key)
            continue
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape != value.shape:
            mismatched.append(key)
            continue
        loaded[key] = value
    result = model.load_state_dict(loaded, strict=False)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "skipped": skipped,
        "mismatched": mismatched,
    }


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
        return_states: bool = False,
    ) -> ModelOutput:
        x = self.embed(tokens)
        states: dict[int, torch.Tensor] = {}
        for idx, block in enumerate(self.blocks, start=1):
            x = block(x)
            if return_states:
                states[idx] = x.clone()
        hidden = self.final_norm(x)
        logits = hidden @ self.vocab_weight.t()
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        return ModelOutput(
            loss=loss,
            loss_per_sample=loss_per_sample,
            logits=logits,
            meta=ModelMeta(hidden=hidden, logits=logits, states=states if return_states else None),
        )


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
        self._coda_hidden_fast = None
        self._coda_logits_fast = None

    @property
    def vocab_weight(self) -> torch.Tensor:
        return self.embed.weight if self.lm_head is None else self.lm_head.weight

    def compile_hot_paths(self, *, mode: str = "reduce-overhead") -> None:
        if not hasattr(torch, "compile"):
            return
        self._prelude_fast = torch.compile(self._prelude_impl, mode=mode)
        self._coda_hidden_fast = torch.compile(self._coda_hidden_impl, mode=mode)
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
        hidden = self._coda_hidden_impl(h)
        logits = hidden @ self.vocab_weight.t()
        return hidden, logits

    def _coda_hidden_impl(self, h: torch.Tensor) -> torch.Tensor:
        z = self.coda(h)
        return self.final_norm(z)

    def _coda_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if self._coda_hidden_fast is not None:
            return self._coda_hidden_fast(h)
        return self._coda_hidden_impl(h)

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
        fixed_recipe_schedule: list[int] | tuple[int, ...] | None = None,
        recipe_schedule: torch.Tensor | None = None,
        fixed_depth: int | None = None,
        state_depths: list[int] | tuple[int, ...] | None = None,
    ) -> ModelOutput:
        h0 = self._prelude(tokens)
        return self.forward_exact_from_h0(
            h0,
            targets,
            return_states=return_states,
            return_loss_per_sample=return_loss_per_sample,
            router_decisions=router_decisions,
            fixed_recipe=fixed_recipe,
            fixed_recipe_schedule=fixed_recipe_schedule,
            recipe_schedule=recipe_schedule,
            fixed_depth=fixed_depth,
            state_depths=state_depths,
        )

    def _validate_fixed_recipe_schedule(
        self,
        fixed_recipe_schedule: list[int] | tuple[int, ...] | None,
    ) -> tuple[int, ...] | None:
        if fixed_recipe_schedule is None:
            return None
        schedule = tuple(int(recipe_id) for recipe_id in fixed_recipe_schedule)
        if len(schedule) < self.config.t_max:
            raise ValueError(
                f"fixed_recipe_schedule length {len(schedule)} is shorter than t_max={self.config.t_max}"
            )
        for recipe_id in schedule[: self.config.t_max]:
            if recipe_id < 0 or recipe_id >= self.config.recipe_count:
                raise ValueError(
                    f"fixed_recipe_schedule recipe {recipe_id} is outside [0, {self.config.recipe_count})"
                )
        return schedule

    def _validate_recipe_schedule_tensor(
        self,
        recipe_schedule: torch.Tensor | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if recipe_schedule is None:
            return None
        if recipe_schedule.ndim != 2:
            raise ValueError("recipe_schedule must have shape [batch, t_max]")
        if recipe_schedule.shape[0] != batch_size:
            raise ValueError(
                f"recipe_schedule batch {recipe_schedule.shape[0]} does not match h0 batch {batch_size}"
            )
        if recipe_schedule.shape[1] < self.config.t_max:
            raise ValueError(
                f"recipe_schedule length {recipe_schedule.shape[1]} is shorter than t_max={self.config.t_max}"
            )
        schedule = recipe_schedule[:, : self.config.t_max].long()
        if bool((schedule < 0).any().item()) or bool((schedule >= self.config.recipe_count).any().item()):
            raise ValueError(
                f"recipe_schedule entries must be in [0, {self.config.recipe_count})"
            )
        return schedule

    def _factorized_topk_routes(
        self,
        route: RouterOutput,
        *,
        pass_idx: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = [
            route.attn_bank_logits_by_pass,
            route.ffn_bank_logits_by_pass,
            route.head_slot_logits_by_pass,
            route.ffn_slot_logits_by_pass,
        ]
        if any(value is None for value in required):
            raise RuntimeError("factorized routing requires factor logits from the router")
        attn_logits = route.attn_bank_logits_by_pass[:, pass_idx, :]
        ffn_bank_logits = route.ffn_bank_logits_by_pass[:, pass_idx, :]
        head_slot_logits = route.head_slot_logits_by_pass[:, pass_idx, :]
        ffn_slot_logits = route.ffn_slot_logits_by_pass[:, pass_idx, :]
        scores = (
            attn_logits[:, :, None, None, None]
            + ffn_bank_logits[:, None, :, None, None]
            + head_slot_logits[:, None, None, :, None]
            + ffn_slot_logits[:, None, None, None, :]
        )
        flat_scores = scores.flatten(start_dim=1)
        k = min(max(1, int(top_k)), flat_scores.shape[-1])
        top_scores, flat_ids = torch.topk(flat_scores, k=k, dim=-1)
        head_slots, ffn_slots = factor_slot_counts(self.config)
        ffn_slot = flat_ids % ffn_slots
        flat_ids = flat_ids // ffn_slots
        head_slot = flat_ids % head_slots
        flat_ids = flat_ids // head_slots
        ffn_bank = flat_ids % self.config.ffn_banks
        attn_bank = flat_ids // self.config.ffn_banks
        factors = torch.stack([attn_bank, ffn_bank, head_slot, ffn_slot], dim=-1).long()
        weights = F.softmax(top_scores, dim=-1)
        return factors, weights

    def forward_exact_factorized_soft(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        top_k: int = 4,
        temperature: float = 1.0,
        fixed_depth: int | None = None,
        router_decisions: RouterOutput | None = None,
    ) -> ModelOutput:
        h0 = self._prelude(tokens)
        return self.forward_exact_factorized_soft_from_h0(
            h0,
            targets,
            return_states=return_states,
            return_loss_per_sample=return_loss_per_sample,
            top_k=top_k,
            temperature=temperature,
            fixed_depth=fixed_depth,
            router_decisions=router_decisions,
        )

    def forward_exact_factorized_soft_from_h0(
        self,
        h0: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        top_k: int = 4,
        temperature: float = 1.0,
        fixed_depth: int | None = None,
        router_decisions: RouterOutput | None = None,
    ) -> ModelOutput:
        route = router_decisions or self.router(
            h0,
            fixed_recipe=None,
            fixed_depth=fixed_depth,
            temperature=temperature,
        )
        h = h0
        states: dict[int, torch.Tensor] = {}
        factor_tuples: list[torch.Tensor] = []
        factor_weights: list[torch.Tensor] = []
        boundaries = set(self.config.depth_choices)
        for t in range(self.config.t_max):
            active = route.depth > t
            candidates, weights = self._factorized_topk_routes(route, pass_idx=t, top_k=top_k)
            base_h = h
            delta = torch.zeros_like(h)
            for idx in range(candidates.shape[1]):
                factors_i = candidates[:, idx, :]
                h_i = self.core.forward_step_factorized(
                    base_h,
                    h0,
                    factors_i[:, 0],
                    factors_i[:, 1],
                    factors_i[:, 2],
                    factors_i[:, 3],
                    active,
                    pass_idx=t,
                )
                delta = delta + weights[:, idx].view(-1, 1, 1) * (h_i - base_h)
            h = base_h + delta
            factor_tuples.append(candidates)
            factor_weights.append(weights)
            boundary = t + 1
            if return_states and boundary in boundaries:
                states[boundary] = h.clone()
        hidden, logits = self._coda_logits(h)
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        active_by_pass = torch.arange(self.config.t_max, device=route.depth.device).unsqueeze(
            0
        ) < route.depth.unsqueeze(1)
        sparse_touch = self.recipe_bank.active_touches_for_spec(self.recipe_bank.recipes[1])
        head_slots, ffn_slots = factor_slot_counts(self.config)
        factor_route_count = self.config.attn_banks * self.config.ffn_banks * head_slots * ffn_slots
        effective_top_k = max(1, min(int(top_k), factor_route_count))
        active_passes = active_by_pass.to(h0.dtype).sum(dim=1).clamp_min(1.0)
        active_touches = (
            active_by_pass.to(h0.dtype).sum(dim=1)
            * float(effective_top_k)
            * sparse_touch
            / active_passes
        )
        meta = ModelMeta(
            router=route,
            depths=route.depth,
            h0=h0,
            recurrent_hidden=h,
            hidden=hidden,
            logits=logits,
            states=states if return_states else None,
            active_touches=active_touches,
            factor_route_tuples=torch.stack(factor_tuples, dim=1),
            factor_route_weights=torch.stack(factor_weights, dim=1),
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)

    def forward_exact_from_h0(
        self,
        h0: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        router_decisions: RouterOutput | None = None,
        fixed_recipe: int | None = None,
        fixed_recipe_schedule: list[int] | tuple[int, ...] | None = None,
        recipe_schedule: torch.Tensor | None = None,
        fixed_depth: int | None = None,
        state_depths: list[int] | tuple[int, ...] | None = None,
    ) -> ModelOutput:
        if fixed_recipe_schedule is not None and recipe_schedule is not None:
            raise ValueError("Use either fixed_recipe_schedule or recipe_schedule, not both")
        schedule = self._validate_fixed_recipe_schedule(fixed_recipe_schedule)
        schedule_tensor = self._validate_recipe_schedule_tensor(
            recipe_schedule,
            batch_size=h0.shape[0],
        )
        route = self._route(
            h0,
            fixed_recipe=fixed_recipe,
            fixed_depth=fixed_depth,
            router_decisions=router_decisions,
        )
        h = h0
        states: dict[int, torch.Tensor] = {}
        boundaries = set(self.config.depth_choices if state_depths is None else state_depths)
        for t in range(self.config.t_max):
            active = route.depth > t
            recipe_id_t = route.recipe_id
            if schedule is not None:
                recipe_id_t = torch.full_like(route.recipe_id, schedule[t])
            elif schedule_tensor is not None:
                recipe_id_t = schedule_tensor[:, t].to(route.recipe_id.device)
            elif route.recipe_id_by_pass is not None:
                recipe_id_t = route.recipe_id_by_pass[:, t]
            h = self.core.forward_step(h, h0, recipe_id_t, active, pass_idx=t)
            boundary = t + 1
            if return_states and boundary in boundaries:
                states[boundary] = h.clone()
        active_touches = self.active_touch_table[route.recipe_id]
        active_by_pass = torch.arange(self.config.t_max, device=route.depth.device).unsqueeze(
            0
        ) < route.depth.unsqueeze(1)
        if schedule_tensor is not None:
            touch_values = self.active_touch_table[schedule_tensor.to(route.depth.device)]
            weights = active_by_pass.to(touch_values.dtype)
            active_touches = (weights * touch_values).sum(dim=1) / weights.sum(dim=1).clamp_min(
                1.0
            )
        elif schedule is not None:
            schedule_ids = torch.tensor(
                schedule[: self.config.t_max],
                dtype=torch.long,
                device=route.depth.device,
            )
            touch_values = self.active_touch_table[schedule_ids]
            weights = active_by_pass.to(touch_values.dtype)
            active_touches = (weights * touch_values.unsqueeze(0)).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
        elif route.recipe_id_by_pass is not None:
            touch_values = self.active_touch_table[route.recipe_id_by_pass]
            weights = active_by_pass.to(touch_values.dtype)
            active_touches = (weights * touch_values).sum(dim=1) / weights.sum(dim=1).clamp_min(
                1.0
            )
        meta_schedule = schedule_tensor
        if meta_schedule is None and schedule is not None:
            meta_schedule = torch.tensor(
                schedule[: self.config.t_max],
                dtype=torch.long,
                device=route.depth.device,
            ).unsqueeze(0).expand(h0.shape[0], -1)
        hidden, logits = self._coda_logits(h)
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        meta = ModelMeta(
            router=route,
            recipe_ids=meta_schedule
            if meta_schedule is not None
            else (route.recipe_id_by_pass if route.recipe_id_by_pass is not None else route.recipe_id),
            recipe_schedule=meta_schedule,
            depths=route.depth,
            h0=h0,
            recurrent_hidden=h,
            hidden=hidden,
            logits=logits,
            states=states if return_states else None,
            active_touches=active_touches,
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)

    def forward_deferred_grouped_exact(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        fixed_recipe: int | None = None,
        fixed_recipe_schedule: list[int] | tuple[int, ...] | None = None,
        fixed_depth: int | None = None,
        state_depths: list[int] | tuple[int, ...] | None = None,
    ) -> ModelOutput:
        h0 = self._prelude(tokens)
        return self.forward_deferred_grouped_exact_from_h0(
            h0,
            targets,
            return_states=return_states,
            return_loss_per_sample=return_loss_per_sample,
            fixed_recipe=fixed_recipe,
            fixed_recipe_schedule=fixed_recipe_schedule,
            fixed_depth=fixed_depth,
            state_depths=state_depths,
        )

    def forward_deferred_grouped_exact_from_h0(
        self,
        h0: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_states: bool = False,
        return_loss_per_sample: bool = False,
        *,
        fixed_recipe: int | None = None,
        fixed_recipe_schedule: list[int] | tuple[int, ...] | None = None,
        fixed_depth: int | None = None,
        state_depths: list[int] | tuple[int, ...] | None = None,
    ) -> ModelOutput:
        route = self._route(h0, fixed_recipe=fixed_recipe, fixed_depth=fixed_depth)
        depth = int(fixed_depth or self.config.t_max)
        depth = max(1, min(depth, self.config.t_max))
        if fixed_recipe_schedule is None:
            recipe = int(fixed_recipe if fixed_recipe is not None else 1)
            schedule = [recipe for _ in range(depth)]
        else:
            schedule = [int(recipe_id) for recipe_id in fixed_recipe_schedule[:depth]]
            if not schedule:
                schedule = [int(fixed_recipe if fixed_recipe is not None else 1)]
            while len(schedule) < depth:
                schedule.append(schedule[-1])
        for recipe_id in schedule:
            if recipe_id < 0 or recipe_id >= self.config.recipe_count:
                raise ValueError(
                    f"deferred grouped recipe {recipe_id} is outside [0, {self.config.recipe_count})"
                )

        active = route.depth > 0
        h = self.core.deferred_grouped_block(h0, h0, schedule, active)
        states: dict[int, torch.Tensor] = {}
        boundaries = set(self.config.depth_choices if state_depths is None else state_depths)
        if return_states:
            for boundary in boundaries:
                if 1 <= int(boundary) <= depth:
                    states[int(boundary)] = h.clone()

        active_by_micro = torch.arange(depth, device=route.depth.device).unsqueeze(0) < route.depth.clamp(
            max=depth
        ).unsqueeze(1)
        schedule_ids = torch.tensor(schedule, dtype=torch.long, device=route.depth.device)
        touch_values = self.active_touch_table[schedule_ids]
        weights = active_by_micro.to(touch_values.dtype)
        active_touches = (weights * touch_values.unsqueeze(0)).sum(dim=1)
        meta_schedule = schedule_ids.unsqueeze(0).expand(h0.shape[0], -1)

        hidden, logits = self._coda_logits(h)
        loss_per_sample = lm_loss_per_sample(logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        meta = ModelMeta(
            router=route,
            recipe_ids=meta_schedule,
            recipe_schedule=meta_schedule,
            depths=route.depth.clamp(max=depth),
            h0=h0,
            recurrent_hidden=h,
            hidden=hidden,
            logits=logits,
            states=states if return_states else None,
            active_touches=active_touches,
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)

    def forward_exact_subset(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        audit_mask: torch.Tensor,
        *,
        reuse_router_decisions: RouterOutput,
        cached_h0: torch.Tensor | None = None,
        return_states: bool = True,
        return_loss_per_sample: bool = True,
    ) -> ModelOutput:
        if not audit_mask.any():
            empty = tokens.new_zeros((0,), dtype=torch.float32)
            return ModelOutput(loss=empty.sum(), loss_per_sample=empty, logits=None, meta=ModelMeta())
        subset_tokens = tokens[audit_mask]
        subset_targets = targets[audit_mask]

        def subset_optional(value: torch.Tensor | None) -> torch.Tensor | None:
            return value[audit_mask] if value is not None else None

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
            recipe_logits_by_pass=reuse_router_decisions.recipe_logits_by_pass[audit_mask]
            if reuse_router_decisions.recipe_logits_by_pass is not None
            else None,
            recipe_probs_by_pass=reuse_router_decisions.recipe_probs_by_pass[audit_mask]
            if reuse_router_decisions.recipe_probs_by_pass is not None
            else None,
            recipe_id_by_pass=reuse_router_decisions.recipe_id_by_pass[audit_mask]
            if reuse_router_decisions.recipe_id_by_pass is not None
            else None,
            recipe_onehot_st_by_pass=reuse_router_decisions.recipe_onehot_st_by_pass[audit_mask]
            if reuse_router_decisions.recipe_onehot_st_by_pass is not None
            else None,
            attn_bank_logits_by_pass=subset_optional(reuse_router_decisions.attn_bank_logits_by_pass),
            attn_bank_probs_by_pass=subset_optional(reuse_router_decisions.attn_bank_probs_by_pass),
            attn_bank_id_by_pass=subset_optional(reuse_router_decisions.attn_bank_id_by_pass),
            ffn_bank_logits_by_pass=subset_optional(reuse_router_decisions.ffn_bank_logits_by_pass),
            ffn_bank_probs_by_pass=subset_optional(reuse_router_decisions.ffn_bank_probs_by_pass),
            ffn_bank_id_by_pass=subset_optional(reuse_router_decisions.ffn_bank_id_by_pass),
            head_slot_logits_by_pass=subset_optional(reuse_router_decisions.head_slot_logits_by_pass),
            head_slot_probs_by_pass=subset_optional(reuse_router_decisions.head_slot_probs_by_pass),
            head_slot_id_by_pass=subset_optional(reuse_router_decisions.head_slot_id_by_pass),
            ffn_slot_logits_by_pass=subset_optional(reuse_router_decisions.ffn_slot_logits_by_pass),
            ffn_slot_probs_by_pass=subset_optional(reuse_router_decisions.ffn_slot_probs_by_pass),
            ffn_slot_id_by_pass=subset_optional(reuse_router_decisions.ffn_slot_id_by_pass),
            factor_entropy=subset_optional(reuse_router_decisions.factor_entropy),
        )
        if cached_h0 is not None:
            return self.forward_exact_from_h0(
                cached_h0[audit_mask],
                subset_targets,
                return_states=return_states,
                return_loss_per_sample=return_loss_per_sample,
                router_decisions=subset_route,
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
        shortlist_result = None
        if shortlist and targets is not None:
            hidden = self._coda_hidden(h)
            full_logits = None
            shortlist_result = self.shortlist_head.loss(hidden, targets, self.vocab_weight, seed=seed)
            loss_per_sample = shortlist_result.loss_per_sample
            logits = shortlist_result.logits
        else:
            hidden, full_logits = self._coda_logits(h)
            logits = full_logits
            loss_per_sample = lm_loss_per_sample(full_logits, targets) if targets is not None else None
        loss = loss_per_sample.mean() if loss_per_sample is not None else None
        if not return_loss_per_sample:
            loss_per_sample = None
        meta = ModelMeta(
            router=route,
            recipe_ids=route.recipe_id,
            depths=route.depth,
            h0=h0,
            recurrent_hidden=h,
            hidden=hidden,
            logits=full_logits,
            states=trace.states if return_states else None,
            macro_trace=trace,
            shortlist=shortlist_result,
            active_touches=self.active_touch_table[route.recipe_id],
        )
        return ModelOutput(loss=loss, loss_per_sample=loss_per_sample, logits=logits, meta=meta)
