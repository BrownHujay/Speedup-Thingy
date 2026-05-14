from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig
from recursive_training_engine.recipes import factor_slot_counts


@dataclass(slots=True)
class RouterOutput:
    recipe_logits: torch.Tensor
    recipe_probs: torch.Tensor
    recipe_id: torch.Tensor
    recipe_onehot_st: torch.Tensor
    depth_logits: torch.Tensor
    depth_probs: torch.Tensor
    depth_id: torch.Tensor
    depth: torch.Tensor
    depth_onehot_st: torch.Tensor
    router_entropy: torch.Tensor
    recipe_entropy: torch.Tensor
    depth_entropy: torch.Tensor
    expected_depth: torch.Tensor
    recipe_logits_by_pass: torch.Tensor | None = None
    recipe_probs_by_pass: torch.Tensor | None = None
    recipe_id_by_pass: torch.Tensor | None = None
    recipe_onehot_st_by_pass: torch.Tensor | None = None
    attn_bank_logits_by_pass: torch.Tensor | None = None
    attn_bank_probs_by_pass: torch.Tensor | None = None
    attn_bank_id_by_pass: torch.Tensor | None = None
    ffn_bank_logits_by_pass: torch.Tensor | None = None
    ffn_bank_probs_by_pass: torch.Tensor | None = None
    ffn_bank_id_by_pass: torch.Tensor | None = None
    head_slot_logits_by_pass: torch.Tensor | None = None
    head_slot_probs_by_pass: torch.Tensor | None = None
    head_slot_id_by_pass: torch.Tensor | None = None
    ffn_slot_logits_by_pass: torch.Tensor | None = None
    ffn_slot_probs_by_pass: torch.Tensor | None = None
    ffn_slot_id_by_pass: torch.Tensor | None = None
    factor_entropy: torch.Tensor | None = None


class Router(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.router_hidden or max(32, config.d_model)
        self.config = config
        self.recipe_mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.recipe_count),
        )
        self.pass_recipe_bias = nn.Parameter(torch.zeros(config.t_max, config.recipe_count))
        head_slots, ffn_slots = factor_slot_counts(config)
        self.head_slots = head_slots
        self.ffn_slots = ffn_slots
        self.factor_mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.pass_factor_bias = nn.Parameter(torch.zeros(config.t_max, hidden))
        self.attn_bank_head = nn.Linear(hidden, config.attn_banks)
        self.ffn_bank_head = nn.Linear(hidden, config.ffn_banks)
        self.head_slot_head = nn.Linear(hidden, head_slots)
        self.ffn_slot_head = nn.Linear(hidden, ffn_slots)
        self.depth_mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(config.depth_choices)),
        )
        self.register_buffer(
            "depth_values", torch.tensor(config.depth_choices, dtype=torch.long), persistent=False
        )

    def forward(
        self,
        h0: torch.Tensor,
        *,
        fixed_recipe: int | None = None,
        fixed_depth: int | None = None,
        temperature: float = 1.0,
    ) -> RouterOutput:
        pooled = h0.mean(dim=1)
        recipe_logits = self.recipe_mlp(pooled) / max(temperature, 1e-6)
        depth_logits = self.depth_mlp(pooled) / max(temperature, 1e-6)
        recipe_logits_by_pass = (
            recipe_logits.unsqueeze(1)
            + self.pass_recipe_bias.to(recipe_logits.device).unsqueeze(0) / max(temperature, 1e-6)
        )
        factor_hidden = self.factor_mlp(pooled).unsqueeze(1)
        factor_hidden = F.silu(factor_hidden + self.pass_factor_bias.to(h0.device).unsqueeze(0))
        inv_temperature = 1.0 / max(temperature, 1e-6)
        attn_bank_logits_by_pass = self.attn_bank_head(factor_hidden) * inv_temperature
        ffn_bank_logits_by_pass = self.ffn_bank_head(factor_hidden) * inv_temperature
        head_slot_logits_by_pass = self.head_slot_head(factor_hidden) * inv_temperature
        ffn_slot_logits_by_pass = self.ffn_slot_head(factor_hidden) * inv_temperature
        recipe_probs = F.softmax(recipe_logits, dim=-1)
        recipe_probs_by_pass = F.softmax(recipe_logits_by_pass, dim=-1)
        attn_bank_probs_by_pass = F.softmax(attn_bank_logits_by_pass, dim=-1)
        ffn_bank_probs_by_pass = F.softmax(ffn_bank_logits_by_pass, dim=-1)
        head_slot_probs_by_pass = F.softmax(head_slot_logits_by_pass, dim=-1)
        ffn_slot_probs_by_pass = F.softmax(ffn_slot_logits_by_pass, dim=-1)
        depth_probs = F.softmax(depth_logits, dim=-1)
        if fixed_recipe is None:
            recipe_id = recipe_probs.argmax(dim=-1)
            recipe_id_by_pass = recipe_probs_by_pass.argmax(dim=-1)
        else:
            recipe_id = torch.full(
                (h0.shape[0],), fixed_recipe, device=h0.device, dtype=torch.long
            )
            recipe_id_by_pass = torch.full(
                (h0.shape[0], self.config.t_max),
                fixed_recipe,
                device=h0.device,
                dtype=torch.long,
            )
        if fixed_depth is None:
            depth_id = depth_probs.argmax(dim=-1)
        else:
            matches = (self.depth_values.to(h0.device) == fixed_depth).nonzero(as_tuple=False)
            if matches.numel() == 0:
                raise ValueError(f"fixed_depth={fixed_depth} is not in depth_choices")
            depth_id = torch.full(
                (h0.shape[0],), int(matches[0].item()), device=h0.device, dtype=torch.long
            )
        recipe_hard = F.one_hot(recipe_id, recipe_probs.shape[-1]).to(recipe_probs.dtype)
        recipe_hard_by_pass = F.one_hot(recipe_id_by_pass, recipe_probs.shape[-1]).to(
            recipe_probs.dtype
        )
        depth_hard = F.one_hot(depth_id, depth_probs.shape[-1]).to(depth_probs.dtype)
        recipe_onehot_st = recipe_hard + recipe_probs - recipe_probs.detach()
        recipe_onehot_st_by_pass = (
            recipe_hard_by_pass + recipe_probs_by_pass - recipe_probs_by_pass.detach()
        )
        depth_onehot_st = depth_hard + depth_probs - depth_probs.detach()
        depth_values = self.depth_values.to(h0.device).to(depth_probs.dtype)
        depth = self.depth_values.to(h0.device)[depth_id]
        expected_depth = (depth_probs * depth_values).sum(dim=-1)
        attn_bank_id_by_pass = attn_bank_probs_by_pass.argmax(dim=-1)
        ffn_bank_id_by_pass = ffn_bank_probs_by_pass.argmax(dim=-1)
        head_slot_id_by_pass = head_slot_probs_by_pass.argmax(dim=-1)
        ffn_slot_id_by_pass = ffn_slot_probs_by_pass.argmax(dim=-1)
        recipe_entropy = -(recipe_probs.clamp_min(1e-9).log() * recipe_probs).sum(dim=-1)
        depth_entropy = -(depth_probs.clamp_min(1e-9).log() * depth_probs).sum(dim=-1)
        factor_entropy_by_pass = (
            -(attn_bank_probs_by_pass.clamp_min(1e-9).log() * attn_bank_probs_by_pass).sum(dim=-1)
            -(ffn_bank_probs_by_pass.clamp_min(1e-9).log() * ffn_bank_probs_by_pass).sum(dim=-1)
            -(head_slot_probs_by_pass.clamp_min(1e-9).log() * head_slot_probs_by_pass).sum(dim=-1)
            -(ffn_slot_probs_by_pass.clamp_min(1e-9).log() * ffn_slot_probs_by_pass).sum(dim=-1)
        )
        factor_entropy = factor_entropy_by_pass.mean(dim=-1)
        return RouterOutput(
            recipe_logits=recipe_logits,
            recipe_probs=recipe_probs,
            recipe_id=recipe_id,
            recipe_onehot_st=recipe_onehot_st,
            depth_logits=depth_logits,
            depth_probs=depth_probs,
            depth_id=depth_id,
            depth=depth,
            depth_onehot_st=depth_onehot_st,
            router_entropy=recipe_entropy + depth_entropy,
            recipe_entropy=recipe_entropy,
            depth_entropy=depth_entropy,
            expected_depth=expected_depth,
            recipe_logits_by_pass=recipe_logits_by_pass,
            recipe_probs_by_pass=recipe_probs_by_pass,
            recipe_id_by_pass=recipe_id_by_pass,
            recipe_onehot_st_by_pass=recipe_onehot_st_by_pass,
            attn_bank_logits_by_pass=attn_bank_logits_by_pass,
            attn_bank_probs_by_pass=attn_bank_probs_by_pass,
            attn_bank_id_by_pass=attn_bank_id_by_pass,
            ffn_bank_logits_by_pass=ffn_bank_logits_by_pass,
            ffn_bank_probs_by_pass=ffn_bank_probs_by_pass,
            ffn_bank_id_by_pass=ffn_bank_id_by_pass,
            head_slot_logits_by_pass=head_slot_logits_by_pass,
            head_slot_probs_by_pass=head_slot_probs_by_pass,
            head_slot_id_by_pass=head_slot_id_by_pass,
            ffn_slot_logits_by_pass=ffn_slot_logits_by_pass,
            ffn_slot_probs_by_pass=ffn_slot_probs_by_pass,
            ffn_slot_id_by_pass=ffn_slot_id_by_pass,
            factor_entropy=factor_entropy,
        )
