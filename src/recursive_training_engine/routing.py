from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig


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
        recipe_probs = F.softmax(recipe_logits, dim=-1)
        depth_probs = F.softmax(depth_logits, dim=-1)
        if fixed_recipe is None:
            recipe_id = recipe_probs.argmax(dim=-1)
        else:
            recipe_id = torch.full(
                (h0.shape[0],), fixed_recipe, device=h0.device, dtype=torch.long
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
        depth_hard = F.one_hot(depth_id, depth_probs.shape[-1]).to(depth_probs.dtype)
        recipe_onehot_st = recipe_hard + recipe_probs - recipe_probs.detach()
        depth_onehot_st = depth_hard + depth_probs - depth_probs.detach()
        depth_values = self.depth_values.to(h0.device).to(depth_probs.dtype)
        depth = self.depth_values.to(h0.device)[depth_id]
        expected_depth = (depth_probs * depth_values).sum(dim=-1)
        recipe_entropy = -(recipe_probs.clamp_min(1e-9).log() * recipe_probs).sum(dim=-1)
        depth_entropy = -(depth_probs.clamp_min(1e-9).log() * depth_probs).sum(dim=-1)
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
        )
