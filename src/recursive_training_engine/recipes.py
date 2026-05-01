from __future__ import annotations

from dataclasses import dataclass

import torch

from recursive_training_engine.config import ModelConfig


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    recipe_id: int
    attention_banks: tuple[int, ...]
    ffn_banks: tuple[int, ...]
    head_groups: tuple[int, ...]
    ffn_groups: tuple[int, ...]
    dense_fallback: bool = False


class RecipeBank:
    """Static balanced recipe templates plus a dense fallback recipe."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.recipes: list[RecipeSpec] = []
        self.usage_ema = torch.zeros(config.recipe_count, dtype=torch.float32)
        self.build_static_templates()
        self.active_touch_table = self._build_active_touch_table()

    def build_static_templates(self) -> None:
        cfg = self.config
        recipes: list[RecipeSpec] = [
            RecipeSpec(
                recipe_id=0,
                attention_banks=tuple(range(cfg.attn_banks)),
                ffn_banks=tuple(range(cfg.ffn_banks)),
                head_groups=tuple(range(cfg.head_groups)),
                ffn_groups=tuple(range(cfg.ffn_groups)),
                dense_fallback=True,
            )
        ]
        sparse_count = cfg.recipe_count - 1
        for idx in range(sparse_count):
            recipe_id = idx + 1
            attn_bank = idx % cfg.attn_banks
            ffn_bank = (idx * 3) % cfg.ffn_banks
            head_start = idx % cfg.head_groups
            ffn_start = (idx * 5) % cfg.ffn_groups
            heads = tuple((head_start + j) % cfg.head_groups for j in range(cfg.active_head_groups))
            slabs = tuple((ffn_start + j) % cfg.ffn_groups for j in range(cfg.active_ffn_groups))
            recipes.append(
                RecipeSpec(
                    recipe_id=recipe_id,
                    attention_banks=(attn_bank,),
                    ffn_banks=(ffn_bank,),
                    head_groups=heads,
                    ffn_groups=slabs,
                    dense_fallback=False,
                )
            )
        self.recipes = recipes

    def get_recipe(self, recipe_id: torch.Tensor | int) -> RecipeSpec | list[RecipeSpec]:
        if isinstance(recipe_id, torch.Tensor):
            if recipe_id.ndim == 0:
                return self.recipes[int(recipe_id.item())]
            return [self.recipes[int(x)] for x in recipe_id.detach().cpu().tolist()]
        return self.recipes[int(recipe_id)]

    def dense_fallback_recipe(self) -> RecipeSpec:
        return self.recipes[0]

    def update_usage(self, recipe_ids: torch.Tensor, beta: float = 0.98) -> None:
        ids = recipe_ids.detach().cpu().long()
        counts = torch.bincount(ids, minlength=self.config.recipe_count).float()
        probs = counts / counts.sum().clamp_min(1.0)
        self.usage_ema.mul_(beta).add_(probs, alpha=1.0 - beta)

    def usage_stats(self) -> dict:
        return {
            "usage_ema": self.usage_ema.tolist(),
            "min_usage": float(self.usage_ema.min().item()),
            "max_usage": float(self.usage_ema.max().item()),
            "dense_fallback_usage": float(self.usage_ema[0].item()),
        }

    def _build_active_touch_table(self) -> torch.Tensor:
        cfg = self.config
        d = cfg.d_model
        touches = []
        for spec in self.recipes:
            attn = len(spec.attention_banks) * 4 * d * d * len(spec.head_groups) / cfg.head_groups
            ffn = len(spec.ffn_banks) * 3 * d * cfg.d_ff * len(spec.ffn_groups) / cfg.ffn_groups
            touches.append(attn + ffn)
        return torch.tensor(touches, dtype=torch.float32)

    def validate_balance(self) -> dict[str, float]:
        cfg = self.config
        attn = torch.zeros(cfg.attn_banks)
        ffn = torch.zeros(cfg.ffn_banks)
        heads = torch.zeros(cfg.head_groups)
        slabs = torch.zeros(cfg.ffn_groups)
        for spec in self.recipes[1:]:
            attn[list(spec.attention_banks)] += 1
            ffn[list(spec.ffn_banks)] += 1
            heads[list(spec.head_groups)] += 1
            slabs[list(spec.ffn_groups)] += 1
        def spread(x: torch.Tensor) -> float:
            return float((x.max() - x.min()).item())
        return {
            "attn_bank_spread": spread(attn),
            "ffn_bank_spread": spread(ffn),
            "head_group_spread": spread(heads),
            "ffn_group_spread": spread(slabs),
        }
