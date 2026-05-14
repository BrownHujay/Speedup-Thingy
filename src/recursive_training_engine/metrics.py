from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from recursive_training_engine.config import ModelConfig, TrainingConfig
from recursive_training_engine.recipes import RecipeBank


def dense_block_param_count(config: ModelConfig) -> int:
    d = config.d_model
    return 4 * d * d + 3 * d * config.d_ff + 4 * d


def dense_param_count(config: ModelConfig) -> int:
    embeddings = config.vocab_size * config.d_model
    blocks = config.n_dense_layers * dense_block_param_count(config)
    final_norm = config.d_model
    untied = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    return embeddings + blocks + final_norm + untied


def recursive_param_count(config: ModelConfig) -> int:
    embeddings = config.vocab_size * config.d_model
    dense_blocks = (config.n_prelude + config.n_coda) * dense_block_param_count(config)
    d = config.d_model
    core = config.attn_banks * 4 * d * d + config.ffn_banks * 3 * d * config.d_ff
    if config.use_global_lowrank_corrector:
        r = config.global_corrector_rank
        core += 3 * d + config.t_max * r + 3 * d * r + config.t_max * r + config.t_max
    router = 2 * (d * (config.router_hidden or max(32, d)) + (config.router_hidden or max(32, d)))
    macro = config.recipe_count * len(config.depth_choices) * (
        d + 3 * d * config.macro_rank + config.macro_rank * d + d
    )
    final_norm = d
    untied = 0 if config.tie_embeddings else config.vocab_size * d
    return embeddings + dense_blocks + core + router + macro + final_norm + untied


def solve_banks_for_fairness(
    config: ModelConfig,
    *,
    max_banks: int = 256,
) -> tuple[int, int, float]:
    target = dense_param_count(config)
    best: tuple[int, int, float] | None = None
    for attn_banks in range(1, max_banks + 1):
        for ffn_banks in range(1, max_banks + 1):
            candidate = ModelConfig(
                **{
                    **asdict(config),
                    "attn_banks": attn_banks,
                    "ffn_banks": ffn_banks,
                }
            )
            rel = abs(recursive_param_count(candidate) - target) / max(target, 1)
            if best is None or rel < best[2]:
                best = (attn_banks, ffn_banks, rel)
    assert best is not None
    return best


def active_parameter_touches(config: ModelConfig, recipe_bank: RecipeBank, recipe_ids: torch.Tensor) -> torch.Tensor:
    del config
    return recipe_bank.active_touch_table.to(recipe_ids.device)[recipe_ids]


def macro_operator_param_count(config: ModelConfig) -> int:
    d = config.d_model
    return d + 3 * d * config.macro_rank + config.macro_rank * d + d


def dense_aligned_param_count(config: ModelConfig) -> int:
    return dense_param_count(ModelConfig(**{**asdict(config), "topology": "dense"}))


def estimate_hot_active_param_equiv_per_token(
    config: ModelConfig,
    *,
    shortlist_size: torch.Tensor | float | int | None,
    include_output: bool,
) -> float:
    active = (config.n_prelude + config.n_coda) * dense_block_param_count(config)
    active += macro_operator_param_count(config)
    if include_output:
        size = config.vocab_size if shortlist_size is None else float(shortlist_size)
        active += size * config.d_model
    return float(active)


def estimate_hot_flops_per_token(active_param_equiv_per_token: float) -> float:
    return 2.0 * active_param_equiv_per_token


def router_aux_losses(
    recipe_probs: torch.Tensor,
    expected_depth: torch.Tensor,
    usage_ema: torch.Tensor,
    training: TrainingConfig,
) -> dict[str, torch.Tensor]:
    r = recipe_probs.shape[-1]
    mean_prob = recipe_probs.mean(dim=0)
    load = r * (mean_prob.square().sum()) - 1.0
    coverage = F.relu(training.coverage_min - usage_ema.to(recipe_probs.device)).square().sum()
    depth = training.lambda_depth * expected_depth.float().mean()
    return {
        "load": training.lambda_load * load,
        "cover": training.lambda_cover * coverage,
        "depth": depth,
    }


def hidden_cosine(hot: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(hot.flatten(1), exact.flatten(1), dim=-1)


def logit_kl(exact_logits: torch.Tensor, hot_logits: torch.Tensor) -> torch.Tensor:
    exact_logp = F.log_softmax(exact_logits, dim=-1)
    hot_logp = F.log_softmax(hot_logits, dim=-1)
    exact_p = exact_logp.exp()
    kl = (exact_p * (exact_logp - hot_logp)).sum(dim=-1)
    return kl.mean(dim=-1)


@dataclass(slots=True)
class FairnessReport:
    dense_params: int
    recursive_params: int
    relative_delta: float
    pass_param_count: bool
    same_tokenizer: bool
    same_data: bool
    same_seq_len: bool
    same_optimizer: bool
    same_objective: bool
    exact_path_available: bool

    @property
    def passed(self) -> bool:
        return all(
            [
                self.pass_param_count,
                self.same_tokenizer,
                self.same_data,
                self.same_seq_len,
                self.same_optimizer,
                self.same_objective,
                self.exact_path_available,
            ]
        )


def build_fairness_report(
    dense_cfg: ModelConfig,
    rec_cfg: ModelConfig,
    *,
    tolerance: float,
    same_tokenizer: bool = True,
    same_data: bool = True,
    same_seq_len: bool = True,
    same_optimizer: bool = True,
    same_objective: bool = True,
    exact_path_available: bool = True,
) -> FairnessReport:
    dense_params = dense_param_count(dense_cfg)
    rec_params = recursive_param_count(rec_cfg)
    rel = abs(rec_params - dense_params) / max(dense_params, 1)
    return FairnessReport(
        dense_params=dense_params,
        recursive_params=rec_params,
        relative_delta=rel,
        pass_param_count=rel <= tolerance,
        same_tokenizer=same_tokenizer,
        same_data=same_data,
        same_seq_len=same_seq_len,
        same_optimizer=same_optimizer,
        same_objective=same_objective,
        exact_path_available=exact_path_available,
    )
