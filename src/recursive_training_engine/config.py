from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(slots=True)
class ModelConfig:
    topology: Literal["dense", "recursive"]
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_dense_layers: int
    n_prelude: int
    n_coda: int
    t_max: int
    depth_choices: list[int]
    attn_banks: int
    ffn_banks: int
    head_groups: int
    ffn_groups: int
    active_head_groups: int
    active_ffn_groups: int
    recipe_count: int
    tie_embeddings: bool = True
    use_rope: bool = True
    use_recursive_input_skip: bool = True
    fairness_tolerance: float = 0.01
    macro_rank: int = 32
    macro_update_scale: float = 0.05
    macro_decomposition: Literal["direct", "binary", "greedy", "consistency_tree"] = "direct"
    router_hidden: int | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.head_groups != 0:
            raise ValueError("n_heads must be divisible by head_groups")
        if self.d_ff % self.ffn_groups != 0:
            raise ValueError("d_ff must be divisible by ffn_groups")
        if self.active_head_groups < 1 or self.active_head_groups > self.head_groups:
            raise ValueError("active_head_groups must be in [1, head_groups]")
        if self.active_ffn_groups < 1 or self.active_ffn_groups > self.ffn_groups:
            raise ValueError("active_ffn_groups must be in [1, ffn_groups]")
        if self.recipe_count < 2:
            raise ValueError("recipe_count must include sparse recipes plus dense fallback")
        if self.t_max not in self.depth_choices:
            raise ValueError("t_max must be present in depth_choices")
        if sorted(self.depth_choices) != self.depth_choices:
            raise ValueError("depth_choices must be sorted ascending")


@dataclass(slots=True)
class TrainingConfig:
    mode: Literal[
        "dense_exact",
        "recursive_exact",
        "recursive_macro",
        "recursive_macro_shortlist",
    ]
    loss_normalization: Literal["token_mean", "sample_sum"] = "token_mean"
    optimizer: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 4
    seq_len: int = 32
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    seed: int = 1337
    precision: Literal["fp32", "bf16"] = "fp32"
    audit_p_min: float = 0.05
    audit_p_max: float = 0.25
    audit_alpha: float = 0.08
    audit_beta: float = 0.08
    audit_gamma: float = 0.08
    audit_cap: int | None = None
    audit_residual_clip: float | None = None
    audit_gradient_correction: bool = True
    target_speedup_vs_dense: float | None = None
    max_active_param_equiv_per_token: float | None = None
    max_hotpath_flops_per_token: float | None = None
    lambda_depth: float = 1e-4
    lambda_load: float = 1e-2
    lambda_cover: float = 1e-2
    lambda_hid: float = 1e-2
    lambda_cos: float = 1e-2
    lambda_kl: float = 1e-2
    lambda_cons: float = 1e-3
    coverage_min: float = 0.01
    coverage_beta: float = 0.98
    fixed_recipe: int | None = None
    fixed_depth: int | None = None
    log_every: int = 1
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    fused_optimizer: bool = True
    foreach_optimizer: bool = True
    allow_tf32: bool = True
    strict_cuda: bool = False
    require_triton: bool = False
    require_flash_attention: bool = True

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.audit_cap is not None and self.audit_cap < 0:
            raise ValueError("audit_cap must be non-negative when provided")
        if self.target_speedup_vs_dense is not None and self.target_speedup_vs_dense <= 0:
            raise ValueError("target_speedup_vs_dense must be positive")
        if (
            self.max_active_param_equiv_per_token is not None
            and self.max_active_param_equiv_per_token <= 0
        ):
            raise ValueError("max_active_param_equiv_per_token must be positive")
        if self.max_hotpath_flops_per_token is not None and self.max_hotpath_flops_per_token <= 0:
            raise ValueError("max_hotpath_flops_per_token must be positive")


@dataclass(slots=True)
class OutputConfig:
    mode: Literal["full", "shortlist"] = "full"
    num_clusters: int = 64
    shortlist_top_clusters: int = 2
    shortlist_max_tokens: int = 128
    hard_negatives: int = 32
    random_negatives: int = 32


@dataclass(slots=True)
class DataConfig:
    dataset: Literal["tinystories", "wikitext103", "local", "synthetic"] = "synthetic"
    tokenizer: Literal["gpt2_bpe", "byte"] = "gpt2_bpe"
    cache_dir: str = ".cache/rte"
    local_text_path: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    max_tokens: int = 200_000
    eval_tokens: int = 20_000
    synthetic_tokens: int = 20_000


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    data: DataConfig = field(default_factory=DataConfig)
    run_name: str = "rte-run"
    output_dir: str = "runs"
    allow_unknown_config_keys: bool = False


def _filter_dataclass(
    cls: type,
    data: dict[str, Any],
    *,
    allow_unknown: bool,
    path: str,
) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown and not allow_unknown:
        joined = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"unknown config key(s): {joined}")
    return {k: v for k, v in data.items() if k in valid}


def _build_dataclass(cls: type, data: dict[str, Any], *, allow_unknown: bool, path: str) -> Any:
    if not is_dataclass(cls):
        raise TypeError(cls)
    return cls(**_filter_dataclass(cls, data, allow_unknown=allow_unknown, path=path))


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    allow_unknown = bool(raw.get("allow_unknown_config_keys", False))
    root_valid = {
        "model",
        "training",
        "output",
        "data",
        "run_name",
        "output_dir",
        "allow_unknown_config_keys",
    }
    root_unknown = sorted(set(raw) - root_valid)
    if root_unknown and not allow_unknown:
        joined = ", ".join(root_unknown)
        raise ValueError(f"unknown config key(s): {joined}")
    model = _build_dataclass(ModelConfig, raw["model"], allow_unknown=allow_unknown, path="model")
    training = _build_dataclass(
        TrainingConfig,
        raw["training"],
        allow_unknown=allow_unknown,
        path="training",
    )
    output = _build_dataclass(
        OutputConfig,
        raw.get("output", {}),
        allow_unknown=allow_unknown,
        path="output",
    )
    data = _build_dataclass(
        DataConfig,
        raw.get("data", {}),
        allow_unknown=allow_unknown,
        path="data",
    )
    return ExperimentConfig(
        model=model,
        training=training,
        output=output,
        data=data,
        run_name=raw.get("run_name", "rte-run"),
        output_dir=raw.get("output_dir", "runs"),
        allow_unknown_config_keys=allow_unknown,
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    import dataclasses

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dataclasses.asdict(config), sort_keys=False))
