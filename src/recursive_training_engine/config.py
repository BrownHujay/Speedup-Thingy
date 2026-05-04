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
    macro_type: Literal["bounded_residual", "v2_delta_radius"] = "bounded_residual"
    macro_rank: int = 32
    macro_update_scale: float = 0.05
    macro_hidden_mult: int = 1
    macro_use_gated_update: bool = False
    macro_update_scale_init: float = 1.0
    macro_include_delta_to_h0: bool = False
    macro_use_delta_to_h0: bool = False
    macro_use_depth_embedding: bool = False
    macro_use_recipe_embedding: bool = False
    macro_radius_init_from_teacher: bool = False
    macro_radius_clamp_mult_min: float = 0.25
    macro_radius_clamp_mult_max: float = 4.0
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
        if self.macro_rank < 1:
            raise ValueError("macro_rank must be positive")
        if self.macro_hidden_mult < 1:
            raise ValueError("macro_hidden_mult must be positive")
        if self.macro_update_scale_init < 0:
            raise ValueError("macro_update_scale_init must be non-negative")
        if self.macro_radius_clamp_mult_min < 0:
            raise ValueError("macro_radius_clamp_mult_min must be non-negative")
        if self.macro_radius_clamp_mult_max <= 0:
            raise ValueError("macro_radius_clamp_mult_max must be positive")
        if self.macro_radius_clamp_mult_max < self.macro_radius_clamp_mult_min:
            raise ValueError("macro_radius_clamp_mult_max must be >= macro_radius_clamp_mult_min")


@dataclass(slots=True)
class TrainingConfig:
    mode: Literal[
        "dense_exact",
        "recursive_exact",
        "recursive_macro",
        "recursive_macro_shortlist",
        "recursive_macro_distill_only",
        "recursive_macro_lm_aligned",
        "recursive_macro_shadow_coda",
    ]
    loss_normalization: Literal["token_mean", "sample_sum"] = "token_mean"
    optimizer: Literal["adamw"] = "adamw"
    lr: float = 3e-4
    lr_base: float | None = None
    lr_macro: float | None = None
    lr_coda: float | None = None
    lr_output: float | None = None
    lr_router: float | None = None
    lr_schedule: Literal["constant", "linear_decay_after"] = "constant"
    lr_decay_start_tokens: int | None = None
    lr_decay_end_tokens: int | None = None
    lr_final_scale: float = 1.0
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
    audit_sampler: Literal["bernoulli", "fixed_count"] = "bernoulli"
    audit_fixed_count: int | None = None
    audit_cap: int | None = None
    audit_fixed_count_per_batch: int | None = None
    audit_mode: Literal["metric_only", "gradient_corrected", "distill_only"] = "gradient_corrected"
    audit_residual_clip: float | None = None
    audit_gradient_correction: bool = True
    audit_schedule_enabled: bool = False
    audit_schedule_min_count: int | None = None
    audit_schedule_gap_threshold: float = 0.05
    audit_schedule_nll_gap_threshold: float = 0.05
    audit_schedule_hidden_cosine_threshold: float = 0.98
    audit_schedule_residual_var_threshold: float = 1.0
    audit_schedule_delta_rms_ratio_min: float = 0.9
    audit_schedule_delta_rms_ratio_max: float = 1.1
    audit_schedule_hidden_mse_threshold: float = 1.0
    audit_schedule_macro_norm_threshold: float | None = None
    audit_schedule_macro_norm_threshold_mult: float = 2.0
    audit_schedule_require_negative_mse_slope: bool = True
    audit_schedule_require_negative_norm_slope: bool = True
    target_speedup_vs_dense: float | None = None
    max_active_param_equiv_per_token: float | None = None
    max_hotpath_flops_per_token: float | None = None
    lambda_depth: float = 1e-4
    lambda_load: float = 1e-2
    lambda_cover: float = 1e-2
    lambda_hid: float = 1e-2
    lambda_cos: float = 1e-2
    lambda_kl: float = 1e-2
    lambda_hidden_mse: float | None = None
    lambda_hidden_cosine: float | None = None
    lambda_logit_kl: float | None = None
    lambda_norm: float = 0.05
    lambda_delta_dir: float = 2.0
    lambda_delta_rms: float = 2.0
    lambda_endpoint_normed: float = 1.0
    lambda_endpoint_raw: float = 0.05
    macro_rms_trust_region: bool = False
    macro_rms_clamp_early: bool = False
    macro_rms_clamp_min: float = 0.5
    macro_rms_clamp_max: float = 2.0
    lambda_macro_rms_trust: float = 2.0
    distill_temperature: float = 2.0
    lambda_cons: float = 1e-3
    coverage_min: float = 0.01
    coverage_beta: float = 0.98
    fixed_recipe: int | None = None
    fixed_recipe_schedule: list[int] | None = None
    fixed_depth: int | None = None
    disable_router_aux: bool = False
    debug_force_full_output: bool = False
    coda_warmup_steps: int = 0
    aligned_lm_teacher_checkpoint: str | None = None
    aligned_lm_freeze_teacher: bool = True
    aligned_lm_phase_a_train: list[str] = field(default_factory=lambda: ["macro"])
    aligned_lm_phase_b_train: list[str] = field(default_factory=lambda: ["macro", "coda"])
    aligned_lm_phase_c_train: list[str] = field(default_factory=lambda: ["macro", "coda", "output"])
    unfreeze_prelude_core_after_gate: bool = True
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
        if self.audit_fixed_count is not None:
            self.audit_fixed_count_per_batch = self.audit_fixed_count
        if self.audit_sampler == "fixed_count" and self.audit_fixed_count_per_batch is None:
            raise ValueError("audit_sampler='fixed_count' requires audit_fixed_count")
        if self.audit_fixed_count_per_batch is not None:
            self.audit_sampler = "fixed_count"
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.coda_warmup_steps < 0:
            raise ValueError("coda_warmup_steps must be non-negative")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if self.lr_final_scale <= 0:
            raise ValueError("lr_final_scale must be positive")
        if self.lr_schedule == "linear_decay_after":
            if self.lr_decay_start_tokens is None or self.lr_decay_end_tokens is None:
                raise ValueError("linear_decay_after requires lr_decay_start_tokens and lr_decay_end_tokens")
            if self.lr_decay_end_tokens <= self.lr_decay_start_tokens:
                raise ValueError("lr_decay_end_tokens must be greater than lr_decay_start_tokens")
        if self.audit_cap is not None and self.audit_cap < 0:
            raise ValueError("audit_cap must be non-negative when provided")
        if self.audit_fixed_count_per_batch is not None:
            if self.audit_fixed_count_per_batch < 0:
                raise ValueError("audit_fixed_count_per_batch must be non-negative when provided")
            if self.audit_fixed_count_per_batch > self.batch_size:
                raise ValueError("audit_fixed_count_per_batch must be <= batch_size")
        if self.audit_schedule_min_count is not None and self.audit_schedule_min_count < 0:
            raise ValueError("audit_schedule_min_count must be non-negative when provided")
        if self.audit_schedule_delta_rms_ratio_min < 0:
            raise ValueError("audit_schedule_delta_rms_ratio_min must be non-negative")
        if self.audit_schedule_delta_rms_ratio_max < self.audit_schedule_delta_rms_ratio_min:
            raise ValueError(
                "audit_schedule_delta_rms_ratio_max must be >= audit_schedule_delta_rms_ratio_min"
            )
        if self.audit_schedule_hidden_mse_threshold < 0:
            raise ValueError("audit_schedule_hidden_mse_threshold must be non-negative")
        if (
            self.audit_schedule_macro_norm_threshold is not None
            and self.audit_schedule_macro_norm_threshold < 0
        ):
            raise ValueError("audit_schedule_macro_norm_threshold must be non-negative")
        if self.audit_schedule_macro_norm_threshold_mult < 0:
            raise ValueError("audit_schedule_macro_norm_threshold_mult must be non-negative")
        if (
            self.audit_mode == "gradient_corrected"
            and not self.audit_gradient_correction
            and self.audit_p_max > 0
        ):
            raise ValueError(
                "audit_mode='gradient_corrected' requires audit_gradient_correction=true "
                "when audits are enabled"
            )
        if self.target_speedup_vs_dense is not None and self.target_speedup_vs_dense <= 0:
            raise ValueError("target_speedup_vs_dense must be positive")
        if (
            self.max_active_param_equiv_per_token is not None
            and self.max_active_param_equiv_per_token <= 0
        ):
            raise ValueError("max_active_param_equiv_per_token must be positive")
        if self.max_hotpath_flops_per_token is not None and self.max_hotpath_flops_per_token <= 0:
            raise ValueError("max_hotpath_flops_per_token must be positive")
        if self.fixed_recipe_schedule is not None:
            if not self.fixed_recipe_schedule:
                raise ValueError("fixed_recipe_schedule must not be empty when provided")
            if any(recipe < 0 for recipe in self.fixed_recipe_schedule):
                raise ValueError("fixed_recipe_schedule entries must be non-negative")
        if self.macro_rms_clamp_min <= 0 or self.macro_rms_clamp_max <= 0:
            raise ValueError("macro_rms_clamp_min/max must be positive")
        if self.macro_rms_clamp_max < self.macro_rms_clamp_min:
            raise ValueError("macro_rms_clamp_max must be >= macro_rms_clamp_min")

    @property
    def effective_lr_base(self) -> float:
        return self.lr if self.lr_base is None else self.lr_base

    @property
    def effective_lr_macro(self) -> float:
        return self.lr if self.lr_macro is None else self.lr_macro

    @property
    def effective_lr_coda(self) -> float:
        return self.lr if self.lr_coda is None else self.lr_coda

    @property
    def effective_lr_output(self) -> float:
        return self.lr if self.lr_output is None else self.lr_output

    @property
    def effective_lr_router(self) -> float:
        return self.lr if self.lr_router is None else self.lr_router

    @property
    def effective_lambda_hid(self) -> float:
        return self.lambda_hid if self.lambda_hidden_mse is None else self.lambda_hidden_mse

    @property
    def effective_lambda_cos(self) -> float:
        return self.lambda_cos if self.lambda_hidden_cosine is None else self.lambda_hidden_cosine

    @property
    def effective_lambda_kl(self) -> float:
        return self.lambda_kl if self.lambda_logit_kl is None else self.lambda_logit_kl


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
    vocab_projection: Literal["filter", "modulo"] = "filter"
    projection_lane: Literal["filter", "modulo"] | None = None
    cache_dir: str = ".cache/rte"
    local_text_path: str | None = None
    train_split: str = "train"
    eval_split: str = "validation"
    seed: int | None = None
    train_tokens: int | None = None
    max_tokens: int = 200_000
    eval_tokens: int = 20_000
    synthetic_tokens: int = 20_000

    def __post_init__(self) -> None:
        if self.projection_lane is None:
            self.projection_lane = self.vocab_projection
        elif self.projection_lane != self.vocab_projection:
            raise ValueError("data.projection_lane and data.vocab_projection must match")
        if self.train_tokens is not None:
            self.max_tokens = self.train_tokens


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
