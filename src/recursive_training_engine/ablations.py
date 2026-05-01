from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from recursive_training_engine.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    category: str
    config: ExperimentConfig


def _with_mode(base: ExperimentConfig, name: str, category: str, mode: str) -> AblationSpec:
    topology = "dense" if mode == "dense_exact" else "recursive"
    cfg = dataclasses.replace(
        base,
        run_name=f"{base.run_name}-{name}",
        model=dataclasses.replace(base.model, topology=topology),
        training=dataclasses.replace(base.training, mode=mode),
    )
    return AblationSpec(name=name, category=category, config=cfg)


def build_ablation_configs(base: ExperimentConfig) -> list[AblationSpec]:
    specs: list[AblationSpec] = [
        _with_mode(base, "dense_baseline", "topology", "dense_exact"),
        AblationSpec(
            "recursive_exact_dense_fallback",
            "topology",
            dataclasses.replace(
                base,
                run_name=f"{base.run_name}-recursive_exact_dense_fallback",
                model=dataclasses.replace(base.model, topology="recursive"),
                training=dataclasses.replace(base.training, mode="recursive_exact", fixed_recipe=0),
            ),
        ),
        _with_mode(base, "recursive_exact_sparse", "topology", "recursive_exact"),
        _with_mode(base, "recursive_macro_sparse", "topology", "recursive_macro"),
        _with_mode(base, "recursive_macro_shortlist", "topology", "recursive_macro_shortlist"),
        AblationSpec(
            "fixed_recipe",
            "routing",
            dataclasses.replace(
                base,
                run_name=f"{base.run_name}-fixed_recipe",
                model=dataclasses.replace(base.model, topology="recursive"),
                training=dataclasses.replace(base.training, mode="recursive_exact", fixed_recipe=1),
            ),
        ),
        _with_mode(base, "routed_recipe", "routing", "recursive_exact"),
        AblationSpec(
            "fixed_depth",
            "routing",
            dataclasses.replace(
                base,
                run_name=f"{base.run_name}-fixed_depth",
                model=dataclasses.replace(base.model, topology="recursive"),
                training=dataclasses.replace(
                    base.training, mode="recursive_exact", fixed_depth=base.model.depth_choices[0]
                ),
            ),
        ),
        _with_mode(base, "routed_depth", "routing", "recursive_exact"),
        AblationSpec(
            "dense_fallback_recipe_only",
            "routing",
            dataclasses.replace(
                base,
                run_name=f"{base.run_name}-dense_fallback_recipe_only",
                model=dataclasses.replace(base.model, topology="recursive"),
                training=dataclasses.replace(base.training, mode="recursive_exact", fixed_recipe=0),
            ),
        ),
        _with_mode(base, "exact_recurrence_only", "macro", "recursive_exact"),
    ]
    for name, choices in [
        ("strides_1_2_4", [1, 2, 4]),
        ("strides_1_2_4_8_16", [1, 2, 4, 8, 16]),
        ("full_stride_set", base.model.depth_choices),
    ]:
        filtered = [x for x in choices if x <= base.model.t_max]
        if not filtered:
            filtered = [base.model.depth_choices[0]]
        cfg = dataclasses.replace(
            base,
            run_name=f"{base.run_name}-{name}",
            model=dataclasses.replace(
                base.model,
                topology="recursive",
                depth_choices=filtered,
                t_max=max(filtered),
            ),
            training=dataclasses.replace(base.training, mode="recursive_macro"),
        )
        specs.append(AblationSpec(name, "macro", cfg))
    specs.extend(
        [
            AblationSpec(
                "exact_full_vocab",
                "output",
                dataclasses.replace(
                    base,
                    run_name=f"{base.run_name}-exact_full_vocab",
                    model=dataclasses.replace(base.model, topology="recursive"),
                    training=dataclasses.replace(base.training, mode="recursive_macro"),
                    output=dataclasses.replace(base.output, mode="full"),
                ),
            ),
            AblationSpec(
                "shortlist_without_audit_correction",
                "output",
                dataclasses.replace(
                    base,
                    run_name=f"{base.run_name}-shortlist_without_audit_correction",
                    model=dataclasses.replace(base.model, topology="recursive"),
                    training=dataclasses.replace(
                        base.training,
                        mode="recursive_macro_shortlist",
                        audit_p_min=0.0,
                        audit_p_max=0.0,
                    ),
                    output=dataclasses.replace(base.output, mode="shortlist"),
                ),
            ),
            AblationSpec(
                "shortlist_with_audit_correction",
                "output",
                dataclasses.replace(
                    base,
                    run_name=f"{base.run_name}-shortlist_with_audit_correction",
                    model=dataclasses.replace(base.model, topology="recursive"),
                    training=dataclasses.replace(base.training, mode="recursive_macro_shortlist"),
                    output=dataclasses.replace(base.output, mode="shortlist"),
                ),
            ),
        ]
    )
    for speedup in [50, 100, 250, 500]:
        specs.append(
            AblationSpec(
                f"active_budget_N_over_{speedup}",
                "active_budget",
                dataclasses.replace(
                    base,
                    run_name=f"{base.run_name}-active_budget_N_over_{speedup}",
                    model=dataclasses.replace(base.model, topology="recursive"),
                    training=dataclasses.replace(
                        base.training,
                        mode="recursive_macro",
                        target_speedup_vs_dense=float(speedup),
                    ),
                ),
            )
        )
    return specs
