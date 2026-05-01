from __future__ import annotations

from recursive_training_engine.ablations import build_ablation_configs
from recursive_training_engine.config import load_config


def test_ablation_matrix_contains_required_categories() -> None:
    specs = build_ablation_configs(load_config("configs/tiny.yaml"))
    names = {spec.name for spec in specs}
    assert "dense_baseline" in names
    assert "recursive_exact_dense_fallback" in names
    assert "recursive_macro_shortlist" in names
    assert "fixed_recipe" in names
    assert "strides_1_2_4" in names
    assert "shortlist_without_audit_correction" in names
    assert "shortlist_with_audit_correction" in names
    assert "active_budget_N_over_500" in names
    categories = {spec.category for spec in specs}
    assert categories == {"topology", "routing", "macro", "output", "active_budget"}
