from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_RUNS = {
    "dense_exact": "proof_d4_dense",
    "recursive_exact_teacher": "proof_d4_exact_teacher",
    "macro_distill": "proof_d4_macro_distill",
    "macro_fullaudit": "proof_d4_macro_fullaudit",
    "macro_scheduled_audit": "proof_d4_macro_sched",
    "macro_noaudit": "proof_d4_macro_noaudit",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_rows(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def summarize_run(run: Path) -> dict[str, Any]:
    manifest = read_json(run / "manifest.json")
    rows = read_rows(run)
    last = rows[-1] if rows else {}
    mode = last.get("mode") or manifest.get("mode")
    exact_keys = ("exact_eval_nll", "eval_exact_nll_per_token")
    if mode in {"dense_exact", "recursive_exact"}:
        exact_keys = ("exact_eval_nll", "eval_exact_nll_per_token", "nll_per_token")
    exact_vals = [first_present(row, *exact_keys) for row in rows]
    exact_vals = [float(v) for v in exact_vals if isinstance(v, (int, float))]
    hot_vals = [first_present(row, "hot_eval_nll", "eval_hot_nll_per_token") for row in rows]
    hot_vals = [float(v) for v in hot_vals if isinstance(v, (int, float))]
    return {
        "run": str(run),
        "manifest": manifest,
        "rows": len(rows),
        "mode": mode,
        "backend_status": manifest.get("backend_status") or manifest.get("backend"),
        "data_fingerprint": manifest.get("data_fingerprint"),
        "tokenizer": manifest.get("tokenizer"),
        "projection_lane": manifest.get("projection_lane"),
        "train_tokens": manifest.get("train_tokens"),
        "eval_tokens": manifest.get("eval_tokens"),
        "best_exact_nll": min(exact_vals) if exact_vals else None,
        "best_hot_nll": min(hot_vals) if hot_vals else None,
        "last_exact_nll": exact_vals[-1] if exact_vals else None,
        "last_hot_nll": hot_vals[-1] if hot_vals else None,
        "last_hot_exact_gap": first_present(last, "hot_exact_nll_gap"),
        "last_tokens_per_sec": last.get("tokens_per_sec"),
        "alignment": {
            "hidden_mse": first_present(last, "hidden_mse_exact_macro", "audit_hidden_mse_exact_macro"),
            "hidden_cosine": first_present(
                last,
                "hidden_cosine_exact_macro",
                "audit_hidden_cosine_exact_macro",
            ),
            "logit_kl": first_present(last, "logit_kl_exact_macro", "audit_logit_kl_exact_macro"),
            "audit_residual_var": first_present(last, "audit_residual_var", "audit_audit_residual_var"),
        },
    }


def same(values: list[Any]) -> bool:
    filtered = [v for v in values if v is not None]
    return bool(filtered) and all(v == filtered[0] for v in filtered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--exact-gap-threshold", type=float, default=0.05)
    parser.add_argument("--hot-exact-gap-threshold", type=float, default=0.05)
    parser.add_argument("--allow-reference-fallback", action="store_true")
    for key, default in REQUIRED_RUNS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    args = parser.parse_args()

    root = Path(args.runs_root)
    summaries = {
        key: summarize_run(root / getattr(args, key))
        for key in REQUIRED_RUNS
    }
    manifests = [summary["manifest"] for summary in summaries.values()]
    same_data = same([summary["data_fingerprint"] for summary in summaries.values()])
    same_tokenizer = same([summary["tokenizer"] for summary in summaries.values()])
    same_projection = same([summary["projection_lane"] for summary in summaries.values()])
    same_train_tokens = same([summary["train_tokens"] for summary in summaries.values()])
    same_eval_tokens = same([summary["eval_tokens"] for summary in summaries.values()])
    backend_fallback = any(
        bool((manifest.get("backend_status") or manifest.get("backend") or {}).get("uses_reference_fallback"))
        for manifest in manifests
    )

    dense_nll = summaries["dense_exact"]["best_exact_nll"]
    macro_exact = summaries["macro_scheduled_audit"]["best_exact_nll"]
    macro_hot = summaries["macro_scheduled_audit"]["best_hot_nll"]
    exact_gap = (
        float(macro_exact) - float(dense_nll)
        if isinstance(dense_nll, (int, float)) and isinstance(macro_exact, (int, float))
        else None
    )
    hot_exact_gap = (
        abs(float(macro_hot) - float(macro_exact))
        if isinstance(macro_hot, (int, float)) and isinstance(macro_exact, (int, float))
        else summaries["macro_scheduled_audit"]["last_hot_exact_gap"]
    )
    dense_tps = summaries["dense_exact"]["last_tokens_per_sec"]
    macro_tps = summaries["macro_scheduled_audit"]["last_tokens_per_sec"]
    speedup = (
        float(macro_tps) / float(dense_tps)
        if isinstance(dense_tps, (int, float)) and isinstance(macro_tps, (int, float)) and dense_tps
        else None
    )
    refusal_reasons = []
    if not same_data:
        refusal_reasons.append("data fingerprints differ")
    if not same_tokenizer:
        refusal_reasons.append("tokenizers differ")
    if not same_projection:
        refusal_reasons.append("projection lanes differ")
    if not same_train_tokens:
        refusal_reasons.append("train token counts differ")
    if not same_eval_tokens:
        refusal_reasons.append("eval token counts differ")
    if backend_fallback and not args.allow_reference_fallback:
        refusal_reasons.append("backend fallback is not a final kernel proof")
    if exact_gap is not None and exact_gap > args.exact_gap_threshold:
        refusal_reasons.append("macro exact NLL gap exceeds threshold")
    if macro_exact is None:
        refusal_reasons.append("missing macro exact-path NLL; hot/corrected NLL cannot prove success")
    if macro_hot is None:
        refusal_reasons.append("missing macro hot-path NLL")
    if hot_exact_gap is not None and abs(float(hot_exact_gap)) > args.hot_exact_gap_threshold:
        refusal_reasons.append("hot/exact gap exceeds threshold")

    print(
        json.dumps(
            {
                "same_data": same_data,
                "same_tokenizer": same_tokenizer,
                "same_projection_lane": same_projection,
                "same_train_tokens": same_train_tokens,
                "same_eval_tokens": same_eval_tokens,
                "backend_fallback": backend_fallback,
                "dense_exact_nll": dense_nll,
                "macro_exact_nll": macro_exact,
                "macro_hot_nll": macro_hot,
                "exact_gap_vs_dense": exact_gap,
                "hot_exact_gap": hot_exact_gap,
                "macro_speedup_vs_dense": speedup,
                "alignment_metrics": summaries["macro_scheduled_audit"]["alignment"],
                "audit_residual_trend": summaries["macro_scheduled_audit"]["alignment"].get(
                    "audit_residual_var"
                ),
                "runs": summaries,
                "success": not refusal_reasons,
                "refusal_reasons": refusal_reasons,
                "note": "local reference diagnostic only"
                if backend_fallback
                else "eligible kernel proof backend",
            },
            indent=2,
            sort_keys=True,
        )
    )
    if refusal_reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
