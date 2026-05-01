from __future__ import annotations

import argparse
import json
import time

import torch

from recursive_training_engine.config import load_config
from recursive_training_engine.kernels import optimized
from recursive_training_engine.layers import BankedAttention, BankedSwiGLU, DenseCausalSelfAttention, DenseSwiGLU
from recursive_training_engine.models import RecursiveModel
from recursive_training_engine.audit import AuditEngine
from recursive_training_engine.output import ShortlistHead
from recursive_training_engine.recipes import RecipeBank, RecipeSpec
from recursive_training_engine.utils import default_device


def bench(fn, args, iters: int, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = default_device()
    b, s, d = cfg.training.batch_size, cfg.training.seq_len, cfg.model.d_model
    x = torch.randn(b, s, d, device=device)
    w = torch.ones(d, device=device)
    q = torch.randn(b, cfg.model.n_heads, s, d // cfg.model.n_heads, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    bank = RecipeBank(cfg.model)
    spec = bank.get_recipe(1)
    assert isinstance(spec, RecipeSpec)
    dense_attn = DenseCausalSelfAttention(cfg.model).to(device)
    dense_mlp = DenseSwiGLU(cfg.model).to(device)
    banked_attn = BankedAttention(cfg.model, bank).to(device)
    banked_mlp = BankedSwiGLU(cfg.model, bank).to(device)
    model = RecursiveModel(cfg.model, cfg.output).to(device)
    head = ShortlistHead(cfg.model.d_model, cfg.model.vocab_size, cfg.output).to(device)
    tokens = torch.randint(0, cfg.model.vocab_size, (b, s), device=device)
    targets = torch.randint(0, cfg.model.vocab_size, (b, s), device=device)
    recipe_ids = torch.ones(b, dtype=torch.long, device=device)
    depths = torch.full((b,), min(cfg.model.depth_choices[-1], cfg.model.depth_choices[0]), dtype=torch.long, device=device)
    hidden = torch.randn(b, s, d, device=device)
    hot = model.forward_macro(tokens, targets, return_states=True, return_loss_per_sample=True)
    audit = AuditEngine(cfg.training)
    audit_mask = torch.zeros(b, dtype=torch.bool, device=device)
    audit_mask[: max(1, b // 4)] = True

    def row(name, fn, fn_args, flops_per_token_est: float, bytes_per_token_est: float, launch_count_estimate: int):
        ms = bench(fn, fn_args, args.iters, device)
        tokens_per_sec = (b * s) / max(ms / 1000.0, 1e-12)
        return {
            "kernel": name,
            "device": str(device),
            "backend": optimized.backend_status()["mode"],
            "ms": ms,
            "tokens_per_sec": tokens_per_sec,
            "launch_count_estimate": launch_count_estimate,
            "memory_bandwidth_gb_s_est": (bytes_per_token_est * b * s) / max(ms / 1000.0, 1e-12) / 1e9,
            "effective_flops_per_token_est": flops_per_token_est,
            "effective_tflops_est": (flops_per_token_est * b * s) / max(ms / 1000.0, 1e-12) / 1e12,
        }

    rows = [
        row("rmsnorm", optimized.k_fused_rmsnorm, (x, w), 5 * d, 3 * d * x.element_size(), 1),
        row("dense_attention", dense_attn, (x,), 8 * d * d + 4 * s * d, 8 * d * x.element_size(), 4),
        row("dense_swiglu_mlp", dense_mlp, (x,), 6 * d * cfg.model.d_ff, 6 * d * x.element_size(), 3),
        row("banked_slab_attention", banked_attn, (x, spec), 2 * bank.active_touch_table[1].item(), 8 * d * x.element_size(), 4),
        row("banked_slab_mlp", banked_mlp, (x, spec), 2 * bank.active_touch_table[1].item(), 6 * d * x.element_size(), 3),
        row("macro_operator", model.macro, (hidden, hidden, recipe_ids, depths), 2 * d * cfg.model.macro_rank * 4, 6 * d * x.element_size(), 1),
        row("full_vocab_head", optimized.k_logits_full, (hidden, model.vocab_weight), 2 * d * cfg.model.vocab_size, d * x.element_size(), 1),
        row(
            "shortlist_head",
            lambda h, t, vw: head.loss(h, t, vw, seed=cfg.training.seed),
            (hidden, targets, model.vocab_weight),
            2 * d * cfg.output.shortlist_max_tokens,
            d * x.element_size(),
            2,
        ),
        row("audit_replay_path", audit.run_exact_subset, (model, tokens, targets, audit_mask, hot.meta), 0.0, 0.0, cfg.model.t_max),
        {"kernel": "backend_status", **optimized.backend_status()},
    ]
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
