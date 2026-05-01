# Project Audit: Recursive Training Engine

Audit date: April 30, 2026

This audit describes the current contents of the repository as it exists on disk.
The git repository has no commits yet, and the main project files are currently
untracked, so there is no committed history to compare against. In practical
terms, this document treats the current working tree as the created artifact.

## Executive Summary

This project is a Python/PyTorch testbed for comparing a same-size dense
decoder-only Transformer with a recursive sparse Transformer that can be trained
through either exact recurrent execution or a faster macro approximation.

The package is named `recursive-training-engine`, and its CLI entry point is
`rte`. It includes:

- a dense Transformer baseline
- a recursive Transformer with prelude blocks, a banked recurrent core, and coda
  blocks
- a router that chooses sparse recipes and recurrence depths per sample
- an exact recursive path used as the source of truth
- a macro approximation path used as the hot path
- an audit engine that replays exact subsets and applies a control-variate loss
  correction
- a shortlist output head for approximate vocabulary scoring
- fairness metrics, ablation generation, CLI commands, benchmark scripts, and
  tests

The project is useful as an experimental research harness. It is not yet a final
CUDA/Triton speedup implementation: the optimized kernel module is currently a
dispatch surface backed mostly by vectorized PyTorch, and local results are from
CPU/MPS-style execution rather than confirmed CUDA/Triton runs.

## Repository State

- Current branch: `master`
- Commit history: none
- Tracked files: none yet
- Untracked project files: package source, configs, docs, tests, benchmarks,
  `pyproject.toml`, `uv.lock`, and `.gitignore`
- Ignored/generated areas: `.venv/`, `.pytest_cache/`, `.cache/`, `runs/`,
  `dist/`, egg metadata, and Python cache files

The `runs/` directory is ignored by `.gitignore`, but it exists locally and
contains prior metric logs and some empty metric placeholders.

## Created Inventory

### Root Files

| Path | Purpose |
| --- | --- |
| `.gitignore` | Ignores virtualenvs, caches, run outputs, build outputs, and Python bytecode. |
| `README.md` | Top-level project description, current local benchmark tables, interpretation, and quick commands. |
| `pyproject.toml` | Package metadata, dependencies, optional CUDA/dev extras, pytest settings, ruff settings, hatch build config, and `rte` script entry point. |
| `uv.lock` | Locked dependency graph for reproducible `uv` installs. |
| `PROJECT_AUDIT.md` | This audit document. |

### Documentation

| Path | Purpose |
| --- | --- |
| `docs/design.md` | Architecture notes covering objective/data, dense baseline, recursive model, audit correction, shortlist output, kernels, fusion, and failure handling. |
| `docs/commands.md` | Common commands for tests, fairness checks, smoke training, evaluation, kernel benchmarks, ablations, and time-to-quality checks. |

### Source Package

The import package is `recursive_training_engine` under `src/`.

| Path | Purpose |
| --- | --- |
| `src/recursive_training_engine/__init__.py` | Public package exports for configs, models, recipes, router, and train engine. |
| `src/recursive_training_engine/config.py` | Dataclass configuration schema plus YAML load/save helpers. |
| `src/recursive_training_engine/cli.py` | Implements the `rte` CLI commands. |
| `src/recursive_training_engine/data.py` | Builds deterministic synthetic or text-tokenized train/eval token streams. |
| `src/recursive_training_engine/layers.py` | RMSNorm, RoPE, dense Transformer layers, banked attention/MLP layers, and recursive core. |
| `src/recursive_training_engine/models.py` | Dense and recursive model definitions plus model output dataclasses. |
| `src/recursive_training_engine/recipes.py` | Static recipe templates, dense fallback recipe, sparse recipe balancing, and active-touch accounting. |
| `src/recursive_training_engine/routing.py` | Router MLPs for per-sample recipe and depth selection. |
| `src/recursive_training_engine/macro.py` | Low-rank macro operators, macro trace metadata, and macro distillation losses. |
| `src/recursive_training_engine/audit.py` | Exact replay subset selection and control-variate correction for macro training. |
| `src/recursive_training_engine/output.py` | Shortlist vocabulary head and shortlist loss path. |
| `src/recursive_training_engine/metrics.py` | Parameter counts, fairness report, router auxiliary losses, hidden cosine, and logit KL. |
| `src/recursive_training_engine/training.py` | `TrainEngine`, optimizer setup, precision/device setup, train steps, logging, and mode dispatch. |
| `src/recursive_training_engine/reporting.py` | JSONL logger and time-to-target summary helper. |
| `src/recursive_training_engine/ablations.py` | Programmatic ablation matrix generation. |
| `src/recursive_training_engine/utils.py` | Seeding, device selection, wall timing, batch iterator, and small tensor helpers. |
| `src/recursive_training_engine/kernels/reference.py` | Pure PyTorch reference implementations for all public kernel symbols. |
| `src/recursive_training_engine/kernels/optimized.py` | Optimized dispatch surface, strict CUDA checks, FlashAttention dispatch attempt, and PyTorch fallbacks. |
| `src/recursive_training_engine/kernels/__init__.py` | Exports `optimized` and `reference` kernel modules. |

### Configurations

| Path | Purpose |
| --- | --- |
| `configs/tiny.yaml` | Synthetic smoke-test preset with small recursive macro shortlist mode. |
| `configs/tiny_mac_fused.yaml` | Mac-friendly tiny preset with larger sequence length and low audit rate. |
| `configs/tiny_mac_real_speed.yaml` | Tiny real TinyStories speed preset for Mac/MPS-like local tests. |
| `configs/tiny_mac_real_audit.yaml` | Tiny real-data preset with audit probability forced to 1.0. |
| `configs/small.yaml` | CUDA-oriented small preset with bf16, TF32, compile hooks, strict CUDA, and shortlist output. |
| `configs/small_mac_real_speed.yaml` | Mac-friendly small real-data speed preset with audits disabled. |
| `configs/medium.yaml` | CUDA-oriented medium preset with larger vocab/model and shortlist output. |
| `configs/medium_mac_real_speed.yaml` | Mac-friendly medium real-data speed preset with audits disabled. |
| `configs/medium_mac_real_hotpath.yaml` | Medium hot-path preset using batch size 32 and full output, audits disabled. |
| `configs/medium_mac_real_1m.yaml` | Medium 1M-token real-data convergence preset with fixed audit probability 0.25 and audit cap 1. |
| `configs/large.yaml` | Larger CUDA-oriented preset with 8192 vocab, d_model 128, and strict CUDA settings. |
| `configs/ablations/matrix.yaml` | Human-readable ablation category matrix. The actual executable ablations are generated in `ablations.py`. |

### Tests

| Path | Coverage |
| --- | --- |
| `tests/conftest.py` | Adds `src/` to the import path for tests. |
| `tests/test_models_and_training.py` | Dense determinism, exact recurrence parity, macro loss shape, shortlist target inclusion, and one training step for all modes. |
| `tests/test_fairness_and_recipes.py` | Tiny fairness pass, recipe validity/balance, usage EMA, and depth decomposition helper. |
| `tests/test_audit.py` | Full-audit correction replay shape and exact equality when audit probability is 1.0. |
| `tests/test_kernels.py` | RMSNorm parity, pack/unpack roundtrip, shortlist logits shape, and CUDA/Triton availability behavior. |
| `tests/test_strict_fast_path.py` | No Python loops in macro hot forward and shortlist builder, plus strict CUDA CPU rejection. |
| `tests/test_ablations.py` | Required ablation categories and named ablations are present. |

### Benchmarks And Scripts

| Path | Purpose |
| --- | --- |
| `benchmarks/benchmark_kernels.py` | Times selected optimized kernel symbols on the default device. |
| `benchmarks/time_to_quality.py` | Runs modes until a target training loss is reached or a max step count expires. |
| `benchmarks/convergence_compare.py` | Runs dense and recursive modes over a token budget with periodic exact/hot evaluation. |
| `scripts/check_all.sh` | Shell smoke script for pytest, fairness, dense training, and recursive macro shortlist training. |

### Local Run Artifacts

`runs/` is ignored by git but currently contains local JSONL metrics.

Non-empty metric files observed:

- `runs/tiny/metrics.jsonl`
- `runs/medium_mac_real_hotpath/metrics.jsonl`
- `runs/medium_mac_real_1m/metrics.jsonl`
- `runs/medium_mac_real_1m_convergence.jsonl`
- `runs/medium_mac_real_1m_macro_fixed.jsonl`
- `runs/medium_mac_real_1m_macro_bounded.jsonl`
- `runs/medium_mac_real_1m_macro_clipped.jsonl`
- `runs/medium_mac_real_1m_macro_stable.jsonl`
- `runs/medium_mac_real_1m-convergence-dense_exact/metrics.jsonl`
- `runs/medium_mac_real_1m-convergence-recursive_macro/metrics.jsonl`

Empty metric placeholders observed:

- `runs/medium_mac_real_speed-quick-noaudit-dense_exact/metrics.jsonl`
- `runs/medium_mac_real_speed-quick-noaudit-recursive_exact/metrics.jsonl`
- `runs/medium_mac_real_speed-quick-noaudit-recursive_macro/metrics.jsonl`
- `runs/medium_mac_real_speed-quick-noaudit-recursive_macro_shortlist/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-dense_exact-bs4/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-dense_exact-bs8/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-dense_exact-bs16/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-dense_exact-bs32/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-recursive_macro-bs4/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-recursive_macro-bs8/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-recursive_macro-bs16/metrics.jsonl`
- `runs/medium_mac_real_speed-batch-grid-recursive_macro-bs32/metrics.jsonl`

## How The Project Works

### CLI Flow

The `rte` command is wired in `pyproject.toml`:

```text
rte -> recursive_training_engine.cli:main
```

The main training flow is:

```text
rte train
  -> load_config(...)
  -> optionally override mode/topology
  -> load_token_streams(...)
  -> TrainEngine(...)
  -> train_step(...)
  -> model forward path
  -> optimizer step
  -> JSONL metrics logging
```

Available CLI subcommands:

- `rte fairness`: compares dense and recursive stored parameter counts and other
  fairness flags.
- `rte train`: runs training steps for one mode.
- `rte evaluate`: runs a single eval batch for one mode.
- `rte benchmark-kernels`: times selected optimized kernel dispatch points.
- `rte run-ablations`: builds and executes the ablation matrix.
- `rte compare-ttq`: scans run metric logs for target-loss reach events.

### Configuration Loading

`config.py` defines four main dataclasses:

- `ModelConfig`: topology, dimensions, depth choices, banks, sparse groups,
  recipes, macro rank, and router shape.
- `TrainingConfig`: mode, optimizer settings, batch/sequence sizes, audit
  probabilities, auxiliary loss weights, precision, compile flags, and strict
  CUDA options.
- `OutputConfig`: full-vocab or shortlist-related output settings.
- `DataConfig`: dataset/tokenizer/cache/local-text settings.

`load_config` reads YAML and filters each dictionary through dataclass field
names. Unknown keys are ignored rather than rejected.

Important validation happens in `ModelConfig.__post_init__`:

- `d_model` must divide evenly into heads.
- `n_heads` must divide evenly into head groups.
- `d_ff` must divide evenly into FFN groups.
- active groups must fit within total groups.
- `recipe_count` must include sparse recipes plus the dense fallback.
- `t_max` must appear in `depth_choices`.
- `depth_choices` must be sorted.

### Data Pipeline

`load_token_streams` returns a `TokenStreams` object with train tokens, eval
tokens, tokenizer name, and a data fingerprint.

For `dataset: synthetic`, the project creates deterministic pseudo-token data
from the configured seed. For text datasets, it attempts to stream TinyStories
or WikiText-103 through Hugging Face `datasets`, then encodes with GPT-2 BPE via
`tokenizers`. If download/tokenization fails, it trains a small byte-level BPE
fallback on available text. If too few valid tokens remain after applying
`vocab_size`, it fills with synthetic tokens.

Batching is sequential and deterministic: `batch_iterator` walks through the
packed token stream and returns `(tokens, targets)` where targets are the next
tokens.

### Dense Baseline

`DenseModel` is a pre-norm decoder-only Transformer:

```text
tokens
  -> embedding
  -> n_dense_layers of TransformerBlock
  -> final RMSNorm
  -> tied embedding head or separate LM head
  -> logits
  -> per-sample summed next-token cross entropy
```

Each `TransformerBlock` uses:

- RMSNorm
- causal self-attention with fused QKV projection
- optional RoPE
- PyTorch scaled dot-product attention
- dense output projection
- RMSNorm
- fused up/gate SwiGLU projection
- down projection

Losses are first computed as per-sample sums over sequence positions. The train
engine divides by sequence length when `loss_normalization` is `token_mean`.

### Recursive Model

`RecursiveModel` has this layout:

```text
tokens
  -> embedding
  -> prelude dense blocks
  -> router chooses recipe and depth
  -> recursive core, either exact or macro
  -> coda dense blocks
  -> final RMSNorm
  -> tied embedding head or separate LM head
  -> logits/loss
```

The recursive model contains:

- `RecipeBank`: static sparse recipe templates plus recipe `0`, the dense
  fallback recipe.
- `Router`: MLPs that choose a recipe and a depth from `depth_choices`.
- `BankedRecursiveCore`: recurrent attention/MLP step over selected banks and
  sparse groups.
- `MacroOperators`: learned low-rank approximate transition operators.
- `ShortlistHead`: optional approximate output scoring over a token shortlist.

### Recipes And Sparse Execution

Recipe `0` is the dense fallback:

- all attention banks
- all FFN banks
- all head groups
- all FFN groups

Sparse recipes are built deterministically with modular arithmetic:

- one attention bank
- one FFN bank
- `active_head_groups` contiguous wrapped head groups
- `active_ffn_groups` contiguous wrapped FFN slabs

`RecipeBank` also builds an active-touch table estimating the parameter touches
for each recipe. Training logs report `active_touches_per_token` from that
table.

The exact recursive core groups samples by selected recipe id. For each active
recipe, it slices the batch, applies the banked step, and writes results back
into the output tensor.

### Exact Recursive Path

`RecursiveModel.forward_exact` is the source-of-truth recurrent path.

It runs:

```text
h0 = prelude(tokens)
route = router(h0)
h = h0
for t in range(t_max):
    active = route.depth > t
    h = core.forward_step(h, h0, route.recipe_id, active)
hidden, logits = coda_logits(h)
```

Samples stop receiving recurrent updates once `t` reaches their routed depth.
If requested, the method stores hidden states at configured depth boundaries.

This path is slower because it is a real sequential recurrence and because the
current banked core still loops over unique recipes.

### Macro Hot Path

`RecursiveModel.forward_macro` is the approximate hot path.

It runs:

```text
h0 = prelude(tokens)
route = router(h0)
h = macro(h0, h0, route.recipe_id, route.depth)
hidden, logits = coda_logits(h)
```

The actual implementation currently applies one vectorized low-rank macro
operator for the chosen `(recipe, depth)` pair. The operator normalizes the
hidden state, pools sequence context, concatenates normalized hidden state,
pooled hidden state, and `h0`, then applies a learned low-rank update:

```text
z = concat(norm(h), pooled(norm(h)), h0)
low = silu(z @ v) @ u
update = d_gain * h + low + bias
out = h + macro_update_scale * tanh(update)
```

The code contains a `greedy_decompose_depth` helper and documentation references
greedy/binary depth decomposition, but the current `MacroOperators.forward`
does not decompose depths into multiple physical macro passes. It maps each
depth directly to one stride id and reports `physical_passes` as `1`.

### Audit Correction

The audit path exists to prevent the macro approximation from being trusted
blindly.

For macro modes, training first computes the hot macro loss per sample. Then
`AuditEngine` chooses an audit probability:

```text
p = audit_p_min
  + audit_alpha * router_uncertainty
  + audit_beta * recent_residual
  + audit_gamma * coverage_deficit
```

The value is clamped to `[audit_p_min, audit_p_max]`. In the current code,
`coverage_deficit` inside this probability formula is always zero, though
coverage is still handled through an auxiliary router loss.

The audit engine samples a mask, optionally caps it with `audit_cap`, and replays
only that subset through `forward_exact_subset`, reusing the hot router
decisions. It then computes:

```text
corrected = hot_loss
corrected[audited] = hot_loss[audited] + (exact_loss - hot_loss) / p_audit
```

This is the intended control-variate correction:

```text
hot + audit_mask / p_audit * (exact - hot)
```

When `audit_p_min = audit_p_max = 1.0`, the corrected per-sample loss equals the
exact per-sample loss. There is a test for this.

If exact audited hidden/logit data is available, macro auxiliary losses are also
added:

- hidden MSE
- hidden cosine distance
- logit KL, only when hot and exact logits have the same shape
- macro consistency loss for stride pairs where one stride is double another

By default, exact audit residuals are detached from gradients unless
`audit_gradient_correction` is enabled.

### Shortlist Output

`recursive_macro_shortlist` uses `ShortlistHead` to avoid full-vocabulary logits
on the hot path.

For each hidden token:

1. A cluster router scores vocabulary clusters.
2. Top clusters are selected.
3. Candidate tokens are assembled from:
   - the true target token
   - members of top clusters
   - cached hard negatives
   - random negatives
   - deterministic filler tokens
4. Candidates are truncated to `shortlist_max_tokens`.
5. The target is always at shortlist position `0`.
6. Gathered vocabulary rows are dot-producted with the hidden state.
7. Cross entropy is computed against target position `0`.

Duplicate shortlist entries are possible, but the target is guaranteed to be
included at position `0`.

Audited samples always replay through the exact full-vocabulary path.

### Training Engine

`TrainEngine` handles:

- seed setup
- device selection: CUDA, then MPS, then CPU
- strict CUDA checks
- TF32 configuration when allowed
- model construction
- optional `torch.compile` hooks
- AdamW optimizer construction with fused/foreach options when available
- macro audit engine construction
- JSONL metric logging

Mode dispatch:

| Mode | Model path | Loss source |
| --- | --- | --- |
| `dense_exact` | `DenseModel.forward` | full dense exact LM loss |
| `recursive_exact` | `RecursiveModel.forward_exact` | exact recurrent full-vocab LM loss plus router auxiliary losses |
| `recursive_macro` | `RecursiveModel.forward_macro` | audited corrected macro/full-vocab loss plus macro/router auxiliary losses |
| `recursive_macro_shortlist` | `RecursiveModel.forward_macro(shortlist=True)` | audited corrected shortlist hot loss plus macro/router auxiliary losses |

Training logs include loss, NLL/token, step time, tokens/sec, peak VRAM for CUDA,
stored parameter count, active touches, average depth, audit metrics, macro
auxiliary losses, and router auxiliary losses depending on mode.

### Fairness Checks

`metrics.py` computes approximate stored parameter counts for dense and
recursive configurations.

The fairness report checks:

- stored parameter count within tolerance
- same tokenizer
- same data
- same sequence length
- same optimizer
- same objective
- exact path availability

The CLI can suggest attention/FFN bank counts that better match dense parameter
count under a maximum bank search.

### Kernels

The project defines a public kernel surface in two modules:

- `kernels/reference.py`: pure PyTorch implementations
- `kernels/optimized.py`: dispatch wrappers with strict CUDA checks and
  FlashAttention attempts

Important current reality: `optimized.py` does not yet contain custom Triton
kernels. Most symbols delegate to the reference PyTorch implementation after
checking device constraints. On CUDA, attention attempts to force PyTorch SDPA
FlashAttention through `sdpa_kernel([SDPBackend.FLASH_ATTENTION])`.

This is a useful abstraction layer because real Triton/CUDA kernels can later
replace the implementation behind the same function names.

## Current Local Results Captured In The Repo

The README records local Apple MPS measurements from April 30, 2026.

Hot-path, no-audit speed numbers for `configs/medium_mac_real_speed.yaml`:

| Mode | Tokens/sec | Speedup vs dense | Speedup vs recursive exact |
| --- | ---: | ---: | ---: |
| `dense_exact` | 3,710 | 1.00x | 2.37x |
| `recursive_exact` | 1,563 | 0.42x | 1.00x |
| `recursive_macro` | 10,974 | 2.96x | 7.02x |
| `recursive_macro_shortlist` | 9,692 | 2.61x | 6.20x |

The repo also contains a non-empty `medium_mac_real_hotpath` metrics file whose
last logged row shows a warmed macro hot-path step at about 146,647 tokens/sec
for batch size 32, sequence length 64, with audits disabled. The batch-grid
metrics files under `runs/medium_mac_real_speed-batch-grid-*` are currently
empty, so the README batch-grid table cannot be independently reconstructed
from those JSONL files alone.

For the 1M-token convergence comparison, `runs/medium_mac_real_1m_macro_fixed.jsonl`
contains the stable recursive macro run summarized in the README:

| Train tokens | Recursive exact eval NLL/token | Recursive hot eval NLL/token |
| ---: | ---: | ---: |
| 250,112 | 4.482 | 4.386 |
| 500,224 | 4.122 | 4.011 |
| 750,336 | 4.218 | 3.902 |
| 1,000,192 | 3.601 | 3.353 |

`runs/medium_mac_real_1m_convergence.jsonl` contains dense checkpoints ending at
1,000,192 train tokens with dense exact eval NLL/token of about 3.232 in about
265 seconds. The same file also contains an earlier recursive macro attempt that
became NaN by 250,112 tokens. The stable macro run is in
`runs/medium_mac_real_1m_macro_fixed.jsonl`.

Current interpretation from the available artifacts:

- The no-audit macro hot path is faster locally than dense and much faster than
  exact recurrence.
- The audited recursive macro path can be stable with clipping/capping/fixed
  audit settings.
- The captured 1M-token Mac run does not yet beat dense exact on exact-path
  validation loss or wall-clock time-to-quality.
- CUDA/Triton runs are still needed before making GPU speedup claims.

## Verified In This Audit

The full test suite was run with:

```bash
uv run pytest
```

Result:

```text
20 passed in 1.73s
```

This verifies the current unit/regression coverage, but it does not verify CUDA
performance, real Triton kernels, long training stability, or time-to-quality
wins.

## Important Caveats And Mismatches

1. The repository has no commits yet. Everything should be reviewed as an
   uncommitted working tree.
2. `optimized.py` is a dispatch layer, not a finished custom Triton kernel
   implementation.
3. Documentation mentions greedy/binary macro depth decomposition, but
   `MacroOperators.forward` currently applies one vectorized macro operator per
   chosen depth and reports one physical pass.
4. `TrainingConfig.grad_accum_steps` exists but is not used in `TrainEngine`.
5. `OutputConfig.mode` is mostly declarative. The training mode controls whether
   the shortlist path is used.
6. `DataConfig.tokenizer` accepts values like `byte`, but the current loader
   always tries GPT-2 BPE first for text data and only uses byte-level BPE as a
   fallback.
7. The audit probability formula has a `coverage_deficit` term, but it is
   currently hard-coded to zero. Coverage is still penalized through router
   auxiliary loss.
8. When `audit_cap` truncates too many sampled audits, it keeps the first sampled
   indices rather than randomly subsampling the sampled mask.
9. Several run metric files exist but are empty. They look like placeholders or
   interrupted/unfinished runs.
10. There is no model checkpoint save/load path in the training engine yet.
11. There is no distributed training path, no experiment manifest writer, and no
   automatic artifact summary beyond JSONL metrics.

## Mental Model For Future Work

The safest way to understand the project is:

```text
Dense exact model
  = fairness baseline

Recursive exact model
  = correctness source of truth for the recursive architecture

Recursive macro model
  = fast approximate hot path

Audit engine
  = statistical correction that periodically pulls macro training back toward
    exact recursive behavior

Shortlist head
  = optional way to reduce output-head cost on the approximate hot path

Kernels package
  = stable API surface where real CUDA/Triton kernels can be inserted later
```

The core research question is whether the recursive macro hot path plus audit
correction can reach equal or better validation quality faster than the dense
baseline under fair stored-parameter, data, optimizer, and evaluation settings.
The current code establishes the harness for that question, but the strongest
claim supported by local artifacts is narrower: the macro hot path is faster
when audits are disabled, while audited end-to-end time-to-quality still needs
larger and better-controlled CUDA/Triton experiments.
