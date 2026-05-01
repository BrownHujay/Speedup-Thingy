#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -m recursive_training_engine.cli fairness --config configs/tiny.yaml --strict
PYTHONPATH=src python3 -m recursive_training_engine.cli train --config configs/tiny.yaml --mode dense_exact --steps 1
PYTHONPATH=src python3 -m recursive_training_engine.cli train --config configs/tiny.yaml --mode recursive_macro_shortlist --steps 1
