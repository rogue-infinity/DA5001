#!/bin/bash
# run_pipeline_dryrun.sh — Dry-run with 10 samples and 2 epochs to check the pipeline end-to-end.
set -euo pipefail

mkdir -p data results logs models

echo "=== [DRY RUN] 10 samples, 2 epochs ==="

echo "=== [1/7] prepare_data.py ==="
python prepare_data.py --n_samples 10 2>&1 | tee logs/prepare_dryrun.log

echo "=== [2/7] finetune.py ==="
python finetune.py --n_epochs 2 2>&1 | tee logs/finetune_dryrun.log

echo "=== [3/7] verify_memorization.py ==="
python verify_memorization.py 2>&1 | tee logs/verify_dryrun.log || echo "[DRY RUN] verify may fail with only 10 samples — continuing"

echo "=== [4/7] run_sama.py ==="
python run_sama.py --n_samples 10 --n_comparisons 16 2>&1 | tee logs/sama_dryrun.log

echo "=== [5/7] run_signals.py ==="
python run_signals.py --n_samples 5 2>&1 | tee logs/signals_dryrun.log

echo "=== [6/7] train_classifier.py ==="
python train_classifier.py --n_bootstraps 50 2>&1 | tee logs/classifier_dryrun.log

echo "=== [7/7] benchmark.py ==="
python benchmark.py --n_bootstraps 50 2>&1 | tee logs/benchmark_dryrun.log

echo ""
echo "Dry-run complete."
