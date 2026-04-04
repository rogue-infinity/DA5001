#!/bin/bash
# run_pipeline.sh — Full MIA pipeline (prepare → finetune → verify → sama → signals → train → benchmark)
set -euo pipefail

mkdir -p data results logs models

echo "=== [1/7] prepare_data.py ==="
python prepare_data.py 2>&1 | tee logs/prepare.log

echo "=== [2/7] finetune.py ==="
python finetune.py 2>&1 | tee logs/finetune.log

echo "=== [3/7] verify_memorization.py ==="
python verify_memorization.py 2>&1 | tee logs/verify.log

echo "=== [4/7] run_sama.py ==="
python run_sama.py 2>&1 | tee logs/sama.log

echo "=== [5/7] run_signals.py ==="
python run_signals.py 2>&1 | tee logs/signals.log

echo "=== [6/7] train_classifier.py ==="
python train_classifier.py 2>&1 | tee logs/classifier.log

echo "=== [7/7] benchmark.py ==="
python benchmark.py 2>&1 | tee logs/benchmark.log

echo ""
echo "Pipeline complete. Results in results/"
