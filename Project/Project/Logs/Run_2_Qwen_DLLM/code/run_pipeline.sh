#!/bin/bash
# run_pipeline.sh — Full MIA pipeline for Run_2_Qwen_DLLM
# Stages: prepare → finetune → verify → sama → attacks → signals → train → benchmark
set -euo pipefail

# Optional first argument: stage to start from (default 1)
# Usage: bash run_pipeline.sh 4   → skips stages 1-3
START_FROM="${1:-1}"

# Source secrets if present (written by JarvisLabs deploy step)
[ -f ~/.env ] && source ~/.env

# Verify required env vars
: "${HF_TOKEN:?ERROR: HF_TOKEN not set. Export it or write to ~/.env}"

# ------------------------------------------------------------------
# SAMA repo setup
# ------------------------------------------------------------------
if [ -z "${SAMA_ROOT:-}" ]; then
    # Check sibling location (local: Project/SAMA)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LOCAL_SAMA="$(realpath "$SCRIPT_DIR/../../../SAMA" 2>/dev/null || true)"

    if [ -d "$LOCAL_SAMA/attack" ]; then
        export SAMA_ROOT="$LOCAL_SAMA"
        echo "Using local SAMA at: $SAMA_ROOT"
    else
        # On remote: clone to /home/SAMA (persists across jl runs, not in rsync scope)
        if [ ! -d "/home/SAMA/attack" ]; then
            echo "Cloning SAMA repo (MIT licensed) to /home/SAMA..."
            rm -rf /home/SAMA  # remove any partial clone
            git clone https://github.com/Stry233/SAMA.git /home/SAMA
        else
            echo "Using cached SAMA at: /home/SAMA"
        fi
        export SAMA_ROOT="/home/SAMA"
    fi
fi

# Install dllm without its conflicting deps (we only need the model architecture)
# dllm pins datasets==4.2.0 which breaks MIMIR loading; we use datasets<3.0 instead
python -c "import dllm" 2>/dev/null || { echo "Installing dllm (no-deps)..." && pip install --no-deps -e ./dllm -q; }

mkdir -p data results logs models

echo ""
echo "=========================================="
echo " Run_2_Qwen_DLLM MIA Pipeline"
echo " SAMA_ROOT: $SAMA_ROOT"
echo "=========================================="
echo ""

if [ "$START_FROM" -le 1 ]; then
  echo "=== [1/7] prepare_data.py ==="
  python prepare_data.py 2>&1 | tee logs/prepare.log
fi

if [ "$START_FROM" -le 2 ]; then
  echo "=== [2/7] finetune.py ==="
  python finetune.py 2>&1 | tee logs/finetune.log
fi

if [ "$START_FROM" -le 3 ]; then
  echo "=== [3/7] verify_memorization.py ==="
  python verify_memorization.py 2>&1 | tee logs/verify.log
fi

if [ "$START_FROM" -le 4 ]; then
  echo "=== [4/8] run_sama.py ==="
  python run_sama.py 2>&1 | tee logs/sama.log
fi

if [ "$START_FROM" -le 5 ]; then
  echo "=== [5/8] run_attacks.py (Loss / Zlib / Ratio) ==="
  python run_attacks.py 2>&1 | tee logs/attacks.log
fi

if [ "$START_FROM" -le 6 ]; then
  echo "=== [6/8] run_signals.py ==="
  python run_signals.py 2>&1 | tee logs/signals.log
fi

if [ "$START_FROM" -le 7 ]; then
  echo "=== [7/8] train_classifier.py ==="
  python train_classifier.py 2>&1 | tee logs/classifier.log
fi

if [ "$START_FROM" -le 8 ]; then
  echo "=== [8/8] benchmark.py ==="
  python benchmark.py 2>&1 | tee logs/benchmark.log
fi

echo ""
echo "Pipeline complete. Results in results/"
