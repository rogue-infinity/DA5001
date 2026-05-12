#!/bin/bash
# run_shadow_setup.sh — Runs ON each JarvisLabs instance after code upload.
# Called remotely via: jl exec <id> -- bash /home/run4/code/run_shadow_setup.sh <DATASET> <START_PHASE>
#
# This script:
#   1. Sources secrets from /home/.env
#   2. Installs Python dependencies
#   3. Pins PyTorch to cu128 wheel
#   4. Authenticates wandb + huggingface
#   5. Installs dllm model package
#   6. Runs run_shadow_mia.py for the specified domain

set -euo pipefail

DATASET="${1:?ERROR: DATASET arg required}"
START_PHASE="${2:-1}"

# cd to code dir so all relative paths (./dllm, ../shadow_results, ../run3_results) resolve correctly
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Run4 Shadow MIA — Instance setup ==="
echo "  Dataset:     $DATASET"
echo "  Start phase: $START_PHASE"
echo "  Working dir: $(pwd)"
echo ""

# ----------------------------------------------------------------
# Source secrets
# ----------------------------------------------------------------
[ -f /home/.env ] && source /home/.env || true
[ -f ~/.env ]     && source ~/.env     || true

: "${HF_TOKEN:?ERROR: HF_TOKEN not set. Check /home/.env upload.}"
: "${WANDB_API_KEY:?ERROR: WANDB_API_KEY not set. Check /home/.env upload.}"

# ----------------------------------------------------------------
# Skip PyTorch reinstall — JarvisLabs pytorch template ships 2.11+cu130
# which is compatible. Only install if no torch found.
# ----------------------------------------------------------------
python3 -c "import torch; print(f'Using pre-installed torch {torch.__version__}')" 2>/dev/null || \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu124 -q

# ----------------------------------------------------------------
# Install project requirements (shap, seaborn, datasets, xgboost…)
# ----------------------------------------------------------------
echo "Installing requirements..."
pip install -r requirements_remote.txt -q

# ----------------------------------------------------------------
# Authenticate
# ----------------------------------------------------------------
echo "Authenticating services..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -2 || true
wandb login "$WANDB_API_KEY" --relogin 2>&1 | tail -2 || true

# ----------------------------------------------------------------
# Install dllm model package (editable install so imports resolve)
# ----------------------------------------------------------------
if [ -d "./dllm" ]; then
    pip install --no-deps -e ./dllm -q 2>/dev/null || true
fi

# ----------------------------------------------------------------
# Create output directories
# ----------------------------------------------------------------
mkdir -p "../shadow_results/$DATASET"
mkdir -p "../run3_results/$DATASET"   # should already exist from launcher upload

# ----------------------------------------------------------------
# Run the main script
# ----------------------------------------------------------------
echo ""
echo "=== Starting run_shadow_mia.py for $DATASET (phase $START_PHASE) ==="
python run_shadow_mia.py \
    --dataset      "$DATASET" \
    --start_phase  "$START_PHASE" \
    --run3_results "../run3_results" \
    2>&1 | tee "../shadow_results/$DATASET/run4_full.log"

echo "=== run_shadow_mia.py finished for $DATASET ==="
