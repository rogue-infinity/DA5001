#!/bin/bash
# jarvis_setup.sh — Runs on each JarvisLabs instance after code upload.
# Called by jarvis_launch.sh via: jl exec <instance> "bash code/jarvis_setup.sh <DATASET>"
#
# Env vars expected (set by jarvis_launch.sh before exec):
#   DATASET     — e.g. arxiv, github, wikitext103
#   HF_TOKEN    — HuggingFace token (for gated MIMIR dataset)
#   WANDB_TOKEN — Weights & Biases API key
#   START_FROM  — optional stage to resume from (default 1)
set -euo pipefail

DATASET="${1:?ERROR: DATASET arg required}"
START_FROM="${START_FROM:-1}"

cd /home/user/run3 2>/dev/null || cd ~/run3

echo "=== JarvisLabs setup: $DATASET ==="
echo "START_FROM=$START_FROM"

# Authenticate
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
wandb login "$WANDB_TOKEN" || true

# Install Python deps
pip install -r code/requirements_remote.txt -q

# dllm is installed by run_pipeline.sh (needs ./dllm local dir or editable install)
# SAMA is cloned by run_pipeline.sh on first run

# Change into code dir so relative paths (data/, models/, logs/, results/) work
cd code/

echo "=== Starting pipeline for $DATASET ==="
bash run_pipeline.sh "$DATASET" "$START_FROM"

echo "=== Pipeline finished for $DATASET ==="
