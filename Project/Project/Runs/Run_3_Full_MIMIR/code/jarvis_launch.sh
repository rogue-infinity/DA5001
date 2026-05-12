#!/bin/bash
# jarvis_launch.sh — Launch 9 parallel JarvisLabs instances (one per dataset).
#
# Prerequisites:
#   jl CLI installed and logged in  (pip install jlabs-cli)
#   HF_TOKEN, WANDB_TOKEN exported in your shell
#
# Usage:
#   export HF_TOKEN=hf_...
#   export WANDB_TOKEN=...
#   bash jarvis_launch.sh                  # launch all 9 datasets
#   bash jarvis_launch.sh github           # launch only one dataset (for testing)
#   RESUME_FROM=4 bash jarvis_launch.sh    # resume all from stage 4
#
# Instance naming:  mia-run3-<DATASET>
# GPU:              3× A100-80G  (3 GPUs for DeepSpeed Stage 1)
# Storage:          100 GB persistent disk (models are large)
set -euo pipefail

: "${HF_TOKEN:?ERROR: export HF_TOKEN first}"
: "${WANDB_TOKEN:?ERROR: export WANDB_TOKEN first}"

RESUME_FROM="${RESUME_FROM:-1}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN3_DIR="$(dirname "$CODE_DIR")"   # Run_3_Full_MIMIR/

# All 9 datasets — can override by passing one as arg
if [ -n "${1:-}" ]; then
    DATASETS=("$1")
else
    DATASETS=(arxiv github hackernews pubmed_central wikipedia pile_cc wikitext103 agnews xsum)
fi

echo "=== Run_3_Full_MIMIR — JarvisLabs Launcher ==="
echo "Datasets: ${DATASETS[*]}"
echo "Resume from stage: $RESUME_FROM"
echo ""

for DATASET in "${DATASETS[@]}"; do
    INSTANCE_NAME="mia-run3-${DATASET}"
    echo "--- Launching: $INSTANCE_NAME ---"

    # Create instance (3× A100-80G; adjust --framework if needed)
    # jl create returns instance ID which we capture
    INSTANCE_ID=$(jl create \
        --name    "$INSTANCE_NAME" \
        --gpu     "A100-80G" \
        --num-gpus 3 \
        --disk    100 \
        --framework "PyTorch" \
        --on-demand \
        --quiet 2>/dev/null || echo "")

    if [ -z "$INSTANCE_ID" ]; then
        echo "  [WARN] Could not create $INSTANCE_NAME — may already exist. Trying to reuse..."
        INSTANCE_ID=$(jl list --quiet | grep "$INSTANCE_NAME" | awk '{print $1}' | head -1)
    fi

    if [ -z "$INSTANCE_ID" ]; then
        echo "  [ERROR] Could not find or create $INSTANCE_NAME — skipping."
        continue
    fi

    echo "  Instance ID: $INSTANCE_ID"

    # Upload Run_3 folder (code + SAMA) to ~/run3 on the instance
    jl upload "$INSTANCE_ID" "$RUN3_DIR" ~/run3 --quiet || true

    # Run setup + pipeline; pass secrets as env vars inline
    jl exec "$INSTANCE_ID" \
        "HF_TOKEN='$HF_TOKEN' WANDB_TOKEN='$WANDB_TOKEN' START_FROM='$RESUME_FROM' \
         bash ~/run3/code/jarvis_setup.sh '$DATASET'" \
        --background \
        --quiet || echo "  [WARN] exec failed for $INSTANCE_NAME"

    echo "  Launched $INSTANCE_NAME in background."
    echo ""
done

echo "=== All instances launched. ==="
echo ""
echo "Monitor:"
echo "  jl list                            # see running instances"
echo "  jl logs mia-run3-<DATASET>         # tail logs for a dataset"
echo "  wandb dashboard: da5001-mia project"
echo ""
echo "Download results when done:"
echo "  for ds in arxiv github hackernews pubmed_central wikipedia pile_cc wikitext103 agnews xsum; do"
echo "    jl download mia-run3-\$ds ~/run3/results/\$ds/ $RUN3_DIR/results/ --quiet"
echo "  done"
echo ""
echo "Then run: python code/analysis.py  (locally)"
