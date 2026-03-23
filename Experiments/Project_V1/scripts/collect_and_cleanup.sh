#!/usr/bin/env bash
# collect_and_cleanup.sh
# ----------------------
# Polls jl run status for r_f0d32018 every 5 minutes.
# When training succeeds: downloads checkpoints + logs locally, destroys instance.
# If training fails:     prints the last 50 log lines and exits with error.

set -euo pipefail

RUN_ID="r_f0d32018"
MACHINE_ID="387206"
LOCAL_ROOT="/Users/skumar/Desktop/DA5001/Experiments/Project_V1"
REMOTE_PROJECT="/home/Project_V1"
POLL_SECONDS=300   # 5 minutes

CKPT_DEST="${LOCAL_ROOT}/checkpoints"
LOG_DEST="${LOCAL_ROOT}/logs"

echo "[collect] Watching run ${RUN_ID} on machine ${MACHINE_ID} ..."
echo "[collect] Poll interval: ${POLL_SECONDS}s"
echo "[collect] Checkpoints → ${CKPT_DEST}"
echo "[collect] Logs        → ${LOG_DEST}"
echo ""

while true; do
    STATUS_JSON=$(jl run status "${RUN_ID}" --json 2>&1)
    STATE=$(echo "${STATUS_JSON}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('state','unknown'))" 2>/dev/null || echo "unknown")
    echo "[$(date '+%H:%M:%S')] state=${STATE}"

    if [[ "${STATE}" == "succeeded" ]]; then
        echo ""
        echo "[collect] Training complete — downloading artifacts ..."

        # Download checkpoints directory
        mkdir -p "${CKPT_DEST}"
        jl download "${MACHINE_ID}" "${REMOTE_PROJECT}/checkpoints" "${CKPT_DEST}" -r
        echo "[collect] Checkpoints saved → ${CKPT_DEST}/"

        # Download finetune log
        mkdir -p "${LOG_DEST}"
        jl download "${MACHINE_ID}" "${REMOTE_PROJECT}/logs/finetune.jsonl" "${LOG_DEST}/finetune.jsonl"
        echo "[collect] Log saved        → ${LOG_DEST}/finetune.jsonl"

        # Destroy instance
        echo "[collect] Destroying instance ${MACHINE_ID} ..."
        jl destroy "${MACHINE_ID}" --yes --json
        echo "[collect] Instance destroyed. Billing stopped."
        echo ""
        echo "=== Done. Artifacts at: ==="
        ls -lh "${CKPT_DEST}/"
        echo ""
        exit 0

    elif [[ "${STATE}" == "failed" ]]; then
        echo ""
        echo "[collect] ERROR: run failed. Last 50 log lines:"
        jl run logs "${RUN_ID}" --tail 50
        echo ""
        echo "[collect] Instance NOT destroyed — inspect manually."
        exit 1

    else
        # Still running — sleep and poll again
        sleep "${POLL_SECONDS}"
    fi
done
