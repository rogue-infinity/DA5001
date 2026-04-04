#!/bin/bash
# monitor.sh — Tail active log files from the pipeline.
# Usage: bash monitor.sh [logfile]
#   e.g. bash monitor.sh finetune   → tails logs/finetune.log
#        bash monitor.sh            → tails all logs

LOG_DIR="logs"

if [ $# -eq 1 ]; then
    tail -f "${LOG_DIR}/${1}.log" 2>/dev/null || echo "Log not found: ${LOG_DIR}/${1}.log"
else
    LOGS=("prepare" "finetune" "verify" "sama" "signals" "classifier" "benchmark")
    FILES=()
    for name in "${LOGS[@]}"; do
        f="${LOG_DIR}/${name}.log"
        [ -f "$f" ] && FILES+=("$f")
    done

    if [ ${#FILES[@]} -eq 0 ]; then
        echo "No log files found in ${LOG_DIR}/. Has the pipeline been started?"
    else
        echo "Tailing: ${FILES[*]}"
        tail -f "${FILES[@]}"
    fi
fi
