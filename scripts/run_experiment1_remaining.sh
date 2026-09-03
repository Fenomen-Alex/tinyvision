#!/bin/bash
# Run remaining classification candidates for Experiment 1 (30 epochs each).
# Run from repo root: nohup bash scripts/run_experiment1_remaining.sh
set -uo pipefail

MODELS="resnet18 mobilenet_v3_small mobilenet_v3_large shufflenet_v2_x1_0"
VENV="training/.venv/bin/tv"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR"

for MODEL in $MODELS; do
    MODEL_CONFIG="training/configs/experiment1_${MODEL}.yaml"
    LOG="$LOG_DIR/${MODEL}.log"

    echo "=== $(date) Training $MODEL ==="
    $VENV train --config "$MODEL_CONFIG" --experiments-dir experiments \
        > "$LOG" 2>&1
    echo "=== $(date) Evaluating $MODEL ==="
    $VENV evaluate --config "$MODEL_CONFIG" --experiments-dir experiments \
        >> "$LOG" 2>&1
    echo "=== $(date) Exporting $MODEL ==="
    $VENV export --config "$MODEL_CONFIG" --experiments-dir experiments \
        >> "$LOG" 2>&1
    echo "=== $(date) Done $MODEL ==="
done

echo "All models trained, evaluated, and exported."
