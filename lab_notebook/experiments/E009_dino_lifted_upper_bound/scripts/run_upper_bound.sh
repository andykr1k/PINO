#!/bin/bash
# Object probe with lifted DINO features (upper bound) and view-averaged student features, on one checkpoint.
# Lifting renders at the checkpoint's training resolution.
#   CKPT=/path/ckpt.pt NAME=e004_448 GPU=7 bash lab_notebook/experiments/E009_dino_lifted_upper_bound/scripts/run_upper_bound.sh [extra probe args]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$EXP_DIR/logs/${NAME:?set NAME}
mkdir -p "$OUT"
cd "$REPO"
uv run python -m pino.object_probe --ckpt "${CKPT:?set CKPT}" --out "$OUT" --lift-views "${LIFT_VIEWS:-24}" "$@" 2>&1 \
    | grep -v -i "warning\|Loading weights" | tee "$OUT/probe.log"
