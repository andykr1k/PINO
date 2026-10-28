#!/bin/bash
# Object-identification probe (pino.object_probe) on an E004 checkpoint.
#   CKPT=/path/to/ckpt.pt GPU=7 bash lab_notebook/experiments/E006_object_probe/scripts/run_probe.sh [extra args]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
CKPT=${CKPT:-$MAIN/outputs/e004_dino_align_fixed_opacity/ckpt_030000.pt}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=${GPU:-7}

cd "$REPO"
STEP=$(uv run python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$CKPT")
OUT=$EXP_DIR/logs/step${STEP}
mkdir -p "$OUT"
uv run python -m pino.object_probe --ckpt "$CKPT" --out "$OUT" "$@" 2>&1 | grep -v -i "warning\|Loading weights" | tee "$OUT/probe.log"
