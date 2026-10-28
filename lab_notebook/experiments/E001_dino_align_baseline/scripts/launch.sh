#!/bin/bash
# Launches the E001 baseline run (default config, 30k steps) detached from the shell.
#   GPU=7 RUN_DIR=/path/to/run bash lab_notebook/experiments/E001_dino_align_baseline/scripts/launch.sh
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
RUN_DIR=${RUN_DIR:-$REPO/outputs/dino_align_$(date +%F)}
GPU=${GPU:-7}

cd "$REPO"
mkdir -p "$RUN_DIR"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}  # gsplat JIT-compiles its kernels with nvcc
export CUDA_VISIBLE_DEVICES=$GPU
setsid nohup uv run python -m pino.train +paths.output_dir="$RUN_DIR" > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "launched pid $! -> $RUN_DIR"
