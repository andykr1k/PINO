#!/bin/bash
# E004: E001's baseline config re-run after the +inf-opacity loader fix, with a larger per-view splat cap
# (scenes keep ~2.5x more splats now). Detached from the shell.
#   GPU=7 RUN_DIR=/path/to/run bash lab_notebook/experiments/E004_dino_align_fixed_opacity/scripts/launch.sh
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
RUN_DIR=${RUN_DIR:-$MAIN/outputs/e004_dino_align_fixed_opacity}
GPU=${GPU:-7}

cd "$REPO"
mkdir -p "$RUN_DIR"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=$GPU
setsid nohup uv run python -m pino.train +paths.output_dir="$RUN_DIR" data.max_splats_per_view=200000 \
    > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "launched pid $! -> $RUN_DIR"
