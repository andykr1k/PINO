#!/bin/bash
# E007: E004's configuration with 896x896 training renders (56x56 DINO patches instead of 28x28). Detached.
#   GPU=7 RUN_DIR=/path/to/run bash lab_notebook/experiments/E007_high_res_views/scripts/launch.sh [overrides, e.g. train.resume=...]
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
RUN_DIR=${RUN_DIR:-$MAIN/outputs/e007_high_res_views}
GPU=${GPU:-7}

cd "$REPO"
mkdir -p "$RUN_DIR"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 76 GB -> 39 GB peak at 896px (fragmentation)
setsid nohup uv run python -m pino.train +paths.output_dir="$RUN_DIR" data.max_splats_per_view=200000 \
    data.image_size="[896,896]" "$@" >> "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "launched pid $! -> $RUN_DIR"
