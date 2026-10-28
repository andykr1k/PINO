#!/bin/bash
# E008: SplatFormer (PTv3) baseline with E007-r1's exact data/loss/optimiser config (896px views, 200k splats/view);
# only the encoder differs.
# Detached from the shell.
#   GPU=4 RUN_DIR=/path/to/run bash lab_notebook/experiments/E008_splatformer_baseline/scripts/launch.sh
set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
RUN_DIR=${RUN_DIR:-$MAIN/outputs/e008_splatformer_baseline}
GPU=${GPU:-4}

cd "$REPO"
mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/commit.txt"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup uv run python -m pino.train +paths.output_dir="$RUN_DIR" data.image_size=[896,896] data.max_splats_per_view=200000 \
    model.arch=splatformer \
    > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "launched pid $! -> $RUN_DIR"
