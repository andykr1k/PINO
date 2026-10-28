#!/bin/bash
# Renders splat-embedding PCA for the E002 scenes with a given checkpoint.
#   CKPT=/path/to/latest.pt GPU=7 bash lab_notebook/experiments/E002_embedding_pca/scripts/run_pca.sh [--viser]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
CKPT=${CKPT:-$MAIN/outputs/dino_align_2026-10-27/latest.pt}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=${GPU:-7}

cd "$REPO"
for scene in 0001_839920 0101_839973; do
    uv run python -m pino.pca_render --ckpt "$CKPT" --scene "$scene" --out "$EXP_DIR/logs/${scene}_dino" "$@"
done
