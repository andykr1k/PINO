#!/bin/bash
# PCA renders of per-splat embeddings (pino.pca_render) for a spread of validation scenes, on fixed data.
#   CKPT=/path/to/ckpt.pt GPU=7 bash lab_notebook/experiments/E005_embedding_pca_fixed/scripts/run_pca.sh [--viser]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
CKPT=${CKPT:-$MAIN/outputs/e004_dino_align_fixed_opacity/latest.pt}
SCENES=${SCENES:-"0001_839920 0101_839973 0251_840828 0451_840422 0651_841463 0851_841772"}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=${GPU:-7}

cd "$REPO"
STEP=$(uv run python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$CKPT")
for scene in $SCENES; do
    uv run python -m pino.pca_render --ckpt "$CKPT" --scene "$scene" --out "$EXP_DIR/logs/step${STEP}/${scene}" --tiles-per-batch 1 "$@"
done
