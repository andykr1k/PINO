#!/bin/bash
# Runs the object probe on several E004 checkpoints to see how object information develops during training.
#   GPU=7 bash lab_notebook/experiments/E006_object_probe/scripts/sweep_checkpoints.sh [step ...]
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
MAIN=$(git -C "$DIR" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
for step in "${@:-002000 010000}"; do
    CKPT=$MAIN/outputs/e004_dino_align_fixed_opacity/ckpt_${step}.pt bash "$DIR/run_probe.sh" | grep -E "model:|random:|attributes:"
done
