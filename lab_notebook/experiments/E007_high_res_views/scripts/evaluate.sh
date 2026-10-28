#!/bin/bash
# E007 evaluation: object probe (E006 tool, same settings) on E007 checkpoints, written to this experiment's logs.
#   GPU=7 bash lab_notebook/experiments/E007_high_res_views/scripts/evaluate.sh [step ...]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
MAIN=$(git -C "$EXP_DIR" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
PROBE=$EXP_DIR/../E006_object_probe/scripts/run_probe.sh
for step in "${@:-010000 030000}"; do
    CKPT=$MAIN/outputs/e007_high_res_views/ckpt_${step}.pt OUT=$EXP_DIR/logs/probe_step$((10#$step)) \
        bash "$PROBE" | grep -E "model:|random:|attributes:|majority"
done
