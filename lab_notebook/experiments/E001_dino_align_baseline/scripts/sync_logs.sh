#!/bin/bash
# Copies the run's logs (not checkpoints) into this experiment's logs/ folder so they get committed.
#   RUN_DIR=/path/to/run bash lab_notebook/experiments/E001_dino_align_baseline/scripts/sync_logs.sh
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
RUN_DIR=${RUN_DIR:-$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')/outputs/dino_align_2026-10-27}

mkdir -p "$EXP_DIR/logs/vis"
cp "$RUN_DIR/config.json" "$RUN_DIR/metrics.jsonl" "$EXP_DIR/logs/"
grep -v -i "warning\|Loading weights" "$RUN_DIR/train.log" > "$EXP_DIR/logs/train.log"
cp "$RUN_DIR"/vis/*.jpg "$EXP_DIR/logs/vis/"
echo "synced from $RUN_DIR:"
grep "val/" "$EXP_DIR/logs/train.log" | tail -3
