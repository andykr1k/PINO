#!/bin/bash
# Copies E008's logs (not checkpoints) into logs/ so they get committed.
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
MAIN=$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{print $2; exit}')
RUN_DIR=${RUN_DIR:-$MAIN/outputs/e008_splatformer_baseline}

mkdir -p "$EXP_DIR/logs/vis"
cp "$RUN_DIR/config.json" "$RUN_DIR/metrics.jsonl" "$RUN_DIR/commit.txt" "$EXP_DIR/logs/"
grep -v -i "warning\|Loading weights\|is_fx_tracing" "$RUN_DIR/train.log" > "$EXP_DIR/logs/train.log"
cp "$RUN_DIR"/vis/*.jpg "$EXP_DIR/logs/vis/" 2>/dev/null || true
echo "synced from $RUN_DIR:"
grep "val/" "$EXP_DIR/logs/train.log" | tail -3
