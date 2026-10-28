#!/bin/bash
# Short training runs at several render resolutions to measure step time and peak GPU memory.
#   GPU=7 bash lab_notebook/experiments/E007_high_res_views/scripts/benchmark_resolution.sh [size ...]
set -euo pipefail
EXP_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(git -C "$EXP_DIR" rev-parse --show-toplevel)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export CUDA_VISIBLE_DEVICES=${GPU:-7}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
mkdir -p "$EXP_DIR/logs/benchmark"
for size in "${@:-448 672 896}"; do
    out=$EXP_DIR/logs/benchmark/size_${size}
    rm -rf "$out"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" -l 2 > "$out.mem" &
    smi=$!
    uv run python -m pino.train +paths.output_dir="$out" data.image_size="[$size,$size]" data.max_splats_per_view=200000 \
        hparams.max_steps=75 hparams.warmup_steps=10 train.log_every=25 train.val_every=100000 train.ckpt_every=100000 \
        2>&1 | grep -E "^\[|Error|Traceback|OutOfMemory" | cut -c1-200
    kill $smi
    rm -f "$out"/*.pt
    echo "size $size: peak GPU memory $(sort -n "$out.mem" | tail -1) MiB"
done
