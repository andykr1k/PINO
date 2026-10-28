# E008: SplatFormer (PTv3) baseline encoder

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** running
- **Commit(s):** `ca81393` (branch `worktree-splatformer-baseline`, on top of E007's `bc45d9d`)
- **Hypotheses:** H9 (also feeds H1, H8)

## Question
Does SplatFormer's Point Transformer V3 encoder distill DINOv3 into per-splat embeddings better than our
SplatTransformer, with identical data, losses, optimiser and step budget?

## Setup
- Reference run: **E007-r1** (896² views, 200k splats/view, SplatTransformer). E008 uses E007's exact config and
  code plus `model.arch=splatformer`; nothing else changes (same seed, split, schedule, 30k steps).
- Encoder (`src/pino/splatformer.py`, vendored PTv3 in `src/pino/third_party/pointcept/`), following
  SplatFormer (github.com/ChenYutongTHU/SplatFormer @ `446ffb5`, `configs/model/ptv3.gin`) and its pinned Pointcept
  fork (`c4aa232`):
  - every splat is a token; each view is min-max normalised into the unit cube (aspect kept) and voxelised at 1/384
    for serialisation and the sparse-conv positional encoding; log-scales shift by the same factor;
  - PTv3 U-Net: MLP embedding; encoder depths (2,2,2,6,2), channels (64,96,128,256,512), heads (2,4,8,16,32);
    pooling strides (1,2,2,2); decoder depths (2,2,2,2), channels (96,96,128,256); attention over 1024-splat
    patches along z / z-trans / hilbert / hilbert-trans curves, curve changing per block and shuffled every
    forward; drop path 0.3;
  - 4-layer ReLU MLP head (width 128) on [backbone feature, input feature] -> 128-d embedding -> shared linear
    projection to DINO space (same as SplatTransformer).
  - 47.85M parameters (SplatTransformer: 14.83M).
- Deliberate differences from SplatFormer:
  - inputs omit higher-order SH (`features_rest`), which `PackedSplats` does not carry; SH DC comes from the clamped
    RGB. SplatTransformer sees the same attributes;
  - the head outputs an embedding, not residual Gaussian updates (no residual activations or zero-init);
  - SplatFormer's own training hyperparameters are not used: optimiser, LR schedule and losses are ours, shared
    with E007.
- Engineering substitutions (no change to the maths, see `src/pino/third_party/pointcept/README.md`):
  - flash-attn -> torch SDPA over the same packed patches;
  - `torch_scatter.segment_csr` -> `torch.segment_reduce`;
  - bf16 -> fp32 before spconv in no-grad paths;
  - per-block activation checkpointing.
- Checks (`scripts/check_splatformer.py`):
  - SDPA varlen matches per-sequence attention (max err 6e-7);
  - `segment_reduce` matches `segment_csr`;
  - checkpointed vs plain: identical outputs, gradients within 8e-6 relative;
  - E004 checkpoints still load.
- Cost (`scripts/profile_real.py`; smoke runs):
  - at 448²: peak 23.6 GB vs 24.5 GB for SplatTransformer, 2.4 s vs 0.9 s per step;
  - at 896² (whole process, `nvidia-smi`): ~37 GB, ~2.7 s/step, so ~23 h for 30k steps;
  - splats per encoder stage on a 1.58M-splat batch: 1577k -> 802k -> 414k -> 172k -> 55k (the stride-1 first
    pooling merges splats sharing a 1/384 voxel).
- Hardware: 1× H100, GPU 4 (shared with another user's light robosim server). GPU 5 refuses new CUDA contexts
  ("busy or unavailable"); GPU 6 holds another job of ours that will grow.
- Reproduce: `GPU=4 bash scripts/launch.sh`; refresh committed logs with `bash scripts/sync_logs.sh`.

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E008-r1 | E007-r1 + `model.arch=splatformer` | 30k | `/scratch4/home/adempst/projects/PINO/outputs/e008_splatformer_baseline` | |

## Results
Pending. Compare against E007-r1 at matched steps:
- validation cosines on the 20 held-out scenes (896² targets, so not comparable to E004);
- the E006 object probe (`python -m pino.object_probe`) on the 10k and 30k checkpoints of both runs.

## Interpretation
Pending. Confounds to keep in mind:
- SplatFormer has 3.2× the parameters, so a win does not isolate the architecture;
- it trains ~3.4× slower per step, so a matched step count is not matched compute;
- its hyperparameters were tuned for a different task (novel-view refinement on objects), not for this one.

## Next steps
- Sync logs at checkpoints (`scripts/sync_logs.sh`), and run the object probe at 10k and 30k for both E007 and
  E008.
- If PTv3 wins, ablate what drives it (per-splat tokens vs changing groups vs multi-scale) with a size-matched
  variant.
