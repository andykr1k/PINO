# E001: Baseline DINOv3 -> splat distillation

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** running
- **Commit(s):** `c43c996`
- **Hypotheses:** H1 (also informs H3)

## Question
With the default configuration, how well do per-splat embeddings from the splat transformer align with DINOv3
patch tokens of rendered views of **held-out** scenes, and does alignment keep improving over 30k steps?

## Setup
- Default `conf/config.yaml`: DINOv3 ViT-L/16 teacher at 448×448 (28×28 patches); 3 scenes × 4 views per batch;
  ≤150k frustum splats per view; SplatTransformer 14.8M params (1024 groups, 8 blocks, width 384, 128-d
  splat embeddings); render loss (weight 1.0, rendered at 1/4 res) + point loss (0.25); standardised targets;
  random z-rotation of model inputs; AdamW lr 3e-4, wd 0.05, 1k warmup, cosine to 1%, 30k steps.
- Data: train = all scenes except every 50th (980); validation = first 20 held-out scenes, fixed cameras.
- Hardware: 1× H100 (GPU 7), ~0.6 s/step, ~5 h total.
- Reproduce: `GPU=7 bash lab_notebook/experiments/E001_dino_align_baseline/scripts/launch.sh`
- Refresh committed logs: `bash scripts/sync_logs.sh`; print the validation curve: `uv run python scripts/summarize_metrics.py`.

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E001-r1 | none | 30k | `/scratch4/home/adempst/projects/PINO/outputs/dino_align_2026-10-27` | Restarted at ~step 100 to pick up `c43c996` fixes. |

## Results
Validation (20 held-out scenes), snapshot of `logs/metrics.jsonl`:

| Step | std cos (render) | raw cos (render) | raw cos (point) | baseline |
|------|------------------|------------------|-----------------|----------|
| 1000 | 0.369 | 0.608 | 0.575 | 0.506 |
| 2000 | 0.447 | 0.630 | 0.614 | 0.506 |

`logs/vis/val_001000.jpg`: predictions capture coarse layout (ceiling / walls / floor / room boundaries) at lower
contrast than DINO; little object-level detail yet.

## Interpretation
- Early support for H1: alignment on unseen scenes clearly exceeds the constant mean-token baseline and is
  still improving. Final numbers pending.
- The DINO PCA's left-right image-position gradient is not reproduced by the view-independent splat embeddings
  (H3), so raw cosine will plateau below 1 for reasons unrelated to 3D understanding.

## Next steps
- Sync logs and fill in final results when the run finishes.
- Linear probe of per-splat embeddings on `labels.json` object classes (H4).
- Loss ablation: render-only vs point-only (H5); raw vs standardised targets matched ablation (H2).
