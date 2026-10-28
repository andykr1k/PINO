# E004: Baseline distillation on fixed data (opaque splats kept)

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** running
- **Commit(s):** `fc57aaf`
- **Hypotheses:** H1, H6 (supersedes E001)

## Question
Same as E001, after the E003 loader fix: how well do per-splat embeddings align with DINOv3 tokens of rendered
views of held-out scenes when the scenes are complete?

## Setup
- E001 configuration, plus `data.max_splats_per_view=200000` (was 150k), since scenes now keep ~2.5x more splats.
- 1× H100 (GPU 7); ~0.86 s/step, ~65 GB; ~7.5 h for 30k steps.
- Reproduce: `GPU=7 bash scripts/launch.sh`; refresh committed logs with `bash scripts/sync_logs.sh`;
  validation curve: `uv run python ../E001_dino_align_baseline/scripts/summarize_metrics.py logs/metrics.jsonl`.

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E004-r1 | `data.max_splats_per_view=200000` | 30k | `/scratch4/home/adempst/projects/PINO/outputs/e004_dino_align_fixed_opacity` | |

## Results
Validation curves: `logs/val_curves.png` (regenerate with `uv run python scripts/plot_val_curves.py`; reads the
live run dir). E001 is drawn faintly for reference only: its validation scenes were also missing their opaque
splats, so its numbers are on different data and not directly comparable.

| Step | val loss | std cos (render) | raw cos (render) | baseline |
|------|----------|------------------|------------------|----------|
| 1000 | 0.738 | 0.413 | 0.605 | 0.463 |
| 2000 | 0.638 | 0.492 | 0.645 | 0.463 |
| 3000 | 0.583 | 0.538 | 0.666 | 0.463 |

`render/valid_frac` ≈ 0.99 vs ≈ 0.96 in E001 (fewer uncovered patches).

## Interpretation
Pending.

## Next steps
- PCA renders (E002 tool) on several validation scenes once a few thousand steps are in.
