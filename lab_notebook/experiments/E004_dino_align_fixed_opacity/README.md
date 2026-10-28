# E004: Baseline distillation on fixed data (opaque splats kept)

- **Date:** 2026-10-27 → 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done (30k steps)
- **Commit(s):** trained at `fc57aaf`; raw-cosine metric corrected afterwards (see below)
- **Hypotheses:** H1, H6, H7 (supersedes E001)

## Question
Same as E001, after the E003 loader fix: how well do per-splat embeddings align with DINOv3 tokens of rendered
views of held-out scenes when the scenes are complete?

## Setup
- E001 configuration, plus `data.max_splats_per_view=200000` (was 150k), since scenes now keep ~2.5x more splats.
- 1× H100 (GPU 7); ~0.86 s/step, ~65 GB; ~7.5 h for 30k steps. Checkpoints every 2k steps.
- Reproduce: `GPU=7 bash scripts/launch.sh`; refresh committed logs with `bash scripts/sync_logs.sh`.
- Analysis scripts:
  - `scripts/plot_val_curves.py` → `logs/val_curves.png`
  - `scripts/norm_check.py` → `logs/norm_check.txt` (prediction-norm diagnostic)
  - `scripts/revalidate.py` → `logs/metrics_revalidated.jsonl`, `logs/vis_revalidated/` (re-scores every checkpoint
    with the corrected raw metric)

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E004-r1 | `data.max_splats_per_view=200000` | 30k | `/scratch4/home/adempst/projects/PINO/outputs/e004_dino_align_fixed_opacity` | |

## Results
Validation on 20 held-out scenes (`logs/val_curves.png`). Raw cosine is the **norm-corrected** re-validation; the
other columns are as logged during training (re-validation reproduces std cos to 4 decimals).

| Step | val loss | std cos (render) | raw cos (render, corrected) | baseline |
|------|----------|------------------|-----------------------------|----------|
| 2000 | 0.638 | 0.492 | 0.623 | 0.463 |
| 4000 | 0.551 | 0.565 | 0.675 | 0.463 |
| 6000 | 0.477 | 0.625 | 0.738 | 0.463 |
| 10000 | 0.430 | 0.665 | 0.765 | 0.463 |
| 20000 | 0.374 | 0.711 | 0.799 | 0.463 |
| 30000 | **0.359** | **0.723** | **0.809** | 0.463 |

All curves improve monotonically and plateau over the last ~6k steps (cosine LR decay).

**Metric artifact (H7).** As logged during training, raw cosine peaked at 0.704 (step 7k) and then *fell* to 0.623,
on training batches too, while every other metric improved. Cause: the loss is a cosine in standardised space, so
prediction norms are unconstrained, and they shrank from 12.0 (2k) to 5.4 (30k) vs a target norm of 33.7
(`logs/norm_check.txt`). The old `to_teacher_space` computed `pred * std + mean`, so as |pred| shrank the result
collapsed towards the mean token. `to_teacher_space` now rescales predictions to the expected standardised-target
norm sqrt(D) first; on the same splats, corrected raw cosine rises 0.552 → 0.661 → 0.701 → 0.727 (2k/6k/14k/30k).

`render/valid_frac` ≈ 0.99 vs ≈ 0.96 in E001 (fewer uncovered patches).

## Interpretation
- H1 supported on complete data: held-out alignment 0.809 raw cosine vs 0.463 for the predict-the-mean baseline,
  0.723 in standardised space (constant prediction ≈ 0).
- Returns are diminishing at 30k steps with this model size (14.8M params); more capacity or data diversity is the
  likely lever, not more steps at this LR schedule.
- Only direction of the projected embeddings is meaningful; anything consuming them should L2-normalise.

## Next steps
- Quantitative semantic probe (H4) on the final checkpoint.
- Consider adding a norm term or normalising predictions inside the loss, so norms don't drift.
