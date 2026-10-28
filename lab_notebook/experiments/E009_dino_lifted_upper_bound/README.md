# E009: DINO upper bound — teacher features lifted onto splats

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** sanity check done; probe runs queued after E007 finishes
- **Commit(s):** `99b07db` (pino.lift, model_views)
- **Hypotheses:** H10, H11

## Question
How well do DINO's own features, projected onto the splats, identify objects? That bounds what the distilled student
could reach, and tells us whether small-object failures (E006/E007) are the student's fault or the teacher's.

## Setup
- `src/pino/lift.py`: for each view, a splat's lifted feature accumulates Σ_p w_ip F_p (w_ip its alpha-blending
  weight at pixel p, F_p the DINO token covering p). Rendering is linear in the colours, so this is exactly the
  gradient of ⟨render(colors), F⟩ w.r.t. the colours: one backward pass per view gives it for all splats, with
  occlusion handled by the blend weights. Dividing by Σ_p w_ip gives the blend-weighted mean token over all views.
  Feature maps are rendered at 1/4 resolution, as in training.
- Views: the training camera sampler and quality filter (best 24 of 72 candidates per scene).
- Probe additions (`pino.object_probe --lift-views N`):
  - **dino_lifted**: lifted teacher tokens, standardised with the student's running target statistics.
  - **model_views**: the student run as in training, one frustum crop per lifting view, with unit-norm projected
    embeddings averaged per splat over the crops. This is the like-for-like counterpart of dino_lifted.
  - All feature sets are scored on the same splats: visible in ≥ 1 lifting view and inside ≥ 1 crop.
- Scripts: `scripts/lift_sanity.py`, `scripts/run_upper_bound.sh` (`CKPT=… NAME=… bash …`).

## Sanity check (`logs/lift_sanity.txt`, E004 step 30k, 16 lifting views, standardised patch cosine to DINO)

| Scene | lifted, seen views | lifted, held-out view | student, whole-scene tiles | student, frustum crops |
|-------|--------------------|-----------------------|----------------------------|------------------------|
| 0001_839920 | 0.920 | 0.478 | 0.416 | 0.644 |
| 0451_840422 | 0.896 | 0.394 | 0.393 | 0.727 |

- Lifting is correct: it reproduces the teacher on the views it came from (0.90–0.92).
- Lifted features transfer poorly to a held-out view (0.39–0.48): consistent with H3 (view-dependent DINO tokens),
  and only 70–77% of splats are seen by 16 views.
- **H11: whole-scene tiled inference (used by E005 PCA renders and the E006/E007 probes) costs the student 0.23–0.33
  cosine** relative to the frustum crops it was trained on. Tiles are out of distribution, so those probe numbers
  understate the student (the E004 vs E007 comparison is still like-for-like). A frustum crop also depends on the
  camera, which lets the student model view-dependent structure; hence model_views for a fair upper-bound comparison.

## Runs (queued)
| Run | Student ckpt | Lifting resolution | Logs |
|-----|--------------|--------------------|------|
| E009-r1 | E004 step 30k | 448² | `logs/e004_448/` |
| E009-r2 | E007 step 30k | 896² | `logs/e007_896/` |

## Results
Pending.
