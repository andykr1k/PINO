# E009: DINO upper bound — teacher features lifted onto splats

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done
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

## Runs
| Run | Student ckpt | Lifting resolution | Logs |
|-----|--------------|--------------------|------|
| smoke | E004 step 30k | 448², 4 views, 3 train / 2 val scenes | `logs/smoke/` (code check only) |
| E009-r1 | E004 step 30k | 448², 24 views | `logs/e004_448/` |
| E009-r2 | E007 step 30k | 896², 24 views | `logs/e007_896/` |

Same probe settings as E006 (55 classes, 150 training and 20 held-out scenes, class-balanced logistic regression).
Splats are restricted to those seen by the lifting views and inside a crop, so `model` numbers differ slightly from
E006/E007.

## Results
Balanced accuracy on held-out scenes (chance 0.018):

| Features | r1 splat (448²) | r1 instance | r2 splat (896²) | r2 instance |
|----------|-----------------|-------------|-----------------|-------------|
| **dino_lifted** (teacher upper bound) | **0.505** | **0.600** | **0.540** | **0.627** |
| model_views (student, training-style crops, averaged) | 0.532 | 0.596 | 0.544 | 0.623 |
| model (student, whole-scene tiles) | 0.502 | 0.573 | 0.515 | 0.590 |
| attributes | 0.203 | 0.437 | 0.214 | 0.471 |
| random-init model | 0.161 | 0.244 | 0.160 | 0.246 |

Plain accuracy is higher for dino_lifted (r2: splat 0.603 vs 0.521, instance 0.697 vs 0.595). The two feature sets
win on different classes, so balanced accuracy is about level.

Small-object splat recall (model / model_views / dino_lifted):

| Class | r1 (448²) | r2 (896²) |
|-------|-----------|-----------|
| spoon | 0.00 / 0.00 / 0.00 | 0.07 / 0.19 / 0.49 |
| bottle | 0.00 / 0.05 / 0.29 | 0.02 / 0.02 / 0.22 |
| jar | 0.00 / 0.00 / 0.04 | 0.02 / 0.00 / 0.15 |
| bowl | 0.00 / 0.15 / 0.22 | 0.04 / 0.17 / 0.56 |
| cup | 0.20 / 0.28 / 0.59 | 0.19 / 0.35 / 0.63 |
| flowerpot | 0.02 / 0.17 / 0.31 | 0.14 / 0.06 / 0.65 |
| book | 0.31 / 0.21 / 0.53 | 0.38 / 0.28 / 0.63 |

The student beats lifted DINO on view-independent, geometry-distinct classes such as shower head (+0.6), bath heater,
gas stoves, pillow and quilt.

## Interpretation
- **H10: partly refuted.** Lifted DINO is **not** an overall upper bound. The student, evaluated like-for-like
  (model_views), matches it on balanced accuracy at both resolutions.
  - The student generalises DINO's semantics through 3D geometry. Averaging over views with lifting only covers what
    the 24 views saw, and lifted tokens carry view-dependent noise (H3).
- **Small objects are the student's failure, not the teacher's.** DINO's lifted features identify cups, bowls,
  books, flowerpots and (at 896²) spoons at 0.5–0.65 recall, while the student stays at 0–0.35.
  - The teacher's advantage on small objects grows with resolution: bowl 0.22 → 0.56, flowerpot 0.31 → 0.65, spoon
    0.00 → 0.49. So the 896² targets do contain the information, but the student only partially absorbs it (H8 weak).
  - This points at the student's spatial bottleneck (1024 groups of ~200 splats; a small object shares a token with
    its support surface). Next test: `model.num_groups`.
- **H11 confirmed on the probe.** model_views beats tiled model by +0.03 (both levels, both runs).
- Confounds:
  - one probe seed and 20 validation scenes;
  - dino_lifted uses the student's running target statistics for standardisation;
  - lifting views use the training camera sampler, so rarely viewed regions are excluded from scoring.

## Next steps
- Student spatial resolution: `model.num_groups` 2048/4096 (and/or more kNN neighbours in the decoder) at 896².
- Make tiled/whole-scene inference in-distribution (train on view-independent crops), or adopt `embed_views` as the
  standard evaluation.
- Probe seeds / bootstrap error bars over scenes.
