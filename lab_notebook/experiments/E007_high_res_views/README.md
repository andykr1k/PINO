# E007: Higher-resolution training views (896² instead of 448²)

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** running
- **Commit(s):** `ed4d30e`
- **Hypotheses:** H4 (small-object failures), H8

## Question
E006 showed small tabletop objects (spoon, bottle, jar, bowl) are not identifiable from the embeddings; they span
well under one 16-px DINO patch in 448² renders. With 896² renders (56×56 DINO patches, each covering a quarter of
the solid angle), do the distilled embeddings identify objects better, especially small ones?

## Setup
- Single change from E004: `data.image_size=[896,896]` (plus `data.max_splats_per_view=200000` as in E004). Same
  model, loss, schedule (30k steps), data split and seed.
- Engineering (no effect on results): the DINO teacher now runs in chunks of 4 views, and the run uses
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. At 896², peak memory went 76 GB → 39 GB with these.
- Benchmark (`scripts/benchmark_resolution.sh`, `logs/benchmark/`): step time is ~0.75–0.80 s at 448², 672² and
  896². The splat model dominates the step, not the image size, so higher resolution is nearly free.
- 1× H100 (GPU 7); ~0.80 s/step, ~36 GB; ~6.7 h.
- Reproduce: `GPU=7 bash scripts/launch.sh`. Evaluate: `GPU=7 bash scripts/evaluate.sh 010000 030000`.

## Evaluation plan
- **Primary:** the E006 object probe (same settings: 55 classes, 150 train / 20 held-out scenes), on E007's 10k
  and 30k checkpoints vs E004's (E006). The probe reads the 3D embeddings directly, so it is comparable across
  render resolutions.
- Per-class recall for the small-object classes (spoon, bottle, jar, bowl, tray, flowerpot).
- Validation cosine is **not** comparable to E004: finer DINO targets at 896² are a different (harder) task.

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E007-r1 | `data.image_size=[896,896]`, `data.max_splats_per_view=200000` | 30k | `/scratch4/home/adempst/projects/PINO/outputs/e007_high_res_views` | |

## Results
Pending.

## Interpretation
Pending.
