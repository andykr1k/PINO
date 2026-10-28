# E007: Higher-resolution training views (896² instead of 448²)

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** running
- **Commit(s):** `ed4d30e`
- **Hypotheses:** H4 (small-object failures), H8

## Question
E006 showed small tabletop objects (spoon, bottle, jar, bowl) are not identifiable from the embeddings; they span
well under one 16-px DINO patch in 448² renders. DINO's patch is fixed at 16×16 px, so the patch grid scales with
image side length: 28×28 at 448² vs 56×56 at 896². The field of view is unchanged, so each patch covers a quarter of
the scene area it did before, and an object spans ~4× as many patches (views are rendered natively at 896², not
upscaled). Do the distilled embeddings then identify objects better, especially small ones?

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
Interim, step 10000 (`logs/probe_step10000/`), E006 probe, balanced accuracy, held-out scenes:

| Level | E004 (448²) @10k | E007 (896²) @10k | attributes | random |
|-------|------------------|------------------|------------|--------|
| splat | 0.488 | 0.475 | 0.214 | 0.158 |
| instance | 0.607 | 0.603 | 0.499 | 0.254 |

Small-object splat recall E004 → E007 @10k: spoon 0.11 → 0.08, bottle 0.00 → 0.05, jar 0.00 → 0.02, bowl 0.03 →
0.03, tray 0.02 → 0.04, cup 0.16 → 0.23. Per-class differences of up to ±0.2 in both directions are within the
noise of a few dozen test objects per class.

Validation std cosine @12k: 0.684 (E004 @12k: 0.681; not strictly comparable, finer targets).

## Interpretation
- Interim: **no evidence for H8 at 10k.** 4× more DINO patches per object in the supervision did not improve object
  identification, including for small objects. Final (30k) probe pending.
- Leading hypothesis: the bottleneck is the student's spatial resolution, not the targets. With 1024 FPS groups for
  ≤200k splats per view (~200 splats per group), a small object's few dozen splats share a group (token) with the
  surface below. Test: `model.num_groups` 2048/4096. A DINO upper bound (teacher tokens lifted onto splats) would
  separately show whether the teacher distinguishes these objects at all.
