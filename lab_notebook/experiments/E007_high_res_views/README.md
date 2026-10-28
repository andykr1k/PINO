# E007: Higher-resolution training views (896² instead of 448²)

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done
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
- Reproduce: `GPU=7 bash scripts/launch.sh`. Copy logs: `bash scripts/sync_logs.sh`. Evaluate: `GPU=7 bash scripts/evaluate.sh 010000 030000`.

## Evaluation plan
- **Primary:** the E006 object probe (same settings: 55 classes, 150 train / 20 held-out scenes), on E007's 10k
  and 30k checkpoints vs E004's (E006). The probe reads the 3D embeddings directly, so it is comparable across
  render resolutions.
- Per-class recall for the small-object classes (spoon, bottle, jar, bowl, tray, flowerpot).
- Validation cosine is **not** comparable to E004: finer DINO targets at 896² are a different (harder) task.

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| E007-r1 | `data.image_size=[896,896]`, `data.max_splats_per_view=200000` | 30k | `/scratch4/home/adempst/projects/PINO/outputs/e007_high_res_views` | crashed at ~14.95k, resumed from 14k; finished |

## Incident: crash at step ~14,950 and resume
- The run died with a CUDA `index out of bounds` assert in an ATen index kernel (the original log is in the run dir
  as `train_part1_crashed.log`), and the first resume hung with the GPU at 100%. Cause (most likely): gsplat counts
  tile intersections in 32 bits (`uint32_t n_isects`, int32 offsets). Each step renders 24 whole-scene candidate
  cameras in one call; at 896² (4× the tiles of 448²) a batch with several cameras inside geometry, where near
  splats cover every tile, can overflow it. E004 at 448² never hit this in 30k steps. A concurrent lifting job on
  the same GPU also slowed the step just before the crash; memory pressure can't be fully ruled out.
- Fix (`99b07db`): `ViewGenerator._render` renders 4 cameras per gsplat call. Outputs are identical.
- Resumed from `ckpt_014000.pt` with `train.resume=…` (model, optimizer, step and target statistics restored; ~950
  steps redone with different random batches). Nothing else runs on GPU 7 until the run finishes.

## Results
Final, step 30000 (`logs/probe_step30000/`). E006 probe with whole-scene tiled inference, balanced accuracy on
held-out scenes:

| Level | E004 (448²) @30k | E007 (896²) @30k | attributes | random |
|-------|------------------|------------------|------------|--------|
| splat | 0.499 | 0.513 | 0.214 | 0.158 |
| instance | 0.573 | 0.604 | 0.499 | 0.254 |

Small-object splat recall, E004 → E007 @30k: spoon 0.00 → 0.26, bottle 0.00 → 0.03, jar 0.00 → 0.03, bowl 0.01 →
0.14, tray 0.03 → 0.08, flowerpot 0.04 → 0.16, cup 0.28 → 0.17.

Validation @30k: std cosine 0.728 (E004 0.723), raw cosine 0.815, val loss 0.352. Logs in `logs/` (`train.log`,
`metrics.jsonl`, `vis/`). The 896² targets are a different task, so these cosines are not strictly comparable.

Under the fairer view-averaged evaluation (E009), E007 beats E004 by ~+0.01–0.03 as well (splat 0.544 vs 0.532,
instance 0.623 vs 0.596). There, though, each student is scored against its own resolution's lifting views.

Interim, step 10000 (`logs/probe_step10000/`):

| Level | E004 (448²) @10k | E007 (896²) @10k |
|-------|------------------|------------------|
| splat | 0.488 | 0.475 |
| instance | 0.607 | 0.603 |

## Interpretation
- **H8: weakly supported at 30k.** Higher-resolution supervision gives a small overall gain (+0.014 splat, +0.031
  instance) and moves several small classes off zero (spoon, bowl, flowerpot). At 10k there was no gain, so the
  benefit appears late in training.
- The probe uses one seed and 20 validation scenes, and per-class recall swings by about ±0.2. The per-class gains
  are suggestive, not significant.
- Small objects remain far below DINO's own lifted features (E009). The student's spatial resolution (1024 groups) is
  still the leading suspect. Next test: `model.num_groups` 2048/4096.
