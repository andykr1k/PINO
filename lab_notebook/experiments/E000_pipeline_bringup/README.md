# E000: Data pipeline and training-loop bring-up

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done
- **Commit(s):** r1 uncommitted (pre-`266d042`, no target standardisation); r2 `266d042`; r3 `c43c996`
- **Hypotheses:** H2

## Question
Can we produce (rendered image, DINOv3 tokens, visible splats, splat->patch correspondence) training
examples from InteriorGS, and does the distillation loop train at all?

## Setup
- InteriorGS at `/z/dat/InteriorGS`: 1000 scenes with `3dgs_uncompressed.ply` (identical 59-float header,
  ~0.4–2.1M splats each), plus `labels.json`, `structure.json`, `occupancy.json`. **No images or cameras.**
- Teacher: HF `facebook/dinov3-vitl16-pretrain-lvd1689m` from the local HF cache (the Meta signed download URLs
  in `aidan-lib/.env` returned 403).
- gsplat 1.5.3 JIT-compiled against torch 2.12 / CUDA 13.0 (`CUDA_HOME=/usr/local/cuda-13.0`, run via `uv run`
  so `ninja` is on PATH). First build ~47 s, cached afterwards.
- Scripts:
  - `scripts/probe_dataset.py`: dataset survey and PLY load speed.
  - `scripts/smoke_dino_gsplat.py`: teacher load + gsplat render check.
  - `scripts/check_view_pipeline.py`: full view pipeline on 3 scenes; writes `logs/view_pipeline_grid.jpg`.
- Smoke training runs used `uv run python -m pino.train` with the overrides in the table, on GPU 6/7.

## Runs
| Run | Config deltas | Steps | Logs | Notes |
|-----|---------------|-------|------|-------|
| E000-r1 | `hparams.max_steps=60 hparams.warmup_steps=10 train.val_scenes=6` | 60 | `logs/smoke_run1/` | Raw DINO targets (code before standardisation was added); `data.num_workers=8` |
| E000-r2 | `hparams.max_steps=200 hparams.warmup_steps=20 train.val_scenes=6` | 200 | `logs/smoke_run2/` | Standardised targets; `num_workers=24` |
| E000-r3 | `hparams.max_steps=30 hparams.warmup_steps=5 train.val_scenes=3` | 30 | `logs/smoke_run3/` | Regression check of empty-view guard / pixel-floor fixes |

## Results
- Dataset load: ~1.1 s cold per scene with `np.fromfile` (vs. several seconds with plyfile).
- View pipeline (`logs/view_pipeline_grid.jpg`): 3 scenes × 24 candidates rendered, filtered to 12 views, embedded
  by DINOv3-L in **0.09 s**. Visible splats cover 86–99% of the 28×28 patches; median 12–67 visible splats per
  patch; frustum crops often hit the 150k cap.
- r1: `val/render/cos` (raw space at the time) 0.423 vs baseline 0.495; predicted PCA maps nearly uniform
  (`logs/smoke_run1/vis/val_000060.jpg`). Data-bound: `data_wait_s` 0.1–1.9 s against ~0.6 s compute.
- r2: raw cos 0.520 vs baseline 0.495, standardised cos 0.239 (rising), point cos 0.286. Data wait ≈ 0.
- r3: ran cleanly; 30 steps too short to read anything into the numbers.

## Interpretation
- The pipeline produces sensible renders and DINO features (floor/wall/ceiling separate cleanly in the PCA).
- H2: consistent with mean collapse under raw targets, but r1 vs r2 differ in run length (60 vs 200 steps) and
  worker count, so this is not a controlled comparison.
- DINO PCA shows a strong left-right image-position gradient (see H3).

## Next steps
- Matched ablation for H2 (raw vs standardised targets, same steps/seed).
- Full-length baseline run → E001.
