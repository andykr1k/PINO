# PINO Lab Notebook

The running record of what we believe, what we tested, and what happened. Code lives in `src/pino`;
everything specific to one experiment (launch scripts, probes, logs, visualisations) lives in
`experiments/EXXX_<short_name>/`.

**Conventions**
- One folder per experiment: `experiments/EXXX_<short_name>/` with `README.md` (from the template below),
  `scripts/` and `logs/`. Number experiments sequentially; never renumber.
- An experiment may contain several runs (`EXXX-rN`). Every run gets a row in the [run table](#runs).
- Commit logs (`metrics.jsonl`, filtered `train.log`, `config.json`, `vis/`) — **never checkpoints**. Raw run
  dirs (with checkpoints) live in the gitignored `outputs/` of the main checkout.
- Record the git commit each run used. If a run used uncommitted code, say so and describe the diff.
- Hypotheses are numbered `H#` and never deleted; update their status as evidence comes in.

## Hypotheses

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H1 | A feed-forward native-3D splat transformer can distill 2D DINOv3 patch features into per-splat embeddings that generalise to unseen scenes (beat the constant mean-token baseline on held-out scenes). | **Supported** | E004-r1 (fixed data, 30k steps): held-out raw cosine 0.809 (norm-corrected) vs 0.463 predict-the-mean baseline; standardised cosine 0.723. Curves plateau by ~24k steps. |
| H2 | Cosine loss against *raw* DINO tokens collapses towards predicting the mean token, because tokens share a large common component; standardising targets (running per-channel mean/std) forces content learning. | **Weakly supported** | E000-r1 (raw targets) predicted near-constant maps; E000-r2 (standardised) beat the baseline within 200 steps. Confounded by run length (60 vs 200 steps). Needs a matched ablation. |
| H3 | DINOv3 patch tokens contain a view-dependent image-position component (left-right gradient in PCA) that a view-independent per-splat embedding cannot reproduce, capping achievable alignment. | Untested (observation) | Visible in E000 view grid and E001 vis. Test: measure cosine ceiling of the best view-independent fit, or regress out patch position from targets. |
| H4 | The distilled per-splat embeddings linearly decode object semantics (InteriorGS `labels.json` boxes). | Untested quantitatively; qualitative support | E005 (fixed data, 6 scenes): whole-scene PCA groups splats by category across room types (yellow armchair = white sofa; chairs vs tables separate by 30k). Linear probe planned. |
| H6 | Per-splat embeddings live in DINO's feature geometry but at splat (not 16-px patch) resolution. | **Supported (qualitative)** | E005: one PCA basis fit on splat embeddings colours both the splat render and the DINO tokens consistently in 6 held-out scenes; thin structures (chair legs, chandelier arms) stay sharp. |
| H7 | With a cosine loss in standardised space, prediction norms drift, so any metric/consumer that de-standardises (`pred·std + mean`) must first fix the norm. | **Supported** | E004: |pred| shrank 12.0 → 5.4 (target 33.7); logged raw cosine fell 0.704 → 0.623 while norm-corrected raw cosine rose to 0.809. Fixed in `to_teacher_space`. |
| H5 | The differentiable rendered-feature loss is what drives learning; the per-splat point loss is a cheap, sufficient substitute (or vice versa). | Untested | Planned ablation: `loss.point_weight=0` vs `loss.render_weight=0`. |

## What has been tested

- **Data pipeline (E000).** InteriorGS has no images/cameras, so views are rendered from the splats with gsplat
  using cameras sampled inside `structure.json` room polygons. Rendered views look like plausible interiors;
  visible splats cover 86–99% of DINO patches (median 12–67 splats/patch); 12 views + DINOv3-L in ~0.09 s.
- **Training loop (E000).** End-to-end training runs at ~0.6 s/step for 12 views (~80k splats/view) on one H100
  once DataLoader workers ≥ 24 (8 workers left it data-bound).
- **Target standardisation (E000, H2).** See hypothesis table.
- **Render fidelity (E003).** ⚠️ The loader dropped every fully opaque splat (stored as +inf opacity logit),
  56–58% of each scene, so all renders, DINO targets and model inputs in E000–E002 came from degraded scenes with
  see-through gaps. Fixed in `fc57aaf`; fixed renders are clean (alpha<0.9 pixels 35% → <0.5%).
- **Baseline distillation run (E001).** Stopped after step 4k following E003; superseded by E004.
- **Distillation on fixed data (E004).** 30k steps: held-out raw cosine 0.809 (baseline 0.463), standardised 0.723,
  plateauing. ⚠️ Raw cosine as logged during training (E000–E004 logs) is distorted by prediction-norm drift (H7);
  use `metrics_revalidated.jsonl` / the corrected `to_teacher_space`.
- **Embedding PCA renders (E002).** `python -m pino.pca_render` embeds a whole scene (tiled like training crops),
  colours splats by the top-3 PCA components and renders them next to RGB and the DINO tokens in the same basis.

## Runs

| Run | Date | Commit | Config deltas from default | Steps | Held-out result | Status |
|-----|------|--------|----------------------------|-------|-----------------|--------|
| E000-r1 | 2026-10-27 | uncommitted (pre-`266d042`, raw targets) | 60 steps, warmup 10, 6 val scenes | 60 | raw cos 0.423 vs baseline 0.495; near-constant predictions | Done |
| E000-r2 | 2026-10-27 | `266d042` | 200 steps, warmup 20, 6 val scenes | 200 | raw cos 0.520 vs baseline 0.495; std cos 0.239 | Done |
| E000-r3 | 2026-10-27 | `c43c996` | 30 steps, warmup 5, 3 val scenes (regression check of fixes) | 30 | raw cos 0.393 vs baseline 0.483 (too short to judge) | Done |
| E001-r1 | 2026-10-27 | `c43c996` | none (30k steps, GPU 7) | 4k / 30k | raw cos 0.662 vs baseline 0.506; std cos 0.528 (step 4k) | Stopped — data bug (E003) |
| E002-r1 | 2026-10-27 | `ac2ba3c` | E001-r1 step-2k ckpt; scene 0001_839920 | — | PCA render separates furniture / floor / walls / ceiling; matches DINO in shared basis | Done |
| E002-r2 | 2026-10-27 | `ac2ba3c` | E001-r1 step-2k ckpt; scene 0101_839973 | — | seats vs carpet separated despite same RGB colour; no tile seams | Done (pre-fix data) |
| E003 | 2026-10-27 | `fc57aaf` | render comparison, old vs fixed loader | — | opaque splats were dropped; fixed renders gap-free | Done |
| E004-r1 | 2026-10-27 | `fc57aaf` | `data.max_splats_per_view=200000` | 30k | raw cos 0.809 (norm-corrected) vs baseline 0.463; std cos 0.723; val loss 0.359 | Done |
| E005-r1 | 2026-10-27 | `760a296` | E004-r1 step-2k and 4k ckpts; 6 val scenes | — | PCA renders separate ceiling/wall/floor/furniture consistently with DINO across room types | Done |
| E005-r2 | 2026-10-28 | this commit | E004-r1 final (30k) ckpt; 6 val scenes; L2-normalised PCA | — | finer categories (chairs vs tables, curtains vs walls); DINO agrees | Done |

Metric glossary: *raw cos* = `val/render/cos_raw`, cosine between rendered predicted features (mapped back to
DINO space) and DINO patch tokens; *std cos* = `val/render/cos`, same in standardised target space (constant
prediction ≈ 0); *baseline* = `val/baseline/cos`, cosine of each token with the batch-mean token.

## TEMPLATE

Copy into `experiments/EXXX_<short_name>/README.md`:

```markdown
# EXXX: <title>

- **Date:** YYYY-MM-DD
- **Owner:** <name>
- **Status:** planned | running | done | abandoned
- **Commit(s):** <hash> (note any uncommitted changes)
- **Hypotheses:** H# (link to main README)

## Question
What are we trying to learn, in one or two sentences?

## Setup
- Code / config changes relative to default (`conf/config.yaml`):
- Data: scenes / splits used.
- Hardware: GPU(s), wall-clock.
- How to reproduce: `bash scripts/<launch script>`

## Runs
| Run | Config deltas | Steps | Output dir | Notes |
|-----|---------------|-------|------------|-------|
| EXXX-r1 | | | | |

## Results
Key numbers (held-out), with the baseline alongside. Reference figures in `logs/`.

## Interpretation
What this means for the hypotheses. Be explicit about confounds and what was *not* controlled.

## Next steps
- 
```
