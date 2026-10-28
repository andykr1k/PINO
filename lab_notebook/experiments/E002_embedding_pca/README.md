# E002: PCA of per-splat embeddings rendered as colour

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done (re-run on the final E001 checkpoint when it finishes)
- **Commit(s):** the commit adding `src/pino/pca_render.py`
- **Hypotheses:** H4, H6

## Question
Do the distilled per-splat embeddings organise a whole scene into meaningful parts, and do they agree with the
DINO teacher when both are viewed in one shared PCA basis?

## Setup
- Tool: `uv run python -m pino.pca_render` (`src/pino/pca_render.py`).
  1. The whole scene is embedded in xy tiles of ≤100k "core" splats plus a 1 m context margin (≤150k total),
     mirroring the training crops; only core splats take their tile's output, so there are no seams.
  2. A 3-component PCA is fit on 200k splats (opacity > 0.1) in standardised DINO space
     (`model.project(features)`); components are mapped to RGB with 1st/99th percentile normalisation.
  3. The scene is rendered with those colours (SH degree 0) from cameras sampled exactly like training, next to
     the RGB render and the DINO tokens of that render projected onto the **same** basis.
  4. Also writes `pca.ply` (PCA colours as SH DC, opens in any 3DGS viewer; not committed, ~18 MB each) and
     `pca_basis.pt`. `--viser` serves the coloured scene interactively.
- Checkpoint: E001-r1 `latest.pt` at **step 2000** (of 30k).
- Scenes: held-out validation scenes `0001_839920` (restaurant, 332k splats after opacity filter) and
  `0101_839973` (theatre, 528k splats).
- Reproduce: `bash scripts/run_pca.sh` (env `CKPT`, `GPU`).

## Runs
| Run | Scene | Logs |
|-----|-------|------|
| E002-r1 | 0001_839920 | `logs/0001_839920_dino/pca_grid.jpg` |
| E002-r2 | 0101_839973 | `logs/0101_839973_dino/pca_grid.jpg` |

Grid columns: RGB render | splat-embedding PCA render | DINO tokens of the RGB render in the same basis.

## Results
- Whole-scene embedding takes ~1 s per scene on an H100.
- Restaurant: chairs/tables red-orange, ceiling green, walls/pillars purple, floor white. Thin structures (chair
  legs, table edges) are sharp at splat resolution.
- Theatre: seats pale pink/white, carpet magenta, ceiling blue, walls yellow-green. Seats and carpet are both
  brown in RGB but separate cleanly in embedding space.
- In both scenes the DINO tokens projected onto the splat-fit basis show the same colour-to-category mapping, at
  16-px patch resolution.
- No visible tile seams.

## Interpretation
- H6 supported qualitatively: the embeddings share DINO's feature geometry (one basis colours both
  consistently) while being far finer-grained than DINO's patch grid.
- Early qualitative evidence for H4: the embeddings group splats by object category (seat vs carpet, chair vs
  floor) rather than by colour. Needs the quantitative linear probe.
- Only 2k training steps, two scenes, no quantitative metric.

## Next steps
- Re-run `scripts/run_pca.sh` on the final E001 checkpoint.
- Quantitative H4 probe using `labels.json` boxes.
