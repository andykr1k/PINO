# E005: Embedding PCA renders on fixed data (E004 checkpoints)

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done (steps 2000 and 4000)
- **Commit(s):** `760a296`
- **Hypotheses:** H4, H6 (redoes E002 without the E003 data bug)

## Question
With complete scenes, do per-splat embeddings organise scenes into meaningful parts, consistently with DINO, across
different room types?

## Setup
- Tool: `python -m pino.pca_render` (see E002), `--tiles-per-batch 1` to share GPU 7 with training.
- Checkpoints: E004-r1 `ckpt_002000.pt` (and `ckpt_004000.pt` when available).
- Six held-out validation scenes: 0001_839920 (restaurant), 0101_839973 (theatre), 0251_840828 (living room /
  bedroom), 0451_840422 (modern apartment), 0651_841463, 0851_841772.
- Reproduce: `CKPT=<ckpt> GPU=<gpu> bash scripts/run_pca.sh` → `logs/step<N>/<scene>/pca_grid.jpg`.

Grid columns: RGB render | splat-embedding PCA render | DINO tokens of the RGB render in the same PCA basis.

## Results (step 2000)
- Whole-scene embedding: 1.4–3.6 s per scene (0.47–1.75M splats).
- Renders are gap-free now; PCA renders are sharp at splat resolution (chair legs, lamp cords, door frames).
- 0001 restaurant: ceiling red, walls/pillars cyan-blue, chairs/tables green, floor pale pink.
- 0251 living room: sofa/armchair/coffee table dark red, floor green, ceiling light green, walls pink-orange. The
  yellow armchair and white sofa share one colour, so the grouping is by object type, not RGB.
- 0451 apartment: ceiling green, walls blue, floor pink, bed and low cabinets red.
- DINO tokens projected onto the splat-fit basis use the same colour per category in every scene.
- Walls show a vertical colour gradient (also present in DINO), i.e. part of the embedding varies with height.

## Results (step 4000)
- Same structure, holding across all six scenes (`logs/step4000/`):
  - 0101 theatre: seats near-white, carpet red, walls yellow-green, ceiling beams blue; music stands/instruments
    group with the seats.
  - 0651 apartment: ceiling green, walls dark violet, floor yellow, sofa/bed magenta, door frames blue.
  - 0851 living room: ceiling green, curtains/walls blue, floor pink, seating/tables dark red; the pink armchair
    and grey sofa share a colour; chandelier arms are thin crisp lines (blobs in DINO's patch grid).
- In every scene the DINO tokens projected onto the splat basis use the same colour per category.

## Interpretation
- H6 holds on fixed data: splat embeddings share DINO's feature geometry at splat resolution.
- H4 (qualitative): grouping by category (seating, tables, floor, wall, ceiling) generalises across room types.
- Still only 2k/30k steps and no quantitative metric.

## Next steps
- Compare with step 4000 and the final checkpoint.
- Quantitative H4 probe with `labels.json`.
