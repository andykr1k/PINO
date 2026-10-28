# E003: Render fidelity — fully opaque splats were being dropped

- **Date:** 2026-10-27
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done
- **Commit(s):** fix in `fc57aaf` (`src/pino/data/scene.py`)
- **Hypotheses:** — (data bug; invalidates E001 and E002)

## Question
InteriorGS splats are high quality, but our renders showed see-through gaps, smeared walls and dark holes. Is
something wrong with loading or rendering?

## Setup
- `scripts/attribute_stats.py`: raw PLY attribute percentiles and non-finite counts per column.
- `scripts/compare_renders.py`: renders the same seeded cameras with the old behaviour (+inf-opacity splats
  dropped) and the fixed loader; writes `logs/<scene>_old_vs_fixed.jpg` (columns: old | fixed).

## Results
- The `opacity` column holds **+inf for 56% (0001_839920) and 58% (0101_839973) of splats**; every finite
  value lies within ±5.54 = logit(255/256). The PLYs were decompressed from 8-bit opacities, so opacity 1.0
  became logit(1) = +inf. No other column has non-finite values.
- `load_ply_gaussians` dropped all non-finite rows, i.e. **every fully opaque splat**.
- Fix: map +inf to logit 10 (opacity 0.99995).

| Scene | Splats old → fixed | Mean alpha old → fixed | Pixels with alpha < 0.9 old → fixed |
|-------|--------------------|------------------------|-------------------------------------|
| 0001_839920 | 331,636 → 846,540 | 0.859 → 0.998 | 34.5% → 0.5% |
| 0101_839973 | 528,221 → 1,746,052 | 0.860 → 0.999 | 36.5% → 0.0% |

Fixed renders are clean and photoreal (walls, windows, ceilings); see `logs/*_old_vs_fixed.jpg`.

## Interpretation
- All earlier renders, DINO targets and model inputs (E000–E002) came from scenes missing over half their splats,
  mostly the solid surfaces. DINO features of those renders describe a degraded scene, so E001's numbers are not
  representative; it is superseded by E004.
- Other conventions check out: unit quaternions (wxyz), log scales, standard SH DC/rest layout.

## Next steps
- E004: re-run the baseline on the fixed data.
- Redo E002-style PCA renders with an E004 checkpoint.
