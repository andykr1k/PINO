# E006: Object-identification probe

- **Date:** 2026-10-28
- **Owner:** Aidan Dempster (run by Claude Code)
- **Status:** done
- **Commit(s):** `6001868`
- **Hypotheses:** H4

## Question
Do the distilled per-splat embeddings linearly decode object categories in held-out scenes, beyond what raw splat
attributes or the untrained architecture give?

## Setup
- Tool: `uv run python -m pino.object_probe` (`src/pino/object_probe.py`).
- Labels: InteriorGS `labels.json` oriented 3D boxes (`logs/label_stats.txt`: 446,813 boxed instances, 714 labels;
  wall/floor/ceiling have no boxes, so they are not probed). Splats get the smallest containing box; splats with
  opacity < 0.1 are ignored. Classes: labels present in ≥ 300 training scenes, excluding "other" → **55 classes**
  (chair, bed, curtain, toilet, tv, cup, book, ...).
- Data: 150 random training scenes fit the probe; the 20 held-out validation scenes evaluate it. Up to 32 splats per
  instance; instances with < 16 labelled splats skipped. Splat level: 795k train / 104k val samples. Instance level
  (mean of unit-norm splat features over all the instance's splats): 25.8k train / 3.4k val objects.
- Probe: multinomial logistic regression with class-balanced cross-entropy (AdamW, 3k steps).
- Feature sets: **model** (E004 embeddings projected to DINO space, L2-normalised), **random** (same architecture,
  random weights), **attributes** (rgb, log-scale, opacity, quaternion, height; Fourier-expanded, L2-normalised).
- Reproduce: `CKPT=<ckpt> GPU=<gpu> bash scripts/run_probe.sh`; checkpoint sweep: `bash scripts/sweep_checkpoints.sh 002000 010000`.

## Runs
| Run | Checkpoint | Logs |
|-----|-----------|------|
| E006-r1 | E004 step 30000 | `logs/step30000/` |
| E006-r2 | E004 step 2000 | `logs/step2000/` |
| E006-r3 | E004 step 10000 | `logs/step10000/` |

## Results
Held-out validation scenes, final checkpoint (step 30000):

| Level | Features | Accuracy | Balanced accuracy |
|-------|----------|----------|-------------------|
| splat | **model** | **0.482** | **0.499** |
| splat | attributes | 0.159 | 0.214 |
| splat | random | 0.131 | 0.159 |
| instance | **model** | **0.535** | **0.573** |
| instance | attributes | 0.403 | 0.499 |
| instance | random | 0.203 | 0.254 |

Majority class: 0.193 (splat), 0.197 (instance). Chance balanced accuracy: 1/55 = 0.018.

Over training (balanced accuracy, model features):

| Step | splat | instance |
|------|-------|----------|
| 2000 | 0.360 | 0.507 |
| 10000 | 0.488 | 0.607 |
| 30000 | 0.499 | 0.573 |

Per class (splat level, model vs attributes recall): large or distinctive objects are near perfect: gas stoves
0.99 (0.82), range hood ("Smoke lampblack machine") 0.87 (0.44), tv 0.86 (0.15), curtain 0.84 (0.13), toilet
0.83 (0.44). Small tabletop items fail: spoon, bottle, jar and bowl ≈ 0.00; tray 0.03; flowerpot 0.04.

## Interpretation
- **H4 supported.** A single splat's embedding identifies its object category at 0.50 balanced accuracy over 55
  classes in unseen scenes, 2.3x raw attributes and 3.1x the untrained architecture. The gain comes from
  distillation, not the architecture alone.
- At the instance level, averaged Fourier attributes are a strong baseline (a kernel mean embedding of each object's
  height/colour/size distribution separates e.g. downlights from carpets); the model still adds +0.07 balanced
  accuracy at 30k (+0.11 at 10k).
- Failures concentrate on objects much smaller than a 16-px DINO patch in the 448-px training renders; their boxes
  also mostly contain the surface below them, so box-derived labels are noisy there.
- Splat-level semantics track the alignment curve (most gain by 10k). The instance-level 10k > 30k difference
  (0.607 vs 0.573) is within what ~3.4k objects over 55 classes can resolve; treat as noise until repeated.
- Caveat: box membership is an imperfect label (walls behind shelves, tables under vases), which understates every
  feature set's accuracy.

## Next steps
- DINO "upper bound": lift teacher patch tokens onto splats via multi-view correspondence and probe them, to see
  how much of DINO's semantics the 3D student keeps.
- Higher-resolution training renders (or more DINO patches per view) for small objects.
- Repeat with several probe seeds / more val scenes to put error bars on the instance-level numbers.
