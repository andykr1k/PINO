"""Renders the same cameras with the old loader behaviour (+inf-opacity splats dropped) and the fixed loader.

Writes logs/<scene>_old_vs_fixed.jpg (rows: old | fixed) and prints splat counts and alpha coverage.

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> \
        uv run python lab_notebook/experiments/E003_render_fidelity/scripts/compare_renders.py [scene ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pino import INTERIOR_GS_PATH
from pino.data.dataset import InteriorGSScenes
from pino.data.scene import OPAQUE_LOGIT
from pino.render import render_rgbd

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
NUM_VIEWS = 4

device = torch.device("cuda")
LOG_DIR.mkdir(exist_ok=True)
for name in sys.argv[1:] or ["0001_839920", "0101_839973"]:
    dataset = InteriorGSScenes(
        [INTERIOR_GS_PATH / name],
        num_candidates=NUM_VIEWS,
        image_size=(448, 448),
        hfov_range_deg=(80, 80),
        height_range=(1.5, 1.5),
        pitch_range_deg=(-10, -10),
        seed=1,
    )
    sample = dataset[0]
    scene = sample.scene.to(device)
    viewmats, Ks = sample.viewmats.to(device), sample.Ks.to(device)
    old = scene.opacity_logits < OPAQUE_LOGIT  # the old loader dropped every +inf (fully opaque) splat
    renders = {}
    for label, mask in [("old", old), ("fixed", torch.ones_like(old))]:
        rgb, _, alpha = render_rgbd(
            scene.means[mask], scene.quats[mask], scene.log_scales[mask], scene.opacity_logits[mask],
            scene.sh[mask], viewmats, Ks, 448, 448,
        )
        renders[label] = rgb
        print(f"{name} {label:>5}: {int(mask.sum()):>8} splats, mean alpha {float(alpha.mean()):.3f}, "
              f"pixels with alpha<0.9: {float((alpha < 0.9).float().mean()):.3f}")
    rows = [
        np.concatenate([(renders[k][v].cpu().numpy() * 255).astype(np.uint8) for k in ("old", "fixed")], axis=1)
        for v in range(NUM_VIEWS)
    ]
    out = LOG_DIR / f"{name}_old_vs_fixed.jpg"
    Image.fromarray(np.concatenate(rows, axis=0)).save(out, quality=90)
    print("wrote", out)
