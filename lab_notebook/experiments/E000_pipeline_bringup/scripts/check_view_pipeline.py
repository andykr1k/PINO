"""Runs the GPU view pipeline on three training scenes and reports coverage/correspondence stats.

Saves logs/view_pipeline_grid.jpg: rows of [rendered view | DINOv3 patch-token PCA].

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> \
        uv run python lab_notebook/experiments/E000_pipeline_bringup/scripts/check_view_pipeline.py
"""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pino import INTERIOR_GS_PATH
from pino.data.dataset import InteriorGSScenes, split_scenes
from pino.data.views import ViewGenerator
from pino.teacher import DINOv3Teacher

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

dirs = split_scenes(INTERIOR_GS_PATH, "train", 50)
ds = InteriorGSScenes(
    dirs,
    num_candidates=24,
    image_size=(448, 448),
    hfov_range_deg=(70, 100),
    height_range=(1.2, 1.8),
    pitch_range_deg=(-25, 10),
    seed=0,
)
start = time.time()
samples = [ds[i] for i in (0, 100, 400)]
print(f"loaded 3 scenes in {time.time() - start:.1f}s", [s.scene.num_splats for s in samples])

device = torch.device("cuda")
teacher = DINOv3Teacher("facebook/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True).to(device)
gen = ViewGenerator(teacher, views_per_scene=4, image_size=(448, 448), max_splats_per_view=150_000)
for _ in range(2):  # second pass is the steady-state timing
    torch.cuda.synchronize()
    start = time.time()
    batch = gen(samples, device)
    torch.cuda.synchronize()
    print(f"view generation {time.time() - start:.2f}s")
assert batch is not None

h, w = batch.patch_grid
for b in range(batch.splats.num_views):
    pi = batch.patch_index[batch.splats.view_slice(b)]
    per_patch = torch.zeros(h * w, device=device).index_add_(0, pi[pi >= 0], torch.ones_like(pi[pi >= 0], dtype=torch.float))
    print(
        b,
        batch.scene_names[b],
        "splats",
        batch.splats.counts[b],
        "visible",
        int((pi >= 0).sum()),
        f"patches covered {float((per_patch > 0).float().mean()):.3f}",
        f"median visible splats/patch {float(per_patch.median()):.0f}",
    )

tokens = batch.teacher.patch_tokens
flat = tokens.reshape(-1, tokens.shape[-1])
flat = flat - flat.mean(0)
_, _, basis = torch.pca_lowrank(flat, q=3, center=False)
proj = flat @ basis[:, :3]
lo, hi = proj.quantile(0.01, 0), proj.quantile(0.99, 0)
proj = ((proj - lo) / (hi - lo)).clamp(0, 1).reshape(*tokens.shape[:3], 3)
rows = []
for b in range(batch.images.shape[0]):
    img = (batch.images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pca = np.kron((proj[b].cpu().numpy() * 255).astype(np.uint8), np.ones((16, 16, 1), dtype=np.uint8))
    rows.append(np.concatenate([img, pca], 1))
LOG_DIR.mkdir(exist_ok=True)
Image.fromarray(np.concatenate(rows, 0)).save(LOG_DIR / "view_pipeline_grid.jpg", quality=85)
print("saved", LOG_DIR / "view_pipeline_grid.jpg")
