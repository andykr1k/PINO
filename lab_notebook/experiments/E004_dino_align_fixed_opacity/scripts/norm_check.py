"""Tests whether E004's falling raw cosine is a prediction-norm artifact.

The loss is a cosine in standardised space, so prediction norms are unconstrained; the raw metric maps predictions
back with `pred * std + mean`, whose direction depends on |pred|. For each checkpoint this reports, on visible
splats of validation views: |pred| vs |standardised target|, raw cosine as logged, and raw cosine after rescaling
each prediction to the norm of its standardised target.

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> uv run python \
        lab_notebook/experiments/E004_dino_align_fixed_opacity/scripts/norm_check.py ckpt_a.pt ckpt_b.pt ...
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from pino.config_schema import AppConfig
from pino.data.dataset import InteriorGSScenes, split_scenes
from pino.data.views import ViewGenerator
from pino.model import SplatTransformer
from pino.teacher import DINOv3Teacher

device = torch.device("cuda")
first = torch.load(sys.argv[1], map_location=device, weights_only=False)
app = AppConfig(**first["config"])
teacher = DINOv3Teacher(app.teacher.checkpoint, local_files_only=True).to(device)
view_gen = ViewGenerator(teacher, views_per_scene=4, image_size=app.data.image_size,
                         max_splats_per_view=app.data.max_splats_per_view, max_depth=app.data.max_depth)
scenes = split_scenes(Path(app.paths.dataset.InteriorGS), "val", app.data.val_every)[:6]
dataset = InteriorGSScenes(scenes, num_candidates=app.data.num_candidates, image_size=app.data.image_size,
                           hfov_range_deg=app.data.hfov_range_deg, height_range=app.data.height_range,
                           pitch_range_deg=app.data.pitch_range_deg, min_opacity=app.data.min_opacity, seed=0)
batches = [view_gen([dataset[i], dataset[i + 1], dataset[i + 2]], device) for i in (0, 3)]

print(f"{'ckpt':>16} {'|pred|':>8} {'|target|':>9} {'raw cos':>8} {'raw cos (rescaled)':>19} {'std cos':>8}")
for path in sys.argv[1:]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SplatTransformer(teacher_dim=teacher.embed_dim, **app.model.model_dump()).to(device).eval()
    model.load_state_dict(ckpt["model"])
    stats = []
    torch.manual_seed(0)
    with torch.no_grad():
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(batch.splats).features.float()
            visible = torch.nonzero(batch.patch_index >= 0).squeeze(1)
            _, h, w, c = batch.teacher.patch_tokens.shape
            target = batch.teacher.patch_tokens.reshape(-1, h * w, c)[batch.splats.view_ids[visible], batch.patch_index[visible]]
            pred = model.project(features[visible])
            target_std = model.standardize_targets(target)
            rescaled = pred / pred.norm(dim=-1, keepdim=True) * target_std.norm(dim=-1, keepdim=True)
            stats.append(torch.stack([
                pred.norm(dim=-1).mean(),
                target_std.norm(dim=-1).mean(),
                F.cosine_similarity(model.to_teacher_space(pred), target, dim=-1).mean(),
                F.cosine_similarity(model.to_teacher_space(rescaled), target, dim=-1).mean(),
                F.cosine_similarity(pred, target_std, dim=-1).mean(),
            ]))
    s = torch.stack(stats).mean(0).tolist()
    print(f"{Path(path).name:>16} {s[0]:8.2f} {s[1]:9.2f} {s[2]:8.4f} {s[3]:19.4f} {s[4]:8.4f}")
