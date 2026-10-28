"""Sanity check for pino.lift: render lifted DINO features back into views and compare with the teacher's tokens.

Reports standardised-space patch cosine (same metric as `val/render/cos`) for
  - lifted, seen:     features lifted from all V views, scored on those same views
  - lifted, held-out: features lifted from V-1 views, scored on the left-out view (leave-one-out over 4 views)
  - student:          the E004 checkpoint's per-splat embeddings on the same views

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> uv run python \
        lab_notebook/experiments/E009_dino_lifted_upper_bound/scripts/lift_sanity.py <ckpt> [scene ...]
"""
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from pino.config_schema import AppConfig
from pino.data.scene import load_scene
from pino.data.views import PackedSplats, ViewGenerator
from pino.lift import lift_teacher_features, sample_good_views
from pino.model import SplatTransformer
from pino.pca_render import embed_scene
from pino.render import render_features
from pino.teacher import DINOv3Teacher

NUM_VIEWS = 16
device = torch.device("cuda")
ckpt = torch.load(sys.argv[1], map_location=device, weights_only=False)
app = AppConfig(**ckpt["config"])
teacher = DINOv3Teacher(app.teacher.checkpoint, local_files_only=True).to(device)
model = SplatTransformer(teacher_dim=teacher.embed_dim, **app.model.model_dump()).to(device).eval()
model.load_state_dict(ckpt["model"])
view_gen = ViewGenerator(teacher, views_per_scene=3 * NUM_VIEWS, image_size=tuple(app.data.image_size),
                         max_splats_per_view=app.data.max_splats_per_view)
ds = app.loss.render_downsample
root = Path(app.paths.dataset.InteriorGS)


@torch.no_grad()
def view_cosine(scene, splat_feats, viewmat, K, tokens, height, width) -> float:
    """Standardised-space cosine between rendered (alpha-normalised, patch-pooled) features and teacher tokens."""
    scale = torch.tensor([1.0 / ds, 1.0 / ds, 1.0], device=device)[:, None]
    rendered = []
    for c0 in range(0, splat_feats.shape[1], 512):
        r, alpha = render_features(scene.means, scene.quats, scene.log_scales, scene.opacity_logits,
                                   splat_feats[:, c0 : c0 + 512], viewmat, K * scale, width // ds, height // ds)
        rendered.append(r)
    rendered = torch.cat(rendered, -1)
    pool = teacher.patch_size // ds
    pooled = F.avg_pool2d(rendered.permute(2, 0, 1)[None], pool)[0].permute(1, 2, 0)
    alpha = F.avg_pool2d(alpha.permute(2, 0, 1)[None], pool)[0, 0]
    pred = pooled / alpha[..., None].clamp(min=1e-3)
    valid = alpha > 0.5
    return float(F.cosine_similarity(pred, model.standardize_targets(tokens), dim=-1)[valid].mean())


for name in sys.argv[2:] or ["0001_839920", "0451_840422"]:
    scene = load_scene(root / name, min_opacity=app.data.min_opacity).to(device)
    images, viewmats, Ks = sample_good_views(scene, view_gen, NUM_VIEWS, 3, app.data.hfov_range_deg,
                                             app.data.height_range, app.data.pitch_range_deg, seed=0)
    height, width = images.shape[-2:]
    tokens = torch.cat([teacher(images[i : i + 4]).patch_tokens for i in range(0, NUM_VIEWS, 4)])

    lifted, weight = lift_teacher_features(scene, teacher, images, viewmats, Ks, downsample=ds)
    lifted_std = model.standardize_targets(lifted)
    seen = [view_cosine(scene, lifted_std, viewmats[v], Ks[v], tokens[v], height, width) for v in range(4)]

    held_out = []
    for v in range(4):
        keep = [i for i in range(NUM_VIEWS) if i != v]
        loo, _ = lift_teacher_features(scene, teacher, images[keep], viewmats[keep], Ks[keep], downsample=ds)
        held_out.append(view_cosine(scene, model.standardize_targets(loo), viewmats[v], Ks[v], tokens[v], height, width))

    student = model.project(embed_scene(model, scene))
    student_cos = [view_cosine(scene, student, viewmats[v], Ks[v], tokens[v], height, width) for v in range(4)]

    # Student run as in training: one frustum crop per view (render-only scoring on the crop's splats).
    crop_cos = []
    for v in range(4):
        with torch.no_grad():
            _, depth, alpha = view_gen._render(scene, viewmats[v : v + 1], Ks[v : v + 1])
            ids, _ = view_gen._crop_view(scene, viewmats[v], Ks[v], depth[0], alpha[0])
            crop = PackedSplats(scene.means[ids], scene.quats[ids], scene.log_scales[ids],
                                scene.opacity_logits[ids], scene.rgb[ids], [0, ids.numel()])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.project(model(crop).features.float())
        crop_scene = replace(scene, means=scene.means[ids], quats=scene.quats[ids], log_scales=scene.log_scales[ids],
                             opacity_logits=scene.opacity_logits[ids], sh=scene.sh[ids])
        crop_cos.append(view_cosine(crop_scene, feats, viewmats[v], Ks[v], tokens[v], height, width))
    print(f"{name}: {scene.num_splats} splats, {float((weight > 0).float().mean()):.2f} seen by {NUM_VIEWS} views | "
          f"std cos — lifted seen {sum(seen) / 4:.3f}, lifted held-out {sum(held_out) / 4:.3f}, "
          f"student whole-scene tiles {sum(student_cos) / 4:.3f}, student frustum crops {sum(crop_cos) / 4:.3f}",
          flush=True)
