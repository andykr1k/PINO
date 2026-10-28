"""Lifting 2D teacher features onto splats ("DINO upper bound" for the 3D student).

For each view, a splat's lifted feature accumulates sum_p w_ip F_p, where w_ip is its alpha-blending weight at
pixel p and F_p the teacher token covering p. Rendering is linear in the splat colours, so this sum is exactly the
gradient of <render(colors), F> w.r.t. the colours: one backward pass per view gives it for every splat, with
occlusion handled by the blending weights. Dividing by sum_p w_ip (same trick with F = 1) gives the
blend-weighted average of the teacher tokens over every pixel and view the splat contributes to.
"""
import numpy as np
import torch

from pino.data.camera import intrinsics_from_hfov, sample_cameras
from pino.data.scene import GaussianScene
from pino.data.views import ViewGenerator
from pino.render import render_features
from pino.teacher import DINOv3Teacher

MAX_CHANNELS = 513  # gsplat's largest supported channel count per rasterization call


def sample_good_views(
    scene: GaussianScene,
    view_gen: ViewGenerator,
    num_views: int,
    oversample: int,
    hfov_range_deg: tuple[float, float],
    height_range: tuple[float, float],
    pitch_range_deg: tuple[float, float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Proposes cameras inside the rooms and keeps the best `num_views` by the training view-quality filter.

    Returns RGB images (V, 3, H, W), viewmats (V, 4, 4), Ks (V, 3, 3), depths (V, H, W) and alphas (V, H, W).
    """
    rng = np.random.default_rng(seed)
    height, width = view_gen.image_size
    num = num_views * oversample
    means = scene.means.cpu().numpy()
    viewmats = sample_cameras(scene.rooms, num, rng, np.percentile(means[:, :2], [5, 95], axis=0),
                              height_range, pitch_range_deg).to(scene.means.device)
    Ks = torch.from_numpy(np.stack([intrinsics_from_hfov(width, height, h)
                                    for h in rng.uniform(*hfov_range_deg, size=num)])).to(scene.means.device)
    images, depths, alphas = [], [], []
    for start in range(0, num, 16):  # bound render memory for large scenes
        rgb, depth, alpha = view_gen._render(scene, viewmats[start : start + 16], Ks[start : start + 16])
        images.append(rgb.permute(0, 3, 1, 2))
        depths.append(depth)
        alphas.append(alpha)
    depth, alpha = torch.cat(depths), torch.cat(alphas)
    keep = view_gen._select_views(depth, alpha)[:num_views]
    return torch.cat(images)[keep], viewmats[keep], Ks[keep], depth[keep], alpha[keep]


@torch.no_grad()
def embed_views(
    model: torch.nn.Module,
    scene: GaussianScene,
    view_gen: ViewGenerator,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    depths: torch.Tensor,
    alphas: torch.Tensor,
    views_per_batch: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student embeddings as in training (one frustum crop per view), averaged per splat over the views.

    Each view's projected embeddings are L2-normalised before averaging. Returns the mean (N, C) and the number of
    crops containing each splat (N,).
    """
    from pino.data.views import PackedSplats  # local import keeps pino.lift importable without a model

    n = scene.num_splats
    total, count = None, torch.zeros(n, device=scene.means.device)
    rgb = scene.rgb
    for start in range(0, viewmats.shape[0], views_per_batch):
        crops = [view_gen._crop_view(scene, viewmats[v], Ks[v], depths[v], alphas[v])[0]
                 for v in range(start, min(start + views_per_batch, viewmats.shape[0]))]
        ids = torch.cat(crops)
        offsets = [0] + list(np.cumsum([c.numel() for c in crops]))
        packed = PackedSplats(scene.means[ids], scene.quats[ids], scene.log_scales[ids],
                              scene.opacity_logits[ids], rgb[ids], offsets)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model(packed).features
        feats = torch.nn.functional.normalize(model.project(feats.float()), dim=-1)
        if total is None:
            total = torch.zeros(n, feats.shape[1], device=scene.means.device)
        total.index_add_(0, ids, feats)
        count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
    assert total is not None
    return total / count.clamp(min=1)[:, None], count


def lift_teacher_features(
    scene: GaussianScene,
    teacher: DINOv3Teacher,
    images: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    downsample: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend-weighted average of teacher patch tokens per splat over the given views.

    Returns features (N, C) (zero where unseen) and total blend weight (N,) (0 = never visible).
    The feature maps are rendered at 1/`downsample` of the image resolution, as in training.
    """
    height, width = images.shape[-2:]
    patch = teacher.patch_size
    assert patch % downsample == 0
    rep = patch // downsample
    scale = torch.tensor([1.0 / downsample, 1.0 / downsample, 1.0], device=Ks.device)[:, None]
    n = scene.num_splats
    weight_sum = torch.zeros(n, device=images.device)
    feature_sum = None

    for start in range(0, images.shape[0], 4):
        tokens = teacher(images[start : start + 4]).patch_tokens  # (b, h, w, C)
        c = tokens.shape[-1]
        if feature_sum is None:
            feature_sum = torch.zeros(n, c, device=images.device)
        for i in range(tokens.shape[0]):
            v = start + i
            # Token map at render resolution (each token covers rep x rep rendered pixels), plus a ones channel
            # whose gradient is the splat's total blend weight in this view.
            grid = tokens[i].repeat_interleave(rep, 0).repeat_interleave(rep, 1)
            grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
            for c0 in range(0, c + 1, MAX_CHANNELS):
                target = grid[..., c0 : c0 + MAX_CHANNELS]
                colors = torch.zeros(n, target.shape[-1], device=images.device, requires_grad=True)
                with torch.enable_grad():
                    rendered, _ = render_features(scene.means, scene.quats, scene.log_scales, scene.opacity_logits,
                                                  colors, viewmats[v], Ks[v] * scale,
                                                  width // downsample, height // downsample)
                    (rendered * target).sum().backward()
                grad = colors.grad
                end = min(c0 + MAX_CHANNELS, c)
                if end > c0:
                    feature_sum[:, c0:end] += grad[:, : end - c0]
                if c0 + target.shape[-1] == c + 1:  # this chunk holds the ones channel
                    weight_sum += grad[:, -1]
    assert feature_sum is not None
    features = feature_sum / weight_sum.clamp(min=1e-8)[:, None]
    return torch.where(weight_sum[:, None] > 0, features, torch.zeros_like(features)), weight_sum
