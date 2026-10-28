"""Losses aligning per-splat embeddings with DINO patch tokens of rendered views."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pino.data.views import ViewBatch
from pino.model import TeacherAlignedModel
from pino.render import render_features


@dataclass
class AlignmentOutput:
    loss: torch.Tensor
    metrics: dict[str, float]
    rendered: torch.Tensor | None = None  # (B, h, w, C) projected rendered embeddings, for visualisation
    rendered_valid: torch.Tensor | None = None  # (B, h, w) bool


def rendered_alignment_loss(
    model: TeacherAlignedModel,
    features: torch.Tensor,
    batch: ViewBatch,
    downsample: int = 4,
    min_alpha: float = 0.5,
    keep_renders: bool = False,
) -> AlignmentOutput:
    """Splat embeddings are alpha-blended into each view with gsplat, average-pooled onto the DINO patch
    grid, normalised by coverage, projected to teacher space and compared with cosine distance.

    Rendering at `1 / downsample` resolution keeps the cost low; every DINO patch still pools
    (patch_size / downsample)^2 rendered pixels.
    """
    height, width = batch.images.shape[-2:]
    patch = height // batch.patch_grid[0]
    assert patch % downsample == 0
    pool = patch // downsample
    scale = torch.tensor([1.0 / downsample, 1.0 / downsample, 1.0], device=batch.Ks.device)[:, None]
    raw_targets = batch.teacher.patch_tokens
    targets = model.standardize_targets(raw_targets)

    losses, cos_all, raw_cos_all, valid_all, renders = [], [], [], [], []
    for b in range(batch.splats.num_views):
        sl = batch.splats.view_slice(b)
        rendered, alpha = render_features(
            batch.splats.means[sl],
            batch.splats.quats[sl],
            batch.splats.log_scales[sl],
            batch.splats.opacity_logits[sl],
            features[sl],
            batch.viewmats[b],
            batch.Ks[b] * scale,
            width // downsample,
            height // downsample,
        )
        rendered = F.avg_pool2d(rendered.permute(2, 0, 1)[None], pool)[0].permute(1, 2, 0)
        alpha = F.avg_pool2d(alpha.permute(2, 0, 1)[None], pool)[0, 0]
        pred = model.project(rendered / alpha[..., None].clamp(min=1e-3))
        valid = alpha > min_alpha
        cos = F.cosine_similarity(pred, targets[b], dim=-1)
        losses.append((1.0 - cos)[valid].sum())
        cos_all.append(cos.detach())
        valid_all.append(valid)
        with torch.no_grad():
            raw_pred = model.to_teacher_space(pred)
            raw_cos_all.append(F.cosine_similarity(raw_pred, raw_targets[b], dim=-1))
        if keep_renders:
            renders.append(raw_pred)

    valid_t = torch.stack(valid_all)
    num_valid = valid_t.sum().clamp(min=1)
    loss = torch.stack(losses).sum() / num_valid
    cos_t, raw_cos_t = torch.stack(cos_all), torch.stack(raw_cos_all)
    any_valid = bool(valid_t.any())
    return AlignmentOutput(
        loss=loss,
        metrics={
            "render/cos": float(cos_t[valid_t].mean()) if any_valid else 0.0,
            "render/cos_raw": float(raw_cos_t[valid_t].mean()) if any_valid else 0.0,
            "render/valid_frac": float(valid_t.float().mean()),
        },
        rendered=torch.stack(renders) if keep_renders else None,
        rendered_valid=valid_t if keep_renders else None,
    )


def point_alignment_loss(
    model: TeacherAlignedModel,
    features: torch.Tensor,
    batch: ViewBatch,
    max_splats: int = 65536,
) -> AlignmentOutput:
    """Each visible splat's projected embedding should match the DINO token of the patch its centre lands in."""
    visible = torch.nonzero(batch.patch_index >= 0).squeeze(1)
    if visible.numel() > max_splats:
        visible = visible[torch.randperm(visible.numel(), device=visible.device)[:max_splats]]
    patch_index = batch.patch_index[visible]
    view_ids = batch.splats.view_ids[visible]
    _, h, w, c = batch.teacher.patch_tokens.shape
    targets = batch.teacher.patch_tokens.reshape(-1, h * w, c)[view_ids, patch_index]
    pred = model.project(features[visible].float())
    cos = F.cosine_similarity(pred, model.standardize_targets(targets), dim=-1)
    with torch.no_grad():
        raw_cos = F.cosine_similarity(model.to_teacher_space(pred), targets, dim=-1)
    return AlignmentOutput(
        loss=(1.0 - cos).mean(),
        metrics={"point/cos": float(cos.detach().mean()), "point/cos_raw": float(raw_cos.mean())},
    )


@torch.no_grad()
def constant_baseline_cos(batch: ViewBatch) -> float:
    """Cosine of each target token with the batch-mean token: what a model that ignores its input scores."""
    tokens = batch.teacher.patch_tokens.reshape(-1, batch.teacher.patch_tokens.shape[-1])
    return float(F.cosine_similarity(tokens, tokens.mean(0, keepdim=True), dim=-1).mean())
