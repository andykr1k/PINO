"""Thin wrappers around gsplat rasterization."""
import torch
from gsplat import rasterization


@torch.autocast("cuda", enabled=False)
def render_rgbd(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    opacity_logits: torch.Tensor,
    sh: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Renders C cameras of one scene.

    Returns rgb (C, H, W, 3) in [0, 1], expected depth (C, H, W) and alpha (C, H, W).
    """
    num_coeffs = (sh_degree + 1) ** 2
    out, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=log_scales.exp(),
        opacities=opacity_logits.sigmoid(),
        colors=sh[:, :num_coeffs].float(),
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
        packed=True,
    )
    return out[..., :3].clamp(0.0, 1.0), out[..., 3], alpha[..., 0]


@torch.autocast("cuda", enabled=False)
def render_features(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    opacity_logits: torch.Tensor,
    features: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha-blends per-splat feature vectors (N, D) into one view (differentiable w.r.t. features).

    Returns features (H, W, D) (not normalised by alpha) and alpha (H, W, 1).
    """
    out, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=log_scales.exp(),
        opacities=opacity_logits.sigmoid(),
        colors=features.float(),
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        render_mode="RGB",
        packed=True,
    )
    return out[0], alpha[0]
