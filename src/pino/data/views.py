"""GPU stage of the data pipeline: turns `SceneSample`s into training views.

For each scene the candidate cameras are rendered with gsplat, poor views (inside furniture,
staring into a wall, looking out of the scene) are rejected, the frozen DINO teacher embeds the
kept renders, and the splats inside each view frustum are cropped out together with their
splat -> DINO-patch correspondence.
"""
import math
from dataclasses import dataclass

import torch

from pino.data.dataset import SceneSample
from pino.data.scene import GaussianScene
from pino.render import render_rgbd
from pino.teacher import DINOv3Teacher, TeacherOutput


@dataclass
class PackedSplats:
    """Splat sets of several views concatenated along dim 0; view b owns rows offsets[b]:offsets[b+1]."""

    means: torch.Tensor  # (S, 3)
    quats: torch.Tensor  # (S, 4) wxyz, w >= 0
    log_scales: torch.Tensor  # (S, 3)
    opacity_logits: torch.Tensor  # (S,)
    rgb: torch.Tensor  # (S, 3) in [0, 1]
    offsets: list[int]  # B + 1 entries

    @property
    def num_views(self) -> int:
        return len(self.offsets) - 1

    @property
    def view_ids(self) -> torch.Tensor:
        counts = torch.tensor(self.counts, device=self.means.device)
        return torch.repeat_interleave(torch.arange(self.num_views, device=self.means.device), counts)

    @property
    def counts(self) -> list[int]:
        return [self.offsets[b + 1] - self.offsets[b] for b in range(self.num_views)]

    def view_slice(self, b: int) -> slice:
        return slice(self.offsets[b], self.offsets[b + 1])


@dataclass
class ViewBatch:
    images: torch.Tensor  # (B, 3, H, W) rendered RGB in [0, 1]
    teacher: TeacherOutput  # DINO tokens of `images`
    viewmats: torch.Tensor  # (B, 4, 4) world-to-camera
    Ks: torch.Tensor  # (B, 3, 3)
    splats: PackedSplats  # frustum-cropped splats of every view (world frame)
    patch_index: torch.Tensor  # (S,) flat DINO patch index the splat is visible in, -1 if occluded
    scene_names: list[str]

    @property
    def patch_grid(self) -> tuple[int, int]:
        return self.teacher.patch_tokens.shape[1], self.teacher.patch_tokens.shape[2]


class ViewGenerator:
    def __init__(
        self,
        teacher: DINOv3Teacher,
        views_per_scene: int,
        image_size: tuple[int, int],
        max_splats_per_view: int,
        max_depth: float = 12.0,
        frustum_margin: float = 0.1,
        min_coverage: float = 0.9,
        min_median_depth: float = 0.8,
        max_near_fraction: float = 0.05,
        near_depth: float = 0.4,
        visibility_rel_tol: float = 0.03,
        visibility_abs_tol: float = 0.05,
        sh_degree: int = 3,
        min_splats_per_view: int = 1024,
    ):
        self.teacher = teacher
        self.min_splats_per_view = min_splats_per_view
        self.views_per_scene = views_per_scene
        self.image_size = image_size
        self.max_splats_per_view = max_splats_per_view
        self.max_depth = max_depth
        self.frustum_margin = frustum_margin
        self.min_coverage = min_coverage
        self.min_median_depth = min_median_depth
        self.max_near_fraction = max_near_fraction
        self.near_depth = near_depth
        self.visibility_rel_tol = visibility_rel_tol
        self.visibility_abs_tol = visibility_abs_tol
        self.sh_degree = sh_degree

    @torch.no_grad()
    def __call__(self, samples: list[SceneSample], device: torch.device) -> ViewBatch | None:
        """Returns None if no candidate view of any scene contains enough splats."""
        images, viewmats, Ks, names = [], [], [], []
        crops: list[tuple[GaussianScene, torch.Tensor, torch.Tensor]] = []
        for sample in samples:
            scene = sample.scene.to(device, non_blocking=True)
            cand_viewmats = sample.viewmats.to(device, non_blocking=True)
            cand_Ks = sample.Ks.to(device, non_blocking=True)
            rgb, depth, alpha = self._render(scene, cand_viewmats, cand_Ks)
            keep = self._select_views(depth, alpha)
            for c in keep.tolist():
                splat_ids, patch_index = self._crop_view(scene, cand_viewmats[c], cand_Ks[c], depth[c], alpha[c])
                if splat_ids.numel() < self.min_splats_per_view:
                    continue
                crops.append((scene, splat_ids, patch_index))
                images.append(rgb[c].permute(2, 0, 1))
                viewmats.append(cand_viewmats[c])
                Ks.append(cand_Ks[c])
                names.append(scene.name)

        if not crops:
            return None
        images_t = torch.stack(images)
        teacher_out = self.teacher(images_t)
        splats = PackedSplats(
            means=torch.cat([s.means[ids] for s, ids, _ in crops]),
            quats=torch.cat([s.quats[ids] for s, ids, _ in crops]),
            log_scales=torch.cat([s.log_scales[ids] for s, ids, _ in crops]),
            opacity_logits=torch.cat([s.opacity_logits[ids] for s, ids, _ in crops]),
            rgb=torch.cat([s.rgb[ids] for s, ids, _ in crops]),
            offsets=[0] + list(torch.tensor([len(ids) for _, ids, _ in crops]).cumsum(0).tolist()),
        )
        return ViewBatch(
            images=images_t,
            teacher=teacher_out,
            viewmats=torch.stack(viewmats),
            Ks=torch.stack(Ks),
            splats=splats,
            patch_index=torch.cat([p for _, _, p in crops]),
            scene_names=names,
        )

    def _render(self, scene: GaussianScene, viewmats: torch.Tensor, Ks: torch.Tensor, cameras_per_call: int = 4):
        """Renders cameras a few at a time: gsplat counts tile intersections in 32 bits, and a batch of whole-scene
        cameras (some inside geometry, so near splats cover every tile) can overflow it at high resolution."""
        height, width = self.image_size
        outs = [
            render_rgbd(
                scene.means,
                scene.quats,
                scene.log_scales,
                scene.opacity_logits,
                scene.sh,
                viewmats[i : i + cameras_per_call],
                Ks[i : i + cameras_per_call],
                width,
                height,
                sh_degree=self.sh_degree,
            )
            for i in range(0, viewmats.shape[0], cameras_per_call)
        ]
        return tuple(torch.cat(parts) for parts in zip(*outs))

    def _select_views(self, depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Picks `views_per_scene` candidates, preferring ones that pass all quality checks."""
        solid = alpha > 0.5
        coverage = (alpha > 0.95).float().mean(dim=(1, 2))
        near_fraction = (solid & (depth < self.near_depth)).float().mean(dim=(1, 2))
        masked_depth = torch.where(solid, depth, torch.full_like(depth, float("nan")))
        median_depth = masked_depth.flatten(1).nanmedian(dim=1).values.nan_to_num(0.0)
        ok = (
            (coverage >= self.min_coverage)
            & (near_fraction <= self.max_near_fraction)
            & (median_depth >= self.min_median_depth)
        )
        # Candidates are already in random order: take passing ones first, then the best of the rest.
        score = ok.float() * 10.0 + coverage - 5.0 * near_fraction
        order = torch.argsort(score, descending=True, stable=True)
        return order[: self.views_per_scene]

    def _crop_view(
        self,
        scene: GaussianScene,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        depth: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the scene indices of the splats in the view frustum and their DINO patch index (-1 if hidden)."""
        height, width = self.image_size
        patch = self.teacher.patch_size
        cam = scene.means @ viewmat[:3, :3].T + viewmat[:3, 3]
        z = cam[:, 2]
        safe_z = z.clamp(min=1e-6)
        u = K[0, 0] * cam[:, 0] / safe_z + K[0, 2]
        v = K[1, 1] * cam[:, 1] / safe_z + K[1, 2]
        mu, mv = self.frustum_margin * width, self.frustum_margin * height
        in_frustum = (
            (z > 0.05) & (z < self.max_depth) & (u > -mu) & (u < width + mu) & (v > -mv) & (v < height + mv)
        )
        ids = torch.nonzero(in_frustum).squeeze(1)

        # Depth test of the splat centre against the rendered depth -> visible splats and their patch.
        u_i, v_i, z_i = u[ids].floor().long(), v[ids].floor().long(), z[ids]
        in_image = (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height)
        u_c, v_c = u_i.clamp(0, width - 1), v_i.clamp(0, height - 1)
        surface = depth[v_c, u_c]
        visible = (
            in_image
            & (alpha[v_c, u_c] > 0.5)
            & (z_i <= surface * (1.0 + self.visibility_rel_tol) + self.visibility_abs_tol)
        )
        patch_index = torch.where(visible, (v_c // patch) * (width // patch) + u_c // patch, -1)

        if ids.numel() > self.max_splats_per_view:
            # Keep every visible splat (they define the rendered surface), fill the rest randomly.
            priority = visible.float() + torch.rand(ids.numel(), device=ids.device)
            keep = torch.topk(priority, self.max_splats_per_view, sorted=False).indices
            ids, patch_index = ids[keep], patch_index[keep]
        return ids, patch_index


def random_z_rotation(splats: PackedSplats, generator: torch.Generator | None = None) -> PackedSplats:
    """Rotates each view's splats about the world z axis by a random angle (gravity-preserving augmentation)."""
    device = splats.means.device
    angles = torch.rand(splats.num_views, device=device, generator=generator) * 2.0 * math.pi
    theta = angles[splats.view_ids]
    c, s = theta.cos(), theta.sin()
    x, y, z = splats.means.unbind(-1)
    means = torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)
    a, d = (theta / 2).cos(), (theta / 2).sin()
    qw, qx, qy, qz = splats.quats.unbind(-1)
    quats = torch.stack([a * qw - d * qz, a * qx - d * qy, a * qy + d * qx, a * qz + d * qw], dim=-1)
    quats = quats * torch.where(quats[:, :1] < 0, -1.0, 1.0)
    return PackedSplats(means, quats, splats.log_scales, splats.opacity_logits, splats.rgb, splats.offsets)
