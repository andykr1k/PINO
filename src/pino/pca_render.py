"""Embeds every splat of a scene with a trained splat encoder, maps the top-3 PCA components of the
embeddings to RGB, and renders the scene with those colours.

Outputs (in --out):
  pca_grid.jpg  rows of [RGB render | splat-embedding PCA render | DINO tokens of the RGB render in the same
                PCA basis], so the 3D embeddings and the 2D teacher can be compared directly
  pca.ply       the scene with PCA colours as SH DC (degree 0), viewable in any 3DGS viewer
  pca_basis.pt  PCA mean / basis / colour normalisation, to colour other scenes consistently

    CUDA_HOME=/usr/local/cuda-13.0 uv run python -m pino.pca_render \
        --ckpt outputs/run/latest.pt --scene 0050_839901 --out pca_out [--viser]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pino.config_schema import AppConfig
from pino.data.dataset import InteriorGSScenes, split_scenes
from pino.data.scene import SH_C0, GaussianScene, load_scene
from pino.data.views import PackedSplats, ViewGenerator
from pino.model import TeacherAlignedModel, build_model
from pino.render import render_rgbd
from pino.teacher import DINOv3Teacher


def tile_scene(
    means: torch.Tensor, max_core: int, margin: float, max_context: int, generator: torch.Generator
) -> list[tuple[torch.Tensor, int]]:
    """Splits the scene into xy tiles (recursive median split) of at most `max_core` splats.

    Returns (ids, num_core) per tile: ids lists the tile's core splats first, followed by context splats
    within `margin` metres of the tile, so the model sees neighbourhoods similar to its training crops.
    """
    xy = means[:, :2]
    cores: list[torch.Tensor] = []
    stack = [torch.arange(means.shape[0], device=means.device)]
    while stack:
        ids = stack.pop()
        if ids.numel() <= max_core:
            cores.append(ids)
            continue
        pts = xy[ids]
        axis = int((pts.max(0).values - pts.min(0).values).argmax())
        order = pts[:, axis].argsort()
        half = ids.numel() // 2
        stack += [ids[order[:half]], ids[order[half:]]]

    tiles = []
    for core in cores:
        lo, hi = xy[core].min(0).values - margin, xy[core].max(0).values + margin
        in_box = ((xy >= lo) & (xy <= hi)).all(1)
        in_box[core] = False
        context = torch.nonzero(in_box).squeeze(1)
        budget = max_context - core.numel()
        if context.numel() > budget:
            context = context[torch.randperm(context.numel(), device=context.device, generator=generator)[:budget]]
        tile = torch.cat([core, context])
        tiles.append((tile, core.numel()))
    return tiles


@torch.no_grad()
def embed_scene(
    model: TeacherAlignedModel,
    scene: GaussianScene,
    max_core: int = 100_000,
    margin: float = 1.0,
    max_context: int = 150_000,
    tiles_per_batch: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Per-splat embeddings (N, out_dim) for every splat of the scene."""
    generator = torch.Generator(device=scene.means.device).manual_seed(seed)
    torch.manual_seed(seed)
    tiles = tile_scene(scene.means, max_core, margin, max_context, generator)
    rgb = scene.rgb
    features = torch.zeros(scene.num_splats, model.out_dim, device=scene.means.device)
    for start in range(0, len(tiles), tiles_per_batch):
        chunk = tiles[start : start + tiles_per_batch]
        ids = torch.cat([t for t, _ in chunk])
        counts = [t.numel() for t, _ in chunk]
        splats = PackedSplats(
            means=scene.means[ids],
            quats=scene.quats[ids],
            log_scales=scene.log_scales[ids],
            opacity_logits=scene.opacity_logits[ids],
            rgb=rgb[ids],
            offsets=[0] + list(np.cumsum(counts)),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(splats).features.float()
        for v, (tile, num_core) in enumerate(chunk):
            features[tile[:num_core]] = out[splats.offsets[v] : splats.offsets[v] + num_core]
    return features


class PCAColoring:
    """3-component PCA of embeddings with percentile-normalised RGB output."""

    def __init__(self, samples: torch.Tensor):
        self.mean = samples.mean(0)
        _, _, basis = torch.pca_lowrank(samples - self.mean, q=3, center=False)
        self.basis = basis[:, :3]
        proj = (samples - self.mean) @ self.basis
        self.lo, self.hi = proj.quantile(0.01, 0), proj.quantile(0.99, 0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (((x - self.mean) @ self.basis - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "basis": self.basis, "lo": self.lo, "hi": self.hi}


def to_space(model: TeacherAlignedModel, features: torch.Tensor, space: str, chunk: int = 262144) -> torch.Tensor:
    """Unit-normalised embeddings: the loss is a cosine, so only direction is meaningful (norms drift in training)."""
    if space == "embedding":
        return F.normalize(features, dim=-1)
    return torch.cat([F.normalize(model.project(features[i : i + chunk]), dim=-1) for i in range(0, features.shape[0], chunk)])


def colors_to_sh(colors: torch.Tensor) -> torch.Tensor:
    return ((colors - 0.5) / SH_C0)[:, None, :]


def write_ply(path: Path, scene: GaussianScene, colors: torch.Tensor) -> None:
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = torch.cat(
        [scene.means, colors_to_sh(colors)[:, 0], scene.opacity_logits[:, None], scene.log_scales, scene.quats], dim=1
    )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {scene.num_splats}"]
    header += [f"property float {n}" for n in names] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(data.float().cpu().numpy().astype("<f4").tobytes())


def quats_scales_to_covariances(quats: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quats.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)
    M = R * log_scales.exp()[:, None, :]
    return M @ M.transpose(1, 2)


def upsample_patches(rgb: torch.Tensor, patch: int) -> np.ndarray:
    img = (rgb.cpu().numpy() * 255).astype(np.uint8)
    return np.kron(img, np.ones((patch, patch, 1), dtype=np.uint8))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--scene", type=str, default=None, help="Scene folder name; default: first validation scene")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--space", choices=["dino", "embedding"], default="dino",
                        help="PCA over projected (standardised DINO-space) embeddings or the raw per-splat embeddings")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--fit-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiles-per-batch", type=int, default=4, help="Lower to reduce GPU memory")
    parser.add_argument("--viser", action="store_true", help="Serve the PCA-coloured scene in viser after rendering")
    args = parser.parse_args()

    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    app = AppConfig(**ckpt["config"])
    teacher = DINOv3Teacher(app.teacher.checkpoint, cache_dir=app.teacher.cache_dir,
                            local_files_only=app.teacher.local_files_only).to(device)
    model = build_model(app.model, teacher.embed_dim).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.ckpt} (step {ckpt['step']})")

    root = Path(app.paths.dataset.InteriorGS)
    scene_dirs = split_scenes(root, "val", app.data.val_every) if args.scene is None else [root / args.scene]
    scene = load_scene(scene_dirs[0], min_opacity=app.data.min_opacity).to(device)
    print(f"Scene {scene.name}: {scene.num_splats} splats")

    start = time.time()
    features = embed_scene(model, scene, tiles_per_batch=args.tiles_per_batch, seed=args.seed)
    print(f"Embedded scene in {time.time() - start:.1f}s")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    fit_pool = torch.nonzero(scene.opacity_logits.sigmoid() > 0.1).squeeze(1)
    fit_ids = fit_pool[torch.randperm(fit_pool.numel(), device=device, generator=generator)[: args.fit_samples]]
    coloring = PCAColoring(to_space(model, features[fit_ids], args.space))
    colors = torch.cat([coloring(to_space(model, features[i : i + 262144], args.space))
                        for i in range(0, scene.num_splats, 262144)])

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({**coloring.state_dict(), "space": args.space, "scene": scene.name, "ckpt": str(args.ckpt)},
               args.out / "pca_basis.pt")
    write_ply(args.out / "pca.ply", scene, colors)

    # Views: same camera sampling and quality filter as training, seeded for reproducibility.
    dataset = InteriorGSScenes([scene_dirs[0]], num_candidates=4 * args.views, image_size=app.data.image_size,
                               hfov_range_deg=app.data.hfov_range_deg, height_range=app.data.height_range,
                               pitch_range_deg=app.data.pitch_range_deg, min_opacity=app.data.min_opacity,
                               seed=args.seed)
    view_gen = ViewGenerator(teacher, views_per_scene=args.views, image_size=app.data.image_size,
                             max_splats_per_view=app.data.max_splats_per_view, max_depth=app.data.max_depth)
    batch = view_gen([dataset[0]], device)
    assert batch is not None, "no usable views of this scene"
    height, width = app.data.image_size
    pca_render, _, _ = render_rgbd(scene.means, scene.quats, scene.log_scales, scene.opacity_logits,
                                   colors_to_sh(colors), batch.viewmats, batch.Ks, width, height, sh_degree=0)

    dino_colors = None
    if args.space == "dino":
        dino_colors = coloring(F.normalize(model.standardize_targets(batch.teacher.patch_tokens), dim=-1))
    patch = height // batch.patch_grid[0]
    rows = []
    for b in range(batch.images.shape[0]):
        row = [(batch.images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
               (pca_render[b].cpu().numpy() * 255).astype(np.uint8)]
        if dino_colors is not None:
            row.append(upsample_patches(dino_colors[b], patch))
        rows.append(np.concatenate(row, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out / "pca_grid.jpg", quality=90)
    print(f"Wrote {args.out / 'pca_grid.jpg'}, {args.out / 'pca.ply'}, {args.out / 'pca_basis.pt'}")

    if args.viser:
        import viser

        server = viser.ViserServer()
        server.scene.add_gaussian_splats(
            "/pca",
            centers=scene.means.cpu().numpy(),
            covariances=quats_scales_to_covariances(scene.quats, scene.log_scales).cpu().numpy(),
            rgbs=colors.cpu().numpy(),
            opacities=scene.opacity_logits.sigmoid()[:, None].cpu().numpy(),
        )
        print("Serving PCA-coloured scene in viser. Ctrl+C to exit.")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    main()
