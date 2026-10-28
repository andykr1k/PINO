"""Trains the splat transformer to align per-splat embeddings with DINOv3 features of rendered views.

    uv run python -m pino.train                       # defaults from conf/config.yaml
    uv run python -m pino.train hparams.max_steps=1000 data.scenes_per_batch=2
"""
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from pino.config_schema import AppConfig
from pino.data.dataset import InteriorGSScenes, SceneSample, collate_scene_samples, split_scenes
from pino.data.views import ViewBatch, ViewGenerator, random_z_rotation
from pino.losses import AlignmentOutput, constant_baseline_cos, point_alignment_loss, rendered_alignment_loss
from pino.model import SplatTransformer
from pino.teacher import DINOv3Teacher


def to_app_config(cfg: DictConfig) -> AppConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Expected dict from OmegaConf, got {type(cfg_dict)}")
    cfg_dict_str_keys: dict[str, Any] = {str(k): v for k, v in cfg_dict.items() if k != "hydra"}
    return AppConfig(**cfg_dict_str_keys)


def make_dataset(app: AppConfig, split: str, seed: int | None, limit: int | None = None) -> InteriorGSScenes:
    scene_dirs = split_scenes(Path(app.paths.dataset.InteriorGS), split, app.data.val_every)
    return InteriorGSScenes(
        scene_dirs[:limit],
        num_candidates=app.data.num_candidates,
        image_size=app.data.image_size,
        hfov_range_deg=app.data.hfov_range_deg,
        height_range=app.data.height_range,
        pitch_range_deg=app.data.pitch_range_deg,
        min_opacity=app.data.min_opacity,
        seed=seed,
    )


def make_loader(dataset: InteriorGSScenes, app: AppConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=app.data.scenes_per_batch,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=app.data.num_workers,
        collate_fn=collate_scene_samples,
        pin_memory=True,
        persistent_workers=shuffle,
        prefetch_factor=2,
    )


def infinite(loader: DataLoader) -> Iterator[list[SceneSample]]:
    while True:
        yield from loader


def lr_at(step: int, app: AppConfig) -> float:
    hp = app.hparams
    if step < hp.warmup_steps:
        return hp.learning_rate * (step + 1) / hp.warmup_steps
    progress = min(1.0, (step - hp.warmup_steps) / max(1, hp.max_steps - hp.warmup_steps))
    return hp.learning_rate * (hp.min_lr_ratio + (1 - hp.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def compute_losses(
    model: SplatTransformer, batch: ViewBatch, app: AppConfig, augment: bool, keep_renders: bool = False
) -> tuple[torch.Tensor, AlignmentOutput, AlignmentOutput | None]:
    splats = random_z_rotation(batch.splats) if augment and app.loss.rotate_inputs else batch.splats
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoding = model(splats)
    render_out = rendered_alignment_loss(
        model,
        encoding.features,
        batch,
        downsample=app.loss.render_downsample,
        min_alpha=app.loss.min_alpha,
        keep_renders=keep_renders,
    )
    loss = app.loss.render_weight * render_out.loss
    point_out = None
    if app.loss.point_weight > 0:
        point_out = point_alignment_loss(model, encoding.features, batch, max_splats=app.loss.point_max_splats)
        loss = loss + app.loss.point_weight * point_out.loss
    return loss, render_out, point_out


@torch.no_grad()
def save_alignment_grid(batch: ViewBatch, render_out: AlignmentOutput, path: Path, max_views: int = 6) -> None:
    """Rows of [rendered RGB | DINO tokens | predicted tokens], features shown in a shared 3-component PCA."""
    assert render_out.rendered is not None and render_out.rendered_valid is not None
    targets = batch.teacher.patch_tokens[:max_views]
    preds = render_out.rendered[:max_views]
    valid = render_out.rendered_valid[:max_views]
    flat = targets.reshape(-1, targets.shape[-1])
    mean = flat.mean(0)
    _, _, basis = torch.pca_lowrank(flat - mean, q=3, center=False)

    def to_rgb(x: torch.Tensor) -> torch.Tensor:
        return (x - mean) @ basis[:, :3]

    proj_t = to_rgb(targets)
    lo, hi = proj_t.flatten(0, 2).quantile(0.01, 0), proj_t.flatten(0, 2).quantile(0.99, 0)
    patch = batch.images.shape[-1] // targets.shape[2]

    def upsample(x: torch.Tensor) -> np.ndarray:
        img = ((x - lo) / (hi - lo)).clamp(0, 1).cpu().numpy()
        return np.kron((img * 255).astype(np.uint8), np.ones((patch, patch, 1), dtype=np.uint8))

    rows = []
    for b in range(targets.shape[0]):
        rgb = (batch.images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pred = upsample(to_rgb(preds[b]))
        pred[~np.kron(valid[b].cpu().numpy(), np.ones((patch, patch), dtype=bool))] = 0
        rows.append(np.concatenate([rgb, upsample(proj_t[b]), pred], axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(path, quality=90)


@torch.no_grad()
def validate(
    model: SplatTransformer, view_gen: ViewGenerator, loader: DataLoader, app: AppConfig, device: torch.device, vis_path: Path
) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(0)
        saved_vis = False
        for samples in loader:
            batch = view_gen(samples, device)
            if batch is None:
                continue
            loss, render_out, point_out = compute_losses(model, batch, app, augment=False, keep_renders=not saved_vis)
            metrics = {"loss": loss.item(), "baseline/cos": constant_baseline_cos(batch), **render_out.metrics}
            if point_out is not None:
                metrics.update(point_out.metrics)
            for k, v in metrics.items():
                totals.setdefault(k, []).append(v)
            if not saved_vis:
                save_alignment_grid(batch, render_out, vis_path)
                saved_vis = True
    model.train()
    return {f"val/{k}": float(np.mean(v)) for k, v in totals.items()}


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = to_app_config(cfg)
    out_dir = Path(app.paths.output_dir or HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    (out_dir / "config.json").write_text(app.model_dump_json(indent=2))

    torch.manual_seed(app.train.seed)
    device = torch.device(app.train.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    teacher = DINOv3Teacher(
        app.teacher.checkpoint, cache_dir=app.teacher.cache_dir, local_files_only=app.teacher.local_files_only
    ).to(device)
    view_gen = ViewGenerator(
        teacher,
        views_per_scene=app.data.views_per_scene,
        image_size=app.data.image_size,
        max_splats_per_view=app.data.max_splats_per_view,
        max_depth=app.data.max_depth,
    )
    model = SplatTransformer(teacher_dim=teacher.embed_dim, **app.model.model_dump()).to(device)
    print(f"SplatTransformer: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": app.hparams.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=app.hparams.learning_rate,
        betas=(0.9, 0.95),
    )

    start_step = 0
    if app.train.resume:
        ckpt = torch.load(app.train.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from {app.train.resume} at step {start_step}")

    train_loader = make_loader(make_dataset(app, "train", seed=None), app, shuffle=True)
    val_loader = make_loader(make_dataset(app, "val", seed=0, limit=app.train.val_scenes), app, shuffle=False)
    train_iter = infinite(train_loader)

    def save_checkpoint(step: int) -> None:
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": app.model_dump()}
        torch.save(state, out_dir / "latest.pt")
        if step % app.train.ckpt_every == 0:
            torch.save(state, out_dir / f"ckpt_{step:06d}.pt")

    metrics_file = open(out_dir / "metrics.jsonl", "a")

    def log(step: int, metrics: dict[str, float]) -> None:
        metrics_file.write(json.dumps({"step": step, **metrics}) + "\n")
        metrics_file.flush()
        print(f"[{step:06d}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)

    model.train()
    window: dict[str, list[float]] = {}
    t_last = time.time()
    for step in range(start_step, app.hparams.max_steps):
        t0 = time.time()
        batch = None
        while batch is None:  # only None if every view of every scene in the batch was empty
            batch = view_gen(next(train_iter), device)
        t_data = time.time() - t0

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, app)
        model.update_target_stats(batch.teacher.patch_tokens)
        loss, render_out, point_out = compute_losses(model, batch, app, augment=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), app.hparams.grad_clip)
        optimizer.step()

        metrics = {"loss": loss.item(), "grad_norm": grad_norm.item(), "data_s": t_data, **render_out.metrics}
        if point_out is not None:
            metrics.update(point_out.metrics)
        for k, v in metrics.items():
            window.setdefault(k, []).append(v)

        done = step + 1
        if done % app.train.log_every == 0:
            summary = {k: float(np.mean(v)) for k, v in window.items()}
            summary["lr"] = lr_at(step, app)
            summary["step_s"] = (time.time() - t_last) / app.train.log_every
            summary["views"] = float(batch.images.shape[0])
            summary["splats_per_view"] = float(np.mean(batch.splats.counts))
            log(done, summary)
            window.clear()
            t_last = time.time()
        if done % app.train.val_every == 0 or done == app.hparams.max_steps:
            log(done, validate(model, view_gen, val_loader, app, device, out_dir / "vis" / f"val_{done:06d}.jpg"))
            t_last = time.time()
        if done % app.train.ckpt_every == 0 or done == app.hparams.max_steps:
            save_checkpoint(done)

    metrics_file.close()


if __name__ == "__main__":
    main()
