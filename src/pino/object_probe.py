"""Object-identification probe: how well do per-splat embeddings linearly decode InteriorGS object classes?

Splats are labelled by the oriented 3D boxes in each scene's labels.json (smallest containing box wins). A linear
classifier is trained on splats from training scenes and evaluated on held-out validation scenes, at two levels:
  - splat: one prediction per splat (up to --splats-per-instance sampled per object instance)
  - instance: one prediction per object, from the mean of its splats' (unit-norm) features
Feature sets compared:
  - model:      trained SplatTransformer embeddings, projected to DINO space and L2-normalised
  - random:     the same architecture with random weights (what the architecture alone gives)
  - attributes: raw splat attributes (rgb, log-scale, opacity, quaternion, height) with Fourier expansion

    CUDA_HOME=/usr/local/cuda-13.0 uv run python -m pino.object_probe --ckpt <ckpt> --out <dir>
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pino.config_schema import AppConfig
from pino.data.dataset import split_scenes
from pino.data.scene import GaussianScene, load_scene
from pino.model import FourierFeatures, SplatTransformer
from pino.pca_render import embed_scene

FEATURE_SETS = ("model", "random", "attributes")


def load_boxes(scene_dir: Path) -> list[tuple[str, np.ndarray]]:
    """(label, corners (8, 3)) for every boxed object; wall/floor/ceiling/room entries have no box."""
    labels = json.loads((scene_dir / "labels.json").read_text())
    return [
        (obj["label"], np.array([[p["x"], p["y"], p["z"]] for p in obj["bounding_box"]], dtype=np.float32))
        for obj in labels
        if "bounding_box" in obj
    ]


def select_classes(scene_dirs: list[Path], min_scenes: int, exclude: tuple[str, ...] = ("other",)) -> list[str]:
    scenes_with = collections.Counter()
    for scene_dir in scene_dirs:
        scenes_with.update({label for label, _ in load_boxes(scene_dir)})
    return sorted(label for label, n in scenes_with.items() if n >= min_scenes and label not in exclude)


@torch.no_grad()
def assign_instances(means: torch.Tensor, boxes: list[np.ndarray]) -> torch.Tensor:
    """Instance index per splat (-1 if in no box). Boxes are oriented: corner 0 plus edges to corners 1, 3 and 4.

    Boxes are applied from largest to smallest volume, so a splat inside nested boxes gets the smallest one.
    """
    owner = torch.full((means.shape[0],), -1, dtype=torch.long, device=means.device)
    corners = torch.from_numpy(np.stack(boxes)).to(means.device)  # (B, 8, 3)
    origin = corners[:, 0]
    edges = torch.stack([corners[:, 1] - origin, corners[:, 3] - origin, corners[:, 4] - origin], dim=1)  # (B, 3, 3)
    volume = torch.linalg.det(edges).abs()
    lo, hi = corners.min(1).values, corners.max(1).values
    for b in torch.argsort(volume, descending=True).tolist():
        if volume[b] < 1e-9:
            continue
        cand = torch.nonzero(((means >= lo[b]) & (means <= hi[b])).all(1)).squeeze(1)
        if cand.numel() == 0:
            continue
        local = (means[cand] - origin[b]) @ edges[b].T / (edges[b] ** 2).sum(-1)
        inside = ((local >= 0) & (local <= 1)).all(1)
        owner[cand[inside]] = b
    return owner


class AttributeFeatures:
    """Raw per-splat attributes with a Fourier expansion so a linear probe can carve non-linear boundaries."""

    def __init__(self, device: torch.device):
        self.fourier = FourierFeatures(12, num_freqs=6, min_wavelength=0.05, max_wavelength=8.0).to(device)

    def __call__(self, scene: GaussianScene, ids: torch.Tensor) -> torch.Tensor:
        attrs = torch.cat(
            [
                scene.rgb[ids] * 2 - 1,
                ((scene.log_scales[ids] + 4) / 2).clamp(-4, 4),
                scene.opacity_logits[ids].sigmoid()[:, None] * 2 - 1,
                scene.quats[ids],
                scene.means[ids, 2:3],  # height above the floor
            ],
            dim=-1,
        )
        return F.normalize(self.fourier(attrs), dim=-1)


def project_unit(model: SplatTransformer, features: torch.Tensor) -> torch.Tensor:
    return F.normalize(model.project(features), dim=-1)


@torch.no_grad()
def extract_scene(
    scene_dir: Path,
    class_index: dict[str, int],
    models: dict[str, SplatTransformer],
    attribute_features: AttributeFeatures,
    app: AppConfig,
    splats_per_instance: int,
    min_splats: int,
    generator: torch.Generator,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]] | None:
    boxes = [(label, corners) for label, corners in load_boxes(scene_dir) if label in class_index]
    if not boxes:
        return None
    scene = load_scene(scene_dir, min_opacity=app.data.min_opacity).to(device)
    owner = assign_instances(scene.means, [c for _, c in boxes])
    owner[scene.opacity_logits.sigmoid() < 0.1] = -1  # near-transparent splats carry little appearance

    embeddings = {name: embed_scene(m, scene) for name, m in models.items()}
    counts = torch.bincount(owner[owner >= 0], minlength=len(boxes))
    keep_instances = torch.nonzero(counts >= min_splats).squeeze(1)
    if keep_instances.numel() == 0:
        return None

    # Splat level: sample up to `splats_per_instance` splats of each kept instance.
    sampled, sampled_inst = [], []
    for inst in keep_instances.tolist():
        members = torch.nonzero(owner == inst).squeeze(1)
        pick = members[torch.randperm(members.numel(), device=device, generator=generator)[:splats_per_instance]]
        sampled.append(pick)
        sampled_inst.append(torch.full_like(pick, inst))
    ids, inst_of = torch.cat(sampled), torch.cat(sampled_inst)
    labels = torch.tensor([class_index[boxes[i][0]] for i in range(len(boxes))], device=device)

    # Instance level: mean of unit-norm features over all of the instance's splats.
    member_mask = torch.isin(owner, keep_instances)
    members, member_inst = torch.nonzero(member_mask).squeeze(1), owner[member_mask]
    remap = torch.full((len(boxes),), -1, dtype=torch.long, device=device)
    remap[keep_instances] = torch.arange(keep_instances.numel(), device=device)

    def instance_means(feats: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros(keep_instances.numel(), feats.shape[1], device=device).index_add_(0, remap[member_inst], feats)
        return F.normalize(sums, dim=-1)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for name in FEATURE_SETS:
        if name == "attributes":
            splat_feats, all_feats = attribute_features(scene, ids), attribute_features(scene, members)
        else:
            model = models[name]
            splat_feats = project_unit(model, embeddings[name][ids])
            all_feats = torch.cat([project_unit(model, embeddings[name][members[i : i + 262144]])
                                   for i in range(0, members.numel(), 262144)])
        out[name] = {
            "splat_x": splat_feats.half().cpu(),
            "splat_y": labels[inst_of].cpu(),
            "inst_x": instance_means(all_feats.float()).half().cpu(),
            "inst_y": labels[keep_instances].cpu(),
        }
    return out


def train_linear_probe(
    x: torch.Tensor, y: torch.Tensor, num_classes: int, device: torch.device,
    steps: int = 3000, batch_size: int = 8192, lr: float = 1e-2, weight_decay: float = 1e-4,
) -> torch.nn.Linear:
    """Multinomial logistic regression (class-balanced cross-entropy) trained with AdamW."""
    x, y = x.to(device).float(), y.to(device)
    probe = torch.nn.Linear(x.shape[1], num_classes).to(device)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp(min=1) * num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (min(batch_size, x.shape[0]),), device=device)
        loss = F.cross_entropy(probe(x[idx]), y[idx], weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return probe


@torch.no_grad()
def evaluate(probe: torch.nn.Linear, x: torch.Tensor, y: torch.Tensor, num_classes: int, device: torch.device) -> dict:
    pred = probe(x.to(device).float()).argmax(-1).cpu()
    per_class = [float((pred[y == c] == c).float().mean()) for c in range(num_classes) if (y == c).any()]
    return {"accuracy": float((pred == y).float().mean()), "balanced_accuracy": float(np.mean(per_class)),
            "per_class_recall": {c: float((pred[y == c] == c).float().mean()) for c in range(num_classes) if (y == c).any()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=150, help="Training scenes used to fit the probe")
    parser.add_argument("--val-scenes", type=int, default=20, help="Held-out validation scenes for evaluation")
    parser.add_argument("--min-class-scenes", type=int, default=300, help="Keep classes present in this many train scenes")
    parser.add_argument("--splats-per-instance", type=int, default=32)
    parser.add_argument("--min-splats", type=int, default=16, help="Skip instances with fewer labelled splats")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    app = AppConfig(**ckpt["config"])
    root = Path(app.paths.dataset.InteriorGS)
    train_dirs = split_scenes(root, "train", app.data.val_every)
    val_dirs = split_scenes(root, "val", app.data.val_every)[: args.val_scenes]
    rng = np.random.default_rng(args.seed)
    probe_train_dirs = [train_dirs[i] for i in sorted(rng.choice(len(train_dirs), args.train_scenes, replace=False))]

    classes = select_classes(train_dirs, args.min_class_scenes)
    class_index = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} classes (present in >= {args.min_class_scenes} train scenes): {classes}")

    teacher_dim = ckpt["model"]["head.weight"].shape[0]
    trained = SplatTransformer(teacher_dim=teacher_dim, **app.model.model_dump()).to(device).eval()
    trained.load_state_dict(ckpt["model"])
    random_model = SplatTransformer(teacher_dim=teacher_dim, **app.model.model_dump()).to(device).eval()
    models = {"model": trained, "random": random_model}
    attribute_features = AttributeFeatures(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    data: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = {
        split: {name: collections.defaultdict(list) for name in FEATURE_SETS} for split in ("train", "val")
    }
    start = time.time()
    for split, dirs in (("train", probe_train_dirs), ("val", val_dirs)):
        for i, scene_dir in enumerate(dirs):
            scene_out = extract_scene(scene_dir, class_index, models, attribute_features, app,
                                      args.splats_per_instance, args.min_splats, generator, device)
            if scene_out is None:
                continue
            for name, tensors in scene_out.items():
                for key, value in tensors.items():
                    data[split][name][key].append(value)
            if (i + 1) % 25 == 0:
                print(f"  {split}: {i + 1}/{len(dirs)} scenes ({time.time() - start:.0f}s)", flush=True)

    results = {"ckpt": str(args.ckpt), "step": ckpt["step"], "classes": classes,
               "train_scenes": len(probe_train_dirs), "val_scenes": len(val_dirs), "levels": {}}
    for level in ("splat", "inst"):
        for name in FEATURE_SETS:
            tr = {k: torch.cat(v) for k, v in data["train"][name].items()}
            va = {k: torch.cat(v) for k, v in data["val"][name].items()}
            probe = train_linear_probe(tr[f"{level}_x"], tr[f"{level}_y"], len(classes), device)
            metrics = evaluate(probe, va[f"{level}_x"], va[f"{level}_y"], len(classes), device)
            metrics.update(n_train=int(tr[f"{level}_y"].numel()), n_val=int(va[f"{level}_y"].numel()))
            results["levels"].setdefault(level, {})[name] = metrics
            print(f"{level:>6} {name:>10}: acc {metrics['accuracy']:.3f}  balanced acc {metrics['balanced_accuracy']:.3f}  "
                  f"(train {metrics['n_train']}, val {metrics['n_val']})", flush=True)

    majority = {}
    for level in ("splat", "inst"):
        y_tr = torch.cat(data["train"]["model"][f"{level}_y"])
        y_va = torch.cat(data["val"]["model"][f"{level}_y"])
        majority[level] = float((y_va == torch.bincount(y_tr).argmax()).float().mean())
    results["majority_class_accuracy"] = majority
    results["chance_balanced_accuracy"] = 1.0 / len(classes)
    print(f"majority-class accuracy: splat {majority['splat']:.3f}, instance {majority['inst']:.3f}; "
          f"chance balanced accuracy {1 / len(classes):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote", args.out / "results.json")


if __name__ == "__main__":
    main()
