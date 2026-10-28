"""Re-runs validation on every E004 checkpoint with the current metric code (norm-corrected `to_teacher_space`).

Writes logs/metrics_revalidated.jsonl and logs/vis_revalidated/val_<step>.jpg.

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> uv run python \
        lab_notebook/experiments/E004_dino_align_fixed_opacity/scripts/revalidate.py [run_dir]
"""
import json
import subprocess
import sys
from pathlib import Path

import torch

from pino.config_schema import AppConfig
from pino.data.views import ViewGenerator
from pino.model import SplatTransformer
from pino.teacher import DINOv3Teacher
from pino.train import make_dataset, make_loader, validate

EXP_DIR = Path(__file__).resolve().parent.parent
main_checkout = Path(
    subprocess.check_output(["git", "-C", str(EXP_DIR), "worktree", "list", "--porcelain"], text=True)
    .splitlines()[0]
    .removeprefix("worktree ")
)
run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else main_checkout / "outputs/e004_dino_align_fixed_opacity"
ckpts = sorted(run_dir.glob("ckpt_*.pt"))

device = torch.device("cuda")
app = AppConfig(**torch.load(ckpts[0], map_location="cpu", weights_only=False)["config"])
teacher = DINOv3Teacher(app.teacher.checkpoint, local_files_only=True).to(device)
view_gen = ViewGenerator(teacher, views_per_scene=app.data.views_per_scene, image_size=app.data.image_size,
                         max_splats_per_view=app.data.max_splats_per_view, max_depth=app.data.max_depth)
val_loader = make_loader(make_dataset(app, "val", seed=0, limit=app.train.val_scenes), app, shuffle=False)
model = SplatTransformer(teacher_dim=teacher.embed_dim, **app.model.model_dump()).to(device)

out = EXP_DIR / "logs/metrics_revalidated.jsonl"
with open(out, "w") as f:
    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        metrics = validate(model, view_gen, val_loader, app, device,
                           EXP_DIR / f"logs/vis_revalidated/val_{ckpt['step']:06d}.jpg")
        f.write(json.dumps({"step": ckpt["step"], **metrics}) + "\n")
        f.flush()
        print(f"[{ckpt['step']:06d}] raw cos {metrics['val/render/cos_raw']:.4f}  std cos {metrics['val/render/cos']:.4f}  "
              f"baseline {metrics['val/baseline/cos']:.4f}", flush=True)
print("wrote", out)
