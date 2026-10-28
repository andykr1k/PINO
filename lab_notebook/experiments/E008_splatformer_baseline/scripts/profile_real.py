"""Memory/time of both encoders on real training batches (full losses, as in pino.train).

    CUDA_VISIBLE_DEVICES=4 CUDA_HOME=/usr/local/cuda-13.0 uv run python \
        lab_notebook/experiments/E008_splatformer_baseline/scripts/profile_real.py hydra.run.dir=/tmp/e008_profile
"""
import time

import hydra
import torch
from omegaconf import DictConfig

from pino.model import build_model
from pino.teacher import DINOv3Teacher
from pino.data.views import ViewGenerator
from pino.train import compute_losses, infinite, make_dataset, make_loader, to_app_config

GiB = 2**30


@hydra.main(version_base="1.3", config_path="../../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    app = to_app_config(cfg)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = DINOv3Teacher(app.teacher.checkpoint, cache_dir=app.teacher.cache_dir,
                            local_files_only=app.teacher.local_files_only).to(device)
    view_gen = ViewGenerator(teacher, views_per_scene=app.data.views_per_scene, image_size=app.data.image_size,
                             max_splats_per_view=app.data.max_splats_per_view, max_depth=app.data.max_depth)
    app.data.num_workers = 8
    it = infinite(make_loader(make_dataset(app, "train", seed=0, limit=64), app, shuffle=True))
    batches = []
    while len(batches) < 4:
        b = view_gen(next(it), device)
        if b is not None:
            batches.append(b)
    print("splats per batch:", [b.splats.offsets[-1] for b in batches])

    for arch in ("splatformer", "splat_transformer"):
        app.model.arch = arch
        model = build_model(app.model, teacher.embed_dim).to(device).train()
        if arch == "splatformer":
            counts = []
            for name, mod in model.backbone.enc.named_children():
                mod.register_forward_hook(lambda mod, i, o, name=name: counts.append((name, o.feat.shape[0])))
        for i, batch in enumerate(batches):
            if arch == "splatformer":
                counts.clear()
            model.update_target_stats(batch.teacher.patch_tokens)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated() / GiB
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss, _, _ = compute_losses(model, batch, app, augment=True)
            loss.backward()
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            print(f"{arch} batch {i}: {batch.splats.offsets[-1] / 1e6:.2f}M splats, {time.time() - t0:.2f}s, "
                  f"peak {torch.cuda.max_memory_allocated() / GiB:.1f} GiB (base {base:.1f}), loss {loss.item():.3f}")
            if arch == "splatformer":
                print("   points per stage:", " ".join(f"{k}:{v / 1e3:.0f}k" for k, v in counts))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
