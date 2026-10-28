"""Checks the DINOv3 teacher loads from the local HF cache and that gsplat's CUDA kernels build and render.

    CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=<gpu> \
        uv run python lab_notebook/experiments/E000_pipeline_bringup/scripts/smoke_dino_gsplat.py
"""
import time

import torch
from gsplat import rasterization
from transformers import AutoModel

model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", local_files_only=True).cuda().eval()
print(type(model).__name__, "registers", model.config.num_register_tokens, "dim", model.config.hidden_size)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(pixel_values=torch.randn(2, 3, 448, 448, device="cuda"))
print("tokens", tuple(out.last_hidden_state.shape))  # 1 cls + 4 registers + 28*28 patches

start = time.time()
n = 1000
means = torch.randn(n, 3, device="cuda") + torch.tensor([0, 0, 5.0], device="cuda")
K = torch.tensor([[[300, 0, 224], [0, 300, 224], [0, 0, 1.0]]], device="cuda")
img, alpha, meta = rasterization(
    means,
    torch.randn(n, 4, device="cuda"),
    torch.rand(n, 3, device="cuda") * 0.1,
    torch.rand(n, device="cuda"),
    torch.rand(n, 3, device="cuda"),
    torch.eye(4, device="cuda")[None],
    K,
    448,
    448,
)
print(f"gsplat render {tuple(img.shape)} in {time.time() - start:.1f}s (first call includes JIT build)")
