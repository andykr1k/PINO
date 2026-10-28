"""Checks for the SplatFormer baseline: SDPA varlen attention == per-sequence attention, segment_reduce ==
segment_csr, checkpointed == plain outputs/gradients, memory/time on random inputs (uniform points are a worst
case: they barely merge when pooled), and that E004 checkpoints still load.

    CUDA_VISIBLE_DEVICES=6 uv run python lab_notebook/experiments/E008_splatformer_baseline/scripts/check_splatformer.py
"""
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pino.config_schema import ModelConfig
from pino.data.views import PackedSplats
from pino.model import build_model
from pino.third_party.pointcept.ptv3 import varlen_qkvpacked_attention

torch.manual_seed(0)
dev = torch.device("cuda")

# 1. varlen attention vs per-sequence reference (mixed lengths -> general path, equal -> fast path)
for lengths in ([64, 64, 64], [64, 64, 17, 64, 5]):
    cu = torch.tensor([0] + lengths, device=dev).cumsum(0).int()
    qkv = torch.randn(cu[-1].item(), 3, 4, 16, device=dev)
    out = varlen_qkvpacked_attention(qkv, cu, softmax_scale=0.25)
    ref = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        q, k, v = qkv[s:e].permute(1, 2, 0, 3)  # (H, L, D)
        ref.append(((q @ k.transpose(-1, -2) * 0.25).softmax(-1) @ v).transpose(0, 1))
    err = (out - torch.cat(ref)).abs().max().item()
    print(f"varlen attention {lengths}: max err {err:.2e}")
    assert err < 1e-4

# 2. segment_reduce == segment_csr semantics
src = torch.randn(10, 5, device=dev)
ptr = torch.tensor([0, 3, 4, 10], device=dev)
for red, fn in (("max", lambda x: x.amax(0)), ("mean", lambda x: x.mean(0))):
    got = torch.segment_reduce(src, red, offsets=ptr)
    ref = torch.stack([fn(src[a:b]) for a, b in zip(ptr[:-1].tolist(), ptr[1:].tolist())])
    assert torch.allclose(got, ref), red
print("segment_reduce ok")


# 2b. checkpointed blocks give the same outputs and gradients
def grads_for(ckpt_on, splats):
    from pino.splatformer import SplatFormer

    torch.manual_seed(1)
    m = SplatFormer(grad_checkpoint=ckpt_on).to(dev).train()
    torch.manual_seed(2)
    out = m(splats).features
    out.pow(2).mean().backward()
    return out.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


# 3. forward/backward at training scale
def random_views(num_views, n):
    counts = [n] * num_views
    total = sum(counts)
    means = torch.rand(total, 3, device=dev) * torch.tensor([8.0, 8.0, 3.0], device=dev)
    return PackedSplats(
        means=means,
        quats=F.normalize(torch.randn(total, 4, device=dev), dim=-1).abs(),
        log_scales=torch.randn(total, 3, device=dev) - 4,
        opacity_logits=torch.randn(total, device=dev),
        rgb=torch.rand(total, 3, device=dev),
        offsets=[0] + torch.tensor(counts).cumsum(0).tolist(),
    )


small = random_views(3, 3000)
out0, g0 = grads_for(False, small)
out1, g1 = grads_for(True, small)
out_err = (out0 - out1).abs().max().item()
# biases feeding straight into BatchNorm have zero true gradient (~1e-12 noise); skip them
grad_err = max(((g0[n] - g1[n]).norm() / g0[n].norm()).item() for n in g0 if g0[n].norm() > 1e-8)
print(f"checkpointing: output max err {out_err:.2e}, max relative grad err {grad_err:.2e}")
assert out_err < 1e-4 and grad_err < 1e-3

for arch in ("splatformer", "splat_transformer"):
    cfg = ModelConfig(arch=arch)
    model = build_model(cfg, 1024).to(dev).train()
    print(f"{arch}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    for num_views, n in ((2, 500), (3, 2000), (12, 40_000)):
        splats = random_views(num_views, n)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc = model(splats)
        # like training: every feature is rendered (here: squared), only a 65k sample is projected per splat
        loss = enc.features.float().pow(2).mean() + model.project(enc.features[:65536].float()).pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        assert enc.features.shape == (splats.offsets[-1], cfg.out_dim), enc.features.shape
        assert torch.isfinite(enc.features).all()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), "missing/non-finite grads"
        model.zero_grad()
        print(f"  {num_views}x{n}: {time.time() - t0:.2f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    del model
    torch.cuda.empty_cache()

# 4. existing SplatTransformer checkpoints still load
ckpt_path = Path("/scratch4/home/adempst/projects/PINO/outputs/e004_dino_align_fixed_opacity/latest.pt")
if ckpt_path.exists():
    from pino.config_schema import AppConfig

    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    app = AppConfig(**ckpt["config"])
    model = build_model(app.model, 1024).to(dev)
    model.load_state_dict(ckpt["model"])
    print(f"E004 checkpoint loads (arch={app.model.arch}, step {ckpt['step']})")
