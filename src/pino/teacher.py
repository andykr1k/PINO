"""Frozen DINOv3 image teacher producing patch-token targets for rendered views."""
from dataclasses import dataclass

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TeacherOutput:
    patch_tokens: torch.Tensor  # (B, h, w, C) float32
    cls_token: torch.Tensor  # (B, C) float32


class DINOv3Teacher(nn.Module):
    def __init__(
        self,
        checkpoint: str,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(checkpoint, cache_dir=cache_dir, local_files_only=local_files_only)
        self.model.eval().requires_grad_(False)
        self.dtype = dtype
        self.patch_size: int = self.model.config.patch_size
        self.embed_dim: int = self.model.config.hidden_size
        self.num_register_tokens: int = self.model.config.num_register_tokens
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> TeacherOutput:
        """images: (B, 3, H, W) in [0, 1] with H and W multiples of the patch size."""
        b, _, height, width = images.shape
        h, w = height // self.patch_size, width // self.patch_size
        assert h * self.patch_size == height and w * self.patch_size == width
        x = (images - self.mean) / self.std
        with torch.autocast(device_type=images.device.type, dtype=self.dtype):
            tokens = self.model(pixel_values=x).last_hidden_state
        tokens = tokens.float()
        patch_tokens = tokens[:, 1 + self.num_register_tokens :].reshape(b, h, w, self.embed_dim)
        return TeacherOutput(patch_tokens=patch_tokens, cls_token=tokens[:, 0])
