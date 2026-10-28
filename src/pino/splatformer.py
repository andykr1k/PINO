"""SplatFormer baseline: the Point Transformer V3 backbone of SplatFormer (Chen et al., ICLR 2025,
github.com/ChenYutongTHU/SplatFormer @ 446ffb5) as a per-splat encoder for DINO alignment.

Faithful to SplatFormer's `FeaturePredictor` + `configs/model/ptv3.gin`:
  - every splat is a token; each view is min-max normalised into the unit cube (aspect ratio kept) and voxelised
    at 1/384 for serialisation and the sparse-conv positional encoding; log-scales shift by the same factor;
  - PTv3 U-Net: MLP embedding, encoder depths (2, 2, 2, 6, 2) / channels (64, 96, 128, 256, 512) / heads
    (2, 4, 8, 16, 32), pooling strides (1, 2, 2, 2), decoder depths (2, 2, 2, 2) / channels (96, 96, 128, 256),
    attention over patches of 1024 consecutive splats along z / z-trans / hilbert / hilbert-trans curves (block i
    uses curve i % 4, with the curve order shuffled every forward), drop path 0.3;
  - output head: `head_nlayer`-layer ReLU MLP of width 128 on [backbone feature (96), input feature].

Deliberate differences (the task differs: SplatFormer predicts residual Gaussian updates):
  - inputs omit higher-order SH (`features_rest`), which PackedSplats does not carry; SplatTransformer gets the
    same attributes. SH DC is recovered from the clamped RGB;
  - the head outputs an `out_dim` embedding (no residual activations, no zero-init of its last layer), followed
    by the shared linear projection into teacher space;
  - flash-attn / torch_scatter are replaced by equivalent torch ops, and PTv3 blocks are activation-checkpointed
    (same maths, less memory); see pino/third_party/pointcept/README.md.
"""
import torch
from torch import nn

from pino.data.scene import SH_C0
from pino.data.views import PackedSplats
from pino.model import SplatEncoding, TeacherAlignedModel
from pino.third_party.pointcept.ptv3 import Block, PointTransformerV3

# SplatFormer's PointTransformerV3Model channel presets, keyed by enc_dim / output_dim.
ENC_CHANNELS = {32: (32, 64, 128, 256, 512), 64: (64, 96, 128, 256, 512)}
DEC_CHANNELS = {64: (64, 64, 128, 256), 96: (96, 96, 128, 256), 128: (128, 128, 256, 256)}
IN_CHANNELS = 14  # means (3), scales (3), opacities (1), quats (4), features_dc (3)


class SplatFormer(TeacherAlignedModel):
    def __init__(
        self,
        grid_resolution: int = 384,
        enc_dim: int = 64,
        backbone_dim: int = 96,
        stride: tuple[int, ...] = (1, 2, 2, 2),
        patch_size: int = 1024,
        drop_path: float = 0.3,
        head_nlayer: int = 4,
        head_width: int = 128,
        grad_checkpoint: bool = True,
        out_dim: int = 128,
        teacher_dim: int = 1024,
    ):
        super().__init__(out_dim, teacher_dim)
        self.grid_resolution = grid_resolution
        enc_channels, dec_channels = ENC_CHANNELS[enc_dim], DEC_CHANNELS[backbone_dim]
        self.backbone = PointTransformerV3(
            in_channels=IN_CHANNELS,
            embedding_type="MLP",
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=stride,
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=enc_channels,
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(patch_size,) * len(enc_channels),
            dec_depths=(2, 2, 2, 2),
            dec_channels=dec_channels,
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(patch_size,) * len(dec_channels),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=drop_path,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
            cls_mode=False,
            turn_off_bn=False,
        )
        # A batch is ~1.8M splat tokens; without checkpointing the full-resolution stages need > 60 GB.
        for module in self.backbone.modules():
            if isinstance(module, Block):
                module.use_checkpoint = grad_checkpoint
        layers: list[nn.Module] = []
        in_dim = dec_channels[0] + IN_CHANNELS
        for _ in range(head_nlayer - 1):
            layers += [nn.Linear(in_dim, head_width), nn.ReLU()]
            in_dim = head_width
        layers.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*layers)

    @staticmethod
    def normalize_inputs(splats: PackedSplats) -> torch.Tensor:
        """SplatFormer's per-scene MinMaxScaler (preserve_ratio): each view's means into [0, 1]^3, centred, with
        the largest extent spanning 1; log-scales shifted by log(scale). Returns the (S, 14) input features."""
        view_ids = splats.view_ids
        lo = torch.full((splats.num_views, 3), float("inf"), device=splats.means.device)
        hi = torch.full((splats.num_views, 3), float("-inf"), device=splats.means.device)
        lo = lo.scatter_reduce(0, view_ids[:, None].expand(-1, 3), splats.means, reduce="amin")
        hi = hi.scatter_reduce(0, view_ids[:, None].expand(-1, 3), splats.means, reduce="amax")
        scale = 1.0 / (hi - lo).amax(-1).clamp(min=1e-6)
        trans = 0.5 - (lo + hi) / 2 * scale[:, None]
        means = splats.means * scale[view_ids, None] + trans[view_ids]
        return torch.cat(
            [
                means,
                splats.log_scales + scale.log()[view_ids, None],
                splats.opacity_logits[:, None],
                splats.quats,
                (splats.rgb - 0.5) / SH_C0,
            ],
            dim=-1,
        )

    def forward(self, splats: PackedSplats) -> SplatEncoding:
        feat = self.normalize_inputs(splats)
        coord = feat[:, :3].contiguous()
        point = self.backbone(
            {
                "coord": coord,
                "grid_coord": torch.floor(coord * self.grid_resolution).int(),
                "grid_size": torch.ones(3, device=coord.device) / self.grid_resolution,
                "offset": torch.tensor(splats.offsets[1:], device=coord.device, dtype=torch.long),
                "feat": feat,
            }
        )
        features = self.decoder(torch.cat([point.feat, feat.to(point.feat.dtype)], dim=-1))
        return SplatEncoding(features=features)
