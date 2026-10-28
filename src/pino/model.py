"""Native 3D transformer over Gaussian splats producing per-splat embeddings.

Pipeline (per view, batched over views):
  1. Farthest point sampling picks `num_groups` patch centres; every splat joins its nearest centre
     (Voronoi grouping, so no splat is dropped, unlike a fixed-K KNN).
  2. A Point-MAE style mini-PointNet embeds each splat's localised 14D feature
     [position - centre (3), rgb (3), log-scale (3), quaternion (4), opacity (1)] and max-pools
     each group into a token.
  3. A pre-norm transformer encoder mixes the group tokens (with Fourier positional embeddings
     of the centres).
  4. Each splat's embedding is decoded from its own point feature plus the inverse-distance
     interpolation of its `k_interp` nearest group tokens.

`project` maps splat embeddings into the teacher (DINO) space. It is linear, so it commutes with
alpha blending: projecting a rendered embedding map equals rendering the projected embeddings.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pino.data.views import PackedSplats


@dataclass
class SplatEncoding:
    features: torch.Tensor  # (S, out_dim) per-splat embeddings
    tokens: torch.Tensor  # (B, G, token_dim) group tokens after the transformer
    centers: torch.Tensor  # (B, G, 3) group centres (model frame)


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int, min_wavelength: float, max_wavelength: float):
        super().__init__()
        wavelengths = torch.logspace(math.log10(max_wavelength), math.log10(min_wavelength), num_freqs)
        self.register_buffer("freqs", 2.0 * math.pi / wavelengths, persistent=False)
        self.out_dim = in_dim * (2 * num_freqs + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = (x[..., None] * self.freqs).flatten(-2)
        return torch.cat([x, angles.sin(), angles.cos()], dim=-1)


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


@torch.no_grad()
def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Batched FPS. points: (B, P, 3) -> indices (B, num_samples)."""
    b, p, _ = points.shape
    idx = torch.empty(b, num_samples, dtype=torch.long, device=points.device)
    dist = torch.full((b, p), float("inf"), device=points.device)
    farthest = torch.randint(0, p, (b,), device=points.device)
    batch = torch.arange(b, device=points.device)
    for i in range(num_samples):
        idx[:, i] = farthest
        d = ((points - points[batch, farthest][:, None]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return idx


@torch.no_grad()
def knn_to_centers(points: torch.Tensor, centers: torch.Tensor, k: int, chunk: int = 32768):
    """points: (N, 3), centers: (G, 3) -> (dist (N, k), idx (N, k)) sorted nearest first."""
    dists, idxs = [], []
    for start in range(0, points.shape[0], chunk):
        d = torch.cdist(points[start : start + chunk], centers)
        top = d.topk(k, dim=-1, largest=False)
        dists.append(top.values)
        idxs.append(top.indices)
    return torch.cat(dists), torch.cat(idxs)


class SplatTransformer(nn.Module):
    def __init__(
        self,
        num_groups: int = 1024,
        fps_subset: int = 16384,
        splat_dim: int = 128,
        token_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        out_dim: int = 128,
        teacher_dim: int = 1024,
        k_interp: int = 3,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.fps_subset = fps_subset
        self.k_interp = k_interp

        self.local_pos_enc = FourierFeatures(3, num_freqs=6, min_wavelength=0.02, max_wavelength=2.0)
        self.center_pos_enc = FourierFeatures(3, num_freqs=8, min_wavelength=0.1, max_wavelength=32.0)
        attr_dim = 11  # rgb (3), log-scale (3), quaternion (4), opacity (1)

        self.point_embed = mlp(self.local_pos_enc.out_dim + attr_dim, splat_dim, splat_dim)
        self.point_refine = mlp(2 * splat_dim, 2 * splat_dim, splat_dim)
        self.token_embed = nn.Linear(splat_dim, token_dim)
        self.center_embed = mlp(self.center_pos_enc.out_dim, token_dim, token_dim)
        self.blocks = nn.ModuleList([Block(token_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(token_dim)
        self.decoder = mlp(splat_dim + token_dim, 2 * splat_dim, out_dim)
        self.head = nn.Linear(out_dim, teacher_dim)

        # Running per-channel statistics of the teacher tokens. DINO patch tokens share a large common
        # component, so targets are standardised before the loss; otherwise predicting the mean token
        # already scores a high cosine similarity.
        self.target_momentum = 0.99
        self.register_buffer("target_mean", torch.zeros(teacher_dim))
        self.register_buffer("target_std", torch.ones(teacher_dim))
        self.register_buffer("target_stats_initialized", torch.tensor(False))

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Maps splat embeddings (or alpha-normalised renders of them) into standardised teacher space."""
        return self.head(features)

    @torch.no_grad()
    def update_target_stats(self, tokens: torch.Tensor) -> None:
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        mean, std = flat.mean(0), flat.std(0).clamp(min=1e-4)
        if not bool(self.target_stats_initialized):
            self.target_mean.copy_(mean)
            self.target_std.copy_(std)
            self.target_stats_initialized.fill_(True)
        else:
            self.target_mean.lerp_(mean, 1.0 - self.target_momentum)
            self.target_std.lerp_(std, 1.0 - self.target_momentum)

    def standardize_targets(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - self.target_mean) / self.target_std

    def to_teacher_space(self, projected: torch.Tensor) -> torch.Tensor:
        """Inverse of `standardize_targets`: projected embeddings -> raw DINO token space."""
        return projected * self.target_std + self.target_mean

    @staticmethod
    def normalize_inputs(splats: PackedSplats) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns model-frame positions (each view centred in xy, z kept since the floor is at 0) and attributes."""
        view_ids = splats.view_ids
        xy_sum = torch.zeros(splats.num_views, 2, device=splats.means.device).index_add_(0, view_ids, splats.means[:, :2])
        counts = torch.tensor(splats.counts, device=splats.means.device, dtype=torch.float32).clamp(min=1)
        offset = F.pad(xy_sum / counts[:, None], (0, 1))
        pos = splats.means - offset[view_ids]
        attrs = torch.cat(
            [
                splats.rgb * 2.0 - 1.0,
                ((splats.log_scales + 4.0) / 2.0).clamp(-4.0, 4.0),
                splats.quats,
                splats.opacity_logits.sigmoid()[:, None] * 2.0 - 1.0,
            ],
            dim=-1,
        )
        return pos, attrs

    @torch.no_grad()
    def group(self, pos: torch.Tensor, splats: PackedSplats):
        """FPS centres per view and each splat's k nearest centres (indices are global: view * G + g)."""
        device = pos.device
        b, g = splats.num_views, self.num_groups
        subsets = []
        for v in range(b):
            n = splats.counts[v]
            sel = torch.randint(0, n, (self.fps_subset,), device=device) if n > self.fps_subset else torch.arange(n, device=device)
            if n < self.fps_subset:
                sel = torch.cat([sel, sel[torch.randint(0, n, (self.fps_subset - n,), device=device)]])
            subsets.append(pos[splats.view_slice(v)][sel])
        subsets_t = torch.stack(subsets)
        center_idx = farthest_point_sampling(subsets_t, g)
        centers = torch.gather(subsets_t, 1, center_idx[..., None].expand(-1, -1, 3))

        nbr_dist, nbr_idx = [], []
        for v in range(b):
            d, i = knn_to_centers(pos[splats.view_slice(v)], centers[v], self.k_interp)
            nbr_dist.append(d)
            nbr_idx.append(i + v * g)
        return centers, torch.cat(nbr_dist), torch.cat(nbr_idx)

    def forward(self, splats: PackedSplats) -> SplatEncoding:
        b, g = splats.num_views, self.num_groups
        pos, attrs = self.normalize_inputs(splats)
        centers, nbr_dist, nbr_idx = self.group(pos, splats)
        group_id = nbr_idx[:, 0]

        local = pos - centers.reshape(-1, 3)[group_id]
        h = self.point_embed(torch.cat([self.local_pos_enc(local), attrs], dim=-1))
        pooled = self._group_max(h, group_id, b * g)
        h = self.point_refine(torch.cat([h, pooled[group_id]], dim=-1))
        pooled = self._group_max(h, group_id, b * g)

        tokens = self.token_embed(pooled).view(b, g, -1) + self.center_embed(self.center_pos_enc(centers))
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        weights = 1.0 / (nbr_dist + 1e-3)
        weights = (weights / weights.sum(-1, keepdim=True)).to(tokens.dtype)
        interp = (tokens.reshape(b * g, -1)[nbr_idx] * weights[..., None]).sum(1)
        features = self.decoder(torch.cat([h, interp.to(h.dtype)], dim=-1))
        return SplatEncoding(features=features, tokens=tokens, centers=centers)

    @staticmethod
    def _group_max(x: torch.Tensor, group_id: torch.Tensor, num_groups: int) -> torch.Tensor:
        out = torch.full((num_groups, x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
        out = out.scatter_reduce(0, group_id[:, None].expand_as(x), x, reduce="amax", include_self=True)
        # FPS on a sampled subset can pick duplicate centres, leaving some groups empty.
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))
