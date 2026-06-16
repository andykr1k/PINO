import torch
import torch.nn as nn

class SplatPatchEmbed(nn.Module):
    """
    Point-MAE style Patch Embedding for Gaussian Splats.
    Processes localized 14D features for K splats in a patch and aggregates them.
    """
    def __init__(self, in_channels: int = 14, embed_dim: int = 384):
        super().__init__()
        # A simple MLP to process each splat independently
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_patches, K, in_channels)
        Returns: (B, num_patches, embed_dim)
        """
        # Apply MLP across the K splats
        x = self.mlp(x) # (B, num_patches, K, embed_dim)
        
        # Max-pool over the K splats to get a single token per patch
        x = torch.max(x, dim=2).values # (B, num_patches, embed_dim)
        return x

class PositionalEmbedding(nn.Module):
    """
    Simple Linear projection for 3D coordinates.
    """
    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """
        centers: (B, num_patches, 3)
        Returns: (B, num_patches, embed_dim)
        """
        return self.pos_proj(centers)

class Native3DTransformer(nn.Module):
    """
    Native 3D Transformer processing Gaussian Splat patches.
    """
    def __init__(self, in_channels: int = 14, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.patch_embed = SplatPatchEmbed(in_channels=in_channels, embed_dim=embed_dim)
        self.pos_embed = PositionalEmbedding(embed_dim=embed_dim)
        
        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_features: torch.Tensor, patch_centers: torch.Tensor) -> torch.Tensor:
        """
        patch_features: (B, num_patches, K, 14)
        patch_centers: (B, num_patches, 3)
        Returns: (B, num_patches, embed_dim)
        """
        # 1. Patch Embedding (MLP + Pool)
        x = self.patch_embed(patch_features) # (B, num_patches, embed_dim)
        
        # 2. Add Positional Embedding
        pos = self.pos_embed(patch_centers) # (B, num_patches, embed_dim)
        x = x + pos
        
        # 3. Transformer Encoder
        x = self.transformer(x) # (B, num_patches, embed_dim)
        
        # 4. Final Norm
        x = self.norm(x)
        return x
