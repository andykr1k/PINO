import torch
import math

def random_3d_rotation(patch_features: torch.Tensor, patch_centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a random 3D rotation (around Y axis) to the patches and centers.
    """
    B = patch_centers.shape[0]
    device = patch_centers.device
    
    angles = torch.rand(B, device=device) * 2 * math.pi
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    rot_matrices = torch.zeros(B, 3, 3, device=device)
    rot_matrices[:, 0, 0] = cos_a
    rot_matrices[:, 0, 2] = sin_a
    rot_matrices[:, 1, 1] = 1.0
    rot_matrices[:, 2, 0] = -sin_a
    rot_matrices[:, 2, 2] = cos_a
    
    # Rotate centers
    new_centers = torch.bmm(patch_centers, rot_matrices.transpose(1, 2))
    
    new_features = patch_features.clone()
    local_means = new_features[:, :, :, 0:3]
    B, P, K, _ = local_means.shape
    local_means_flat = local_means.view(B, P*K, 3)
    rotated_local_means = torch.bmm(local_means_flat, rot_matrices.transpose(1, 2))
    new_features[:, :, :, 0:3] = rotated_local_means.view(B, P, K, 3)
    
    return new_features, new_centers

def random_jitter(patch_centers: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Adds small gaussian noise to centers"""
    noise = torch.randn_like(patch_centers) * std
    return patch_centers + noise

def mask_patches(patch_features: torch.Tensor, patch_centers: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly masks out a percentage of patches.
    Returns: kept_features, kept_centers, ids_keep
    """
    B, num_patches, K, C = patch_features.shape
    num_keep = int(num_patches * (1 - mask_ratio))
    
    noise = torch.rand(B, num_patches, device=patch_features.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    
    ids_keep = ids_shuffle[:, :num_keep]
    
    ids_keep_feat = ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, C)
    kept_features = torch.gather(patch_features, dim=1, index=ids_keep_feat)
    
    ids_keep_centers = ids_keep.unsqueeze(-1).expand(-1, -1, 3)
    kept_centers = torch.gather(patch_centers, dim=1, index=ids_keep_centers)
    
    return kept_features, kept_centers, ids_keep
