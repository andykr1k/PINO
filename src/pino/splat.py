import numpy as np
import torch
from pathlib import Path
from plyfile import PlyData
import viser

class Splat:
    def __init__(self, ply_path: str | Path):
        self.ply_path: Path = Path(ply_path)
        self.means: np.ndarray = np.array([])
        self.colors: np.ndarray = np.array([])
        self.covariances: np.ndarray = np.array([])
        self.scales: np.ndarray = np.array([])
        self.quats: np.ndarray = np.array([])
        self.opacities: np.ndarray = np.array([])
        
        self.patch_centers = None
        self.patch_features = None
        self._load_ply()
        
    def _load_ply(self):
        print(f"Loading {self.ply_path}...")
        plydata = PlyData.read(str(self.ply_path))
        v = plydata.elements[0].data
        
        self.means = np.vstack([v['x'], v['y'], v['z']]).T
        
        if 'f_dc_0' in v.dtype.names:
            # SH DC components to RGB
            dc = np.vstack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).T
            self.colors = np.clip(dc * 0.28209 + 0.5, 0.0, 1.0)
        elif 'red' in v.dtype.names:
             self.colors = np.vstack([v['red'], v['green'], v['blue']]).T / 255.0
        else:
             self.colors = np.ones_like(self.means) * 0.5
             
        if 'scale_0' in v.dtype.names and 'rot_0' in v.dtype.names:
            log_scales = np.vstack([v['scale_0'], v['scale_1'], v['scale_2']]).T
            self.scales = np.exp(log_scales)
            
            quats = np.vstack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).T
            self.quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
            w, x, y, z = self.quats[:, 0], self.quats[:, 1], self.quats[:, 2], self.quats[:, 3]
            
            R = np.zeros((self.quats.shape[0], 3, 3))
            R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
            R[:, 0, 1] = 2.0 * (x * y - w * z)
            R[:, 0, 2] = 2.0 * (x * z + w * y)
            R[:, 1, 0] = 2.0 * (x * y + w * z)
            R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
            R[:, 1, 2] = 2.0 * (y * z - w * x)
            R[:, 2, 0] = 2.0 * (x * z - w * y)
            R[:, 2, 1] = 2.0 * (y * z + w * x)
            R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
            
            S = np.zeros((self.scales.shape[0], 3, 3))
            S[:, 0, 0] = self.scales[:, 0]
            S[:, 1, 1] = self.scales[:, 1]
            S[:, 2, 2] = self.scales[:, 2]
            
            M = R @ S
            self.covariances = M @ M.transpose(0, 2, 1)
        else:
            self.scales = np.ones_like(self.means)
            self.quats = np.zeros((self.means.shape[0], 4))
            self.quats[:, 0] = 1.0
            
        if 'opacity' in v.dtype.names:
            op = v['opacity']
            self.opacities = (1.0 / (1.0 + np.exp(-op)))[..., np.newaxis]
        else:
            self.opacities = np.ones((self.means.shape[0], 1))

    def _farthest_point_sampling(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Farthest Point Sampling algorithm using PyTorch.
        """
        N = points.shape[0]
        centroids = torch.zeros(num_samples, dtype=torch.long, device=points.device)
        distance = torch.ones(N, device=points.device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=points.device)
        
        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :].view(1, 3)
            dist = torch.sum((points - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
            
        return centroids

    def preprocess_to_patches(self, num_patches: int, k_neighbors: int) -> torch.Tensor:
        """
        Groups splats into patches using FPS and KNN, returning localized 14D features.
        Output shape: (num_patches, k_neighbors, 14)
        """
        print(f"Preprocessing to {num_patches} patches with {k_neighbors} neighbors each...")
        
        if self.means.size == 0:
            raise ValueError("No splats loaded.")
            
        # Convert to PyTorch tensors for fast operations
        means_t = torch.tensor(self.means, dtype=torch.float32)
        colors_t = torch.tensor(self.colors, dtype=torch.float32)
        scales_t = torch.tensor(self.scales, dtype=torch.float32)
        quats_t = torch.tensor(self.quats, dtype=torch.float32)
        opacities_t = torch.tensor(self.opacities, dtype=torch.float32)
        
        N = means_t.shape[0]
        actual_num_patches = min(num_patches, N)
        actual_k = min(k_neighbors, N)
        
        # 1. Farthest Point Sampling
        print("Running FPS...")
        fps_indices = self._farthest_point_sampling(means_t, actual_num_patches)
        patch_centers = means_t[fps_indices] # (num_patches, 3)
        self.patch_centers = patch_centers.numpy()
        
        # 2. K-Nearest Neighbors
        print("Running KNN...")
        chunk_size = 64
        knn_indices_list = []
        for i in range(0, actual_num_patches, chunk_size):
            end = min(i + chunk_size, actual_num_patches)
            centers_chunk = patch_centers[i:end]
            dists_chunk = torch.cdist(centers_chunk, means_t)
            knn_chunk = torch.topk(dists_chunk, k=actual_k, dim=-1, largest=False).indices
            knn_indices_list.append(knn_chunk)
            
        knn_indices = torch.cat(knn_indices_list, dim=0)
        
        # 3. Gather localized features
        print("Constructing 14D features...")
        knn_means = means_t[knn_indices] # (num_patches, k_neighbors, 3)
        knn_colors = colors_t[knn_indices] # (num_patches, k_neighbors, 3)
        knn_scales = scales_t[knn_indices] # (num_patches, k_neighbors, 3)
        knn_quats = quats_t[knn_indices] # (num_patches, k_neighbors, 4)
        knn_opacities = opacities_t[knn_indices] # (num_patches, k_neighbors, 1)
        
        # Make positions relative to patch center
        patch_centers_expanded = patch_centers.unsqueeze(1).expand(-1, actual_k, -1)
        knn_means_local = knn_means - patch_centers_expanded
        
        # Concatenate into 14D vector: [local_mean(3), color(3), scale(3), quat(4), opacity(1)]
        features = torch.cat([
            knn_means_local, 
            knn_colors, 
            knn_scales, 
            knn_quats, 
            knn_opacities
        ], dim=-1) # (num_patches, k_neighbors, 14)
        
        self.patch_features = features
        print(f"Created patch features tensor of shape {features.shape}")
        
        return features

    def visualize(self, server: viser.ViserServer):
        if self.means.size == 0 or self.colors.size == 0:
            return
            
        print("Sending splats to viser...")
        if self.covariances.size > 0 and self.opacities.size > 0:
            server.scene.add_gaussian_splats(
                name="/splats",
                centers=np.ascontiguousarray(self.means),
                covariances=np.ascontiguousarray(self.covariances),
                rgbs=np.ascontiguousarray((self.colors * 255.0).astype(np.uint8)),
                opacities=np.ascontiguousarray(self.opacities)
            )
        else:
            server.scene.add_point_cloud(
                name="/splats",
                points=np.ascontiguousarray(self.means),
                colors=np.ascontiguousarray((self.colors * 255.0).astype(np.uint8)),
                point_size=0.01
            )
            
        if self.patch_centers is not None:
            print("Sending patch centers to viser...")
            server.scene.add_point_cloud(
                name="/patch_centers",
                points=np.ascontiguousarray(self.patch_centers),
                colors=np.ascontiguousarray(np.zeros_like(self.patch_centers) + np.array([255, 0, 0], dtype=np.uint8)),
                point_size=0.05
            )
