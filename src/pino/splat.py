import numpy as np
from pathlib import Path
from plyfile import PlyData
import viser
from pino.voxel import Voxel

from typing import Optional
from pydantic import BaseModel, Field

class Splat:
    def __init__(self, ply_path: str | Path):
        self.ply_path: Path = Path(ply_path)
        self.means: np.ndarray = np.array([])
        self.colors: np.ndarray = np.array([])
        self.covariances: np.ndarray = np.array([])
        self.opacities: np.ndarray = np.array([])
        self.voxels: dict[tuple[int, int, int], Voxel] = {}
        self.voxel_size: float = 0.0
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
            scales = np.exp(log_scales)
            
            quats = np.vstack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']]).T
            quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
            w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
            
            R = np.zeros((quats.shape[0], 3, 3))
            R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
            R[:, 0, 1] = 2.0 * (x * y - w * z)
            R[:, 0, 2] = 2.0 * (x * z + w * y)
            R[:, 1, 0] = 2.0 * (x * y + w * z)
            R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
            R[:, 1, 2] = 2.0 * (y * z - w * x)
            R[:, 2, 0] = 2.0 * (x * z - w * y)
            R[:, 2, 1] = 2.0 * (y * z + w * x)
            R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
            
            S = np.zeros((scales.shape[0], 3, 3))
            S[:, 0, 0] = scales[:, 0]
            S[:, 1, 1] = scales[:, 1]
            S[:, 2, 2] = scales[:, 2]
            
            M = R @ S
            self.covariances = M @ M.transpose(0, 2, 1)
            
        if 'opacity' in v.dtype.names:
            op = v['opacity']
            self.opacities = (1.0 / (1.0 + np.exp(-op)))[..., np.newaxis]
            
    def preprocess_to_voxels(self, voxel_size: float):
        print(f"Preprocessing to voxels of size {voxel_size}...")
        self.voxel_size = voxel_size
        self.voxels = {}
        
        if self.means.size == 0:
            return
            
        min_bounds = self.means.min(axis=0)
        
        # Calculate grid indices for all means
        grid_indices = np.floor((self.means - min_bounds) / voxel_size).astype(int)
        
        for idx, g_idx in enumerate(grid_indices):
            g_idx_tuple = tuple(g_idx)
            if g_idx_tuple not in self.voxels:
                voxel_min = min_bounds + np.array(g_idx) * voxel_size
                voxel_max = voxel_min + voxel_size
                self.voxels[g_idx_tuple] = Voxel(
                    grid_index=g_idx_tuple,
                    min_bounds=voxel_min,
                    max_bounds=voxel_max
                )
            self.voxels[g_idx_tuple].add_splat(idx)
        print(f"Created {len(self.voxels)} voxels.")
            
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
        
        print("Sending voxels to viser...")
        for g_idx, voxel in self.voxels.items():
            name_str = f"/voxels/voxel_{g_idx[0]}_{g_idx[1]}_{g_idx[2]}"
            # We use add_box to render the voxel bounds. We set a low opacity so we can see through it, or wireframe if supported.
            # Typically in viser, setting a color and viewing it serves the purpose. 
            server.scene.add_box(
                name=name_str,
                dimensions=(self.voxel_size, self.voxel_size, self.voxel_size),
                position=voxel.center,
                color=(255, 0, 0),
                wireframe=True
            )
