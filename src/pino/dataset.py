import torch
from torch.utils.data import Dataset
from pathlib import Path
from pino.splat import Splat

class SplatDataset(Dataset):
    def __init__(self, raw_dir: str, processed_dir: str, num_patches: int, k_neighbors: int, on_the_fly: bool = False):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.num_patches = num_patches
        self.k_neighbors = k_neighbors
        self.on_the_fly = on_the_fly
        
        self.raw_files = list(self.raw_dir.rglob("*.ply"))
        self.raw_files = [f for f in self.raw_files if "compressed" not in f.name or "uncompressed" in f.name]
        
        if not self.on_the_fly:
            self.processed_files = list(self.processed_dir.rglob("*.pt"))
            if len(self.processed_files) == 0:
                print(f"Warning: No .pt files found in {self.processed_dir}. Run preprocess_dataset.py first or use on_the_fly=True.")
            else:
                print(f"Found {len(self.processed_files)} processed .pt files in {self.processed_dir}.")
        else:
            print(f"Found {len(self.raw_files)} raw .ply files in {self.raw_dir}.")
            
    def __len__(self):
        if self.on_the_fly:
            return len(self.raw_files)
        return len(self.processed_files)

    def __getitem__(self, index):
        if self.on_the_fly:
            ply_path = self.raw_files[index]
            splat = Splat(ply_path)
            # Suppress print statements from splat processing by wrapping or assuming it's fast. 
            # In on-the-fly mode, this will print heavily. It's recommended to do offline.
            features = splat.preprocess_to_patches(num_patches=self.num_patches, k_neighbors=self.k_neighbors)
            centers = torch.tensor(splat.patch_centers, dtype=torch.float32)
            return features, centers
        else:
            pt_path = self.processed_files[index]
            data = torch.load(pt_path, weights_only=True)
            return data["features"], data["centers"]
