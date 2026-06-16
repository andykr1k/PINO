import os
import torch
import hydra
from pathlib import Path
from multiprocessing import Pool
from pino.config_schema import AppConfig
from omegaconf import DictConfig, OmegaConf
from pino.splat import Splat
import sys
from tqdm import tqdm

# Suppress prints from workers
class DummyWriter:
    def write(self, x): pass
    def flush(self): pass

def process_single_ply(args):
    ply_path, raw_dir, out_dir, num_patches, k_neighbors = args
    
    rel_path = Path(ply_path).relative_to(Path(raw_dir))
    out_path = Path(out_dir) / rel_path.with_suffix(".pt")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.exists():
        return f"Skipped {rel_path} (already processed)"
        
    try:
        # Suppress prints inside the worker to prevent console spam
        sys.stdout = DummyWriter()
        splat = Splat(ply_path)
        features = splat.preprocess_to_patches(num_patches=num_patches, k_neighbors=k_neighbors)
        centers = torch.tensor(splat.patch_centers, dtype=torch.float32)
        
        torch.save({"features": features, "centers": centers}, out_path)
        sys.stdout = sys.__stdout__
        return f"Successfully processed {rel_path}"
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error processing {rel_path}: {str(e)}"

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Configuration must be a dictionary")
    cfg_dict_str_keys = {str(k): v for k, v in cfg_dict.items() if k != "hydra"}
    app_config = AppConfig(**cfg_dict_str_keys)
    
    raw_dir = Path(app_config.paths.dataset.InteriorGS)
    out_dir = Path(app_config.paths.dataset.ProcessedGS)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = list(raw_dir.rglob("*.ply"))
    raw_files = [f for f in raw_files if "compressed" not in f.name or "uncompressed" in f.name]
    
    print(f"Found {len(raw_files)} PLY files to process.")
    print(f"Saving processed tensors to {out_dir}")
    
    num_patches = app_config.preprocess.num_patches
    k_neighbors = app_config.preprocess.k_neighbors
    num_workers = app_config.preprocess.num_workers
    
    args_list = [(str(f), str(raw_dir), str(out_dir), num_patches, k_neighbors) for f in raw_files]
    
    print(f"Starting multiprocessing with {num_workers} workers...")
    
    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(process_single_ply, args_list), total=len(args_list), desc="Processing PLY files"))
        
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
