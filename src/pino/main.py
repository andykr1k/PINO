import hydra
import viser
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pino.config_schema import AppConfig
from pino.splat import Splat

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve the config to replace variables like ${now:%Y-%m-%d}
    OmegaConf.resolve(cfg)
    
    # Validate configuration against Pydantic schema
    from typing import Any
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Expected dict from OmegaConf, got {type(cfg_dict)}")
        
    cfg_dict_str_keys: dict[str, Any] = {str(k): v for k, v in cfg_dict.items()}
        
    if "hydra" in cfg_dict_str_keys:
        del cfg_dict_str_keys["hydra"]

    app_config = AppConfig(**cfg_dict_str_keys)
    
    print(f"Project Name: {app_config.project_name}")
    print(f"InteriorGS Dataset Path: {app_config.paths.dataset.InteriorGS}")
    
    dataset_path = Path(app_config.paths.dataset.InteriorGS)
    if not dataset_path.exists():
        print(f"Error: Path {dataset_path} does not exist.")
        return
        
    ply_files = list(dataset_path.rglob("*.ply"))
    # Filter out compressed splats as they need uncompressing first
    ply_files = [f for f in ply_files if "compressed" not in f.name or "uncompressed" in f.name]
    if not ply_files:
        print(f"No .ply files found in {dataset_path}")
        return
        
    # Use the first PLY file for demonstration
    target_ply = ply_files[0]
    
    # Initialize viser
    server = viser.ViserServer()
    print("Viser server started. You can view it in your browser.")

    # Load and preprocess
    splat = Splat(target_ply)
    
    # Time the preprocessing step
    start_time = time.time()
    splat.preprocess_to_voxels(voxel_size=app_config.preprocess.voxel_size)
    end_time = time.time()
    print(f"Preprocessing took {end_time - start_time:.4f} seconds.")
    
    splat.visualize(server)
    
    # Keep server alive
    print("Visualization complete. Press Ctrl+C to exit.")
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
