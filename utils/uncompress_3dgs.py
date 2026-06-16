import os
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from pino.config_schema import AppConfig

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    if "hydra" in cfg_dict:
        del cfg_dict["hydra"]

    app_config = AppConfig(**cfg_dict)
    
    # Use the InteriorGS dataset path from config
    dataset_path = Path(app_config.paths.dataset.InteriorGS)

    if not dataset_path.exists():
        print(f"Error: Path {dataset_path} does not exist.")
        return

    ply_files = list(dataset_path.rglob("*.ply"))
    
    if not ply_files:
        print(f"No .ply files found in {dataset_path}")
        return

    print(f"Found {len(ply_files)} .ply files in {dataset_path}.")

    output_suffix = "_uncompressed.ply"
    input_suffix = "_compressed"
    overwrite = False

    for ply_file in ply_files:
        if not overwrite and ply_file.name.endswith(output_suffix):
            continue
            
        if overwrite:
            output_file = ply_file
        else:
            file_name = ply_file.stem.replace(input_suffix, output_suffix)
            output_file = ply_file.with_name(f"{file_name}")
            
        print(f"Processing {ply_file} -> {output_file}")
        
        cmd = ["npx", "@playcanvas/splat-transform", str(ply_file), str(output_file)]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully processed {ply_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {ply_file}")
            print(f"Command run: {' '.join(cmd)}")
            print(f"Error output: {e.stderr}")
        except FileNotFoundError:
            print("Error: command not found. Please ensure it is installed and in your PATH.")
            break

if __name__ == "__main__":
    main()