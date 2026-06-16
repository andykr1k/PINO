import hydra
import viser
import time
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pino.config_schema import AppConfig
from pino.dataset import SplatDataset
from torch.utils.data import DataLoader
from pino.dino import DINO_3DGS, DINOLoss
from pino.augmentations import random_3d_rotation, random_jitter, mask_patches
import torch
import time
import wandb
from tqdm import tqdm

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve the config to replace variables like ${now:%Y-%m-%d}
    OmegaConf.resolve(cfg)
    
    # Validate configuration against Pydantic schema
    from typing import Any
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Expected dict from OmegaConf, got {type(cfg_dict)}")
        
    cfg_dict_str_keys: dict[str, Any] = {str(k): v for k, v in cfg_dict.items() if k != "hydra"}
        
    app_config = AppConfig(**cfg_dict_str_keys)
    
    if app_config.wandb.enabled:
        project_name = app_config.wandb.project or app_config.project_name
        wandb.init(
            project=project_name,
            entity=app_config.wandb.entity,
            config=cfg_dict_str_keys,
        )
    
    print(f"Project Name: {app_config.project_name}")
    print(f"InteriorGS Dataset Path: {app_config.paths.dataset.InteriorGS}")
    print(f"Processed Dataset Path: {app_config.paths.dataset.ProcessedGS}")
    
    dataset = SplatDataset(
        raw_dir=app_config.paths.dataset.InteriorGS,
        processed_dir=app_config.paths.dataset.ProcessedGS,
        num_patches=app_config.preprocess.num_patches,
        k_neighbors=app_config.preprocess.k_neighbors,
        on_the_fly=app_config.preprocess.on_the_fly
    )
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=app_config.hparams.batch_size, 
        shuffle=True, 
        num_workers=app_config.preprocess.num_workers
    )
    
    model = DINO_3DGS(app_config)
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = DINOLoss(
        out_dim=app_config.dino.out_dim,
        teacher_temp=app_config.dino.teacher_temp,
        student_temp=app_config.dino.student_temp
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=app_config.hparams.learning_rate)
    
    print(f"Running DINO Training Loop on {device}...")
    start_time = time.time()
    
    for epoch in range(app_config.hparams.epochs):
        epoch_loss = 0.0
        
        # Wrap dataloader with tqdm
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{app_config.hparams.epochs}]")
        
        for batch_idx, (patch_features_b, patch_centers_b) in enumerate(pbar):
            patch_features_b = patch_features_b.to(device)
            patch_centers_b = patch_centers_b.to(device)
            
            optimizer.zero_grad()

            # 1. Generate Teacher View (Unmasked, Mild Jitter)
            teacher_centers = random_jitter(patch_centers_b, std=0.01)
            teacher_features = patch_features_b.clone()
            
            with torch.no_grad():
                teacher_output_full = model.forward_teacher(teacher_features, teacher_centers)
                
            # 2. Generate Student View (Masked, Rotated)
            student_features, student_centers = random_3d_rotation(patch_features_b, patch_centers_b)
            student_features_masked, student_centers_masked, ids_keep = mask_patches(
                student_features, student_centers, mask_ratio=app_config.dino.mask_ratio
            )
            
            student_output_masked = model.forward_student(student_features_masked, student_centers_masked)
            
            # 3. Match teacher and student lengths for loss
            out_dim = teacher_output_full.shape[-1]
            ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, out_dim)
            teacher_output_masked = torch.gather(teacher_output_full, dim=1, index=ids_keep_expanded)
            
            # 4. Compute Loss
            loss = criterion(student_output_masked, teacher_output_masked)
            
            # 5. Backprop & Update
            loss.backward()
            optimizer.step()
            
            # 6. Update Teacher EMA and Loss Center
            model.update_teacher()
            criterion.update_center(teacher_output_full)
            
            epoch_loss += loss.item()
            
            # Update tqdm progress bar with the current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            if app_config.wandb.enabled:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                    "batch_idx": batch_idx + epoch * len(dataloader)
                })
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"=== Epoch [{epoch+1}/{app_config.hparams.epochs}] completed. Avg Loss: {avg_loss:.4f} ===")
        if app_config.wandb.enabled:
            wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
            
    end_time = time.time()
    print(f"Training complete! Took {end_time - start_time:.4f} seconds.")
    
    if app_config.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
