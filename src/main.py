import hydra
from omegaconf import DictConfig, OmegaConf
from config_schema import AppConfig

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve the config to replace variables like ${now:%Y-%m-%d}
    OmegaConf.resolve(cfg)
    
    # Validate configuration against Pydantic schema
    # Convert DictConfig to a primitive dict, then load into Pydantic
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # We remove the hydra-specific configs from Pydantic validation 
    # since we only care about our custom schema.
    if "hydra" in cfg_dict:
        del cfg_dict["hydra"]

    app_config = AppConfig(**cfg_dict)
    
    print(f"Project Name: {app_config.project_name}")
    print(f"Data Directory: {app_config.paths.data_dir}")
    print(f"Output Directory: {app_config.paths.output_dir}")
    print(f"Learning Rate: {app_config.hparams.learning_rate}")
    print(f"Batch Size: {app_config.hparams.batch_size}")
    print("Configuration is valid! Let the research begin.")

if __name__ == "__main__":
    main()
