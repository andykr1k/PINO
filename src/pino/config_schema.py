from typing import Optional, Dict
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    InteriorGS: str = Field(..., description="Path to the InteriorGS dataset")
    ProcessedGS: str = Field(..., description="Path to the ProcessedGS dataset")

class PathsConfig(BaseModel):
    dataset: DatasetConfig = Field(..., description="Dataset paths configurations")
    output_dir: Optional[str] = Field(None, description="Path to the output directory")

class HParamsConfig(BaseModel):
    learning_rate: float = Field(..., description="Learning rate for the optimizer")
    batch_size: int = Field(..., description="Batch size for training")
    epochs: int = Field(..., description="Number of epochs to train")

class PreprocessConfig(BaseModel):
    num_patches: int = Field(1024, description="Number of patches (FPS centers) to sample")
    k_neighbors: int = Field(32, description="Number of neighbors per patch (KNN)")
    on_the_fly: bool = Field(False, description="Whether to preprocess from PLY on the fly during training")
    num_workers: int = Field(8, description="Number of CPU workers for offline preprocessing / dataloading")

class ModelConfig(BaseModel):
    embed_dim: int = Field(384, description="Embedding dimension of the Transformer")
    depth: int = Field(12, description="Number of Transformer layers")
    num_heads: int = Field(6, description="Number of attention heads")

class DINOConfig(BaseModel):
    out_dim: int = Field(256, description="Output dimension of the DINO head")
    momentum_teacher: float = Field(0.996, description="EMA momentum for teacher updates")
    teacher_temp: float = Field(0.04, description="Temperature for teacher logits")
    student_temp: float = Field(0.1, description="Temperature for student logits")
    mask_ratio: float = Field(0.6, description="Ratio of patches to mask in student view")

class WandbConfig(BaseModel):
    enabled: bool = Field(False, description="Whether to use Weights & Biases for logging")
    entity: Optional[str] = Field(None, description="Wandb entity (username or team)")
    project: Optional[str] = Field(None, description="Wandb project name. Defaults to AppConfig.project_name if None")

class AppConfig(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    paths: PathsConfig
    hparams: HParamsConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dino: DINOConfig = Field(default_factory=DINOConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
