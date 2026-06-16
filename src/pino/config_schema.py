from typing import Optional, Dict
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    InteriorGS: str = Field(..., description="Path to the InteriorGS dataset")

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

class AppConfig(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    paths: PathsConfig
    hparams: HParamsConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
