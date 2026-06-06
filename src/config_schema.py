from pydantic import BaseModel, Field

class PathsConfig(BaseModel):
    data_dir: str = Field(..., description="Path to the data directory")
    output_dir: str = Field(..., description="Path to the output directory")

class HParamsConfig(BaseModel):
    learning_rate: float = Field(..., description="Learning rate for the optimizer")
    batch_size: int = Field(..., description="Batch size for training")
    epochs: int = Field(..., description="Number of epochs to train")

class AppConfig(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    paths: PathsConfig
    hparams: HParamsConfig
