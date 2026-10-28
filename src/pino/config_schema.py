from typing import Literal, Optional
from pydantic import BaseModel, Field

class DatasetConfig(BaseModel):
    InteriorGS: str = Field(..., description="Path to the InteriorGS dataset")

class PathsConfig(BaseModel):
    dataset: DatasetConfig = Field(..., description="Dataset paths configurations")
    output_dir: Optional[str] = Field(None, description="Path to the output directory")

class HParamsConfig(BaseModel):
    learning_rate: float = Field(..., description="Peak learning rate for AdamW")
    weight_decay: float = Field(0.05, description="AdamW weight decay")
    max_steps: int = Field(..., description="Number of optimisation steps")
    warmup_steps: int = Field(1000, description="Linear warmup steps before cosine decay")
    min_lr_ratio: float = Field(0.01, description="Final learning rate as a fraction of the peak")
    grad_clip: float = Field(1.0, description="Gradient norm clipping threshold")

class PreprocessConfig(BaseModel):
    num_patches: int = Field(1024, description="Number of patches (FPS centers) to sample")
    k_neighbors: int = Field(32, description="Number of neighbors per patch (KNN)")

class DataConfig(BaseModel):
    val_every: int = Field(50, description="Every n-th scene (sorted by name) is held out for validation")
    image_size: tuple[int, int] = Field((448, 448), description="Rendered view (height, width); multiples of the DINO patch size")
    num_candidates: int = Field(24, description="Candidate cameras proposed per scene before quality filtering")
    views_per_scene: int = Field(4, description="Views kept per scene")
    scenes_per_batch: int = Field(3, description="Scenes per batch; batch size in views is scenes_per_batch * views_per_scene")
    hfov_range_deg: tuple[float, float] = Field((70.0, 100.0), description="Horizontal field of view range")
    height_range: tuple[float, float] = Field((1.2, 1.8), description="Camera height above the floor (m)")
    pitch_range_deg: tuple[float, float] = Field((-25.0, 10.0), description="Camera pitch range (up positive)")
    max_splats_per_view: int = Field(150_000, description="Cap on frustum-cropped splats fed to the model per view")
    max_depth: float = Field(12.0, description="Frustum far plane (m)")
    min_opacity: float = Field(0.005, description="Splats below this opacity are dropped at load time")
    num_workers: int = Field(24, description="DataLoader workers loading scenes")

class TeacherConfig(BaseModel):
    checkpoint: str = Field("facebook/dinov3-vitl16-pretrain-lvd1689m", description="HF DINOv3 checkpoint")
    cache_dir: Optional[str] = Field(None, description="HF cache dir (defaults to HF_HOME)")
    local_files_only: bool = Field(True, description="Load the teacher from the local HF cache only")

class SplatFormerConfig(BaseModel):
    grid_resolution: int = Field(384, description="Voxels per unit of the min-max normalised view (SplatFormer)")
    enc_dim: int = Field(64, description="PTv3 encoder width preset (32 or 64)")
    backbone_dim: int = Field(96, description="PTv3 decoder output width preset (64, 96 or 128)")
    stride: tuple[int, ...] = Field((1, 2, 2, 2), description="Serialized pooling strides between encoder stages")
    patch_size: int = Field(1024, description="Splats per serialized attention patch")
    drop_path: float = Field(0.3, description="Maximum stochastic depth rate")
    head_nlayer: int = Field(4, description="Layers of the per-splat ReLU MLP head")
    head_width: int = Field(128, description="Width of the per-splat head")
    grad_checkpoint: bool = Field(True, description="Recompute PTv3 blocks in backward (needed for ~1.8M splats/batch)")

class ModelConfig(BaseModel):
    arch: Literal["splat_transformer", "splatformer"] = Field(
        "splat_transformer", description="Splat encoder: our SplatTransformer or the SplatFormer (PTv3) baseline"
    )
    splatformer: SplatFormerConfig = Field(default_factory=SplatFormerConfig, description="Used when arch=splatformer")
    num_groups: int = Field(1024, description="FPS group centres (transformer tokens) per view")
    fps_subset: int = Field(16384, description="Splats subsampled per view before running FPS")
    splat_dim: int = Field(128, description="Per-splat point feature width")
    token_dim: int = Field(384, description="Transformer width")
    depth: int = Field(8, description="Transformer blocks")
    num_heads: int = Field(6, description="Attention heads")
    out_dim: int = Field(128, description="Per-splat embedding width (rendered channels)")
    k_interp: int = Field(3, description="Group tokens interpolated into each splat")

class LossConfig(BaseModel):
    render_weight: float = Field(1.0, description="Weight of the rendered-embedding alignment loss")
    point_weight: float = Field(0.25, description="Weight of the per-visible-splat alignment loss")
    render_downsample: int = Field(4, description="Embedding maps are rendered at 1/n of the image resolution")
    min_alpha: float = Field(0.5, description="DINO patches with lower rendered coverage are ignored")
    point_max_splats: int = Field(65536, description="Visible splats sampled per batch for the point loss")
    rotate_inputs: bool = Field(True, description="Randomly rotate model inputs about z (targets are unaffected)")

class TrainConfig(BaseModel):
    device: str = Field("cuda", description="Training device")
    seed: int = Field(0, description="Random seed")
    log_every: int = Field(25, description="Steps between metric logs")
    val_every: int = Field(1000, description="Steps between validation passes")
    val_scenes: int = Field(20, description="Validation scenes evaluated per pass")
    ckpt_every: int = Field(2000, description="Steps between checkpoints")
    resume: Optional[str] = Field(None, description="Checkpoint to resume from")

class AppConfig(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    paths: PathsConfig
    hparams: HParamsConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
