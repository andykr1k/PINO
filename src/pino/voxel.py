import numpy as np
from pydantic import BaseModel, ConfigDict, Field

class Voxel(BaseModel):
    """
    Represents a 3D voxel in space containing Gaussian splats.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid_index: tuple[int, int, int]
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    splat_indices: list[int] = Field(default_factory=list)

    @property
    def center(self) -> np.ndarray:
        """Returns the physical center coordinate of the voxel."""
        return (self.min_bounds + self.max_bounds) / 2.0
    
    @property
    def size(self) -> np.ndarray:
        """Returns the dimensions (width, height, depth) of the voxel."""
        return self.max_bounds - self.min_bounds

    def add_splat(self, idx: int):
        self.splat_indices.append(idx)
