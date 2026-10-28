"""CPU-side dataset: loads InteriorGS scenes and proposes candidate cameras inside their rooms.

Rendering, view filtering, DINO feature extraction and frustum cropping are GPU work and
happen in `pino.data.views.ViewGenerator`, which consumes the samples produced here.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pino.data.camera import intrinsics_from_hfov, sample_cameras
from pino.data.scene import GaussianScene, list_scene_dirs, load_scene


@dataclass
class SceneSample:
    scene: GaussianScene
    viewmats: torch.Tensor  # (C, 4, 4) candidate world-to-camera matrices
    Ks: torch.Tensor  # (C, 3, 3) candidate intrinsics

    def pin_memory(self) -> "SceneSample":
        return SceneSample(self.scene.pin_memory(), self.viewmats.pin_memory(), self.Ks.pin_memory())


def split_scenes(root: Path, split: str, val_every: int) -> list[Path]:
    """Deterministic split: every `val_every`-th scene (sorted by name) is held out for validation."""
    scenes = list_scene_dirs(root)
    if split == "train":
        return [s for i, s in enumerate(scenes) if i % val_every != 0]
    if split == "val":
        return [s for i, s in enumerate(scenes) if i % val_every == 0]
    raise ValueError(f"Unknown split {split}")


class InteriorGSScenes(Dataset):
    def __init__(
        self,
        scene_dirs: list[Path],
        num_candidates: int,
        image_size: tuple[int, int],
        hfov_range_deg: tuple[float, float],
        height_range: tuple[float, float],
        pitch_range_deg: tuple[float, float],
        min_opacity: float = 0.005,
        seed: int | None = None,
    ):
        """
        Args:
            num_candidates: cameras proposed per scene; the GPU stage keeps the good ones.
            image_size: (height, width) of the rendered views.
            seed: if set, scene `i` always gets the same cameras (used for validation).
        """
        self.scene_dirs = scene_dirs
        self.num_candidates = num_candidates
        self.image_size = image_size
        self.hfov_range_deg = hfov_range_deg
        self.height_range = height_range
        self.pitch_range_deg = pitch_range_deg
        self.min_opacity = min_opacity
        self.seed = seed

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> SceneSample:
        rng = np.random.default_rng(None if self.seed is None else (self.seed, idx))
        try:
            scene = load_scene(self.scene_dirs[idx], min_opacity=self.min_opacity)
        except (OSError, ValueError) as e:
            if self.seed is not None:
                raise
            print(f"Skipping unreadable scene {self.scene_dirs[idx]}: {e}")
            return self[int(rng.integers(len(self)))]
        means = scene.means.numpy()
        fallback_bounds = np.percentile(means[:, :2], [5, 95], axis=0)
        viewmats = sample_cameras(
            scene.rooms,
            self.num_candidates,
            rng,
            fallback_bounds,
            self.height_range,
            self.pitch_range_deg,
        )
        height, width = self.image_size
        hfovs = rng.uniform(*self.hfov_range_deg, size=self.num_candidates)
        Ks = torch.from_numpy(np.stack([intrinsics_from_hfov(width, height, h) for h in hfovs]))
        return SceneSample(scene, viewmats, Ks)


def collate_scene_samples(samples: list[SceneSample]) -> list[SceneSample]:
    return samples
