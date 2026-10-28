"""Loading InteriorGS scenes (uncompressed 3DGS PLY + room layout) as torch tensors."""
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

SH_C0 = 0.28209479177387814
PLY_NAME = "3dgs_uncompressed.ply"


@dataclass
class GaussianScene:
    """A full Gaussian splat scene. World frame is Z-up, metres, floor at z ~= 0."""

    name: str
    means: torch.Tensor  # (N, 3) float32
    quats: torch.Tensor  # (N, 4) float32, wxyz, unit norm, w >= 0
    log_scales: torch.Tensor  # (N, 3) float32
    opacity_logits: torch.Tensor  # (N,) float32
    sh: torch.Tensor  # (N, 16, 3) float16, degree-3 SH coefficients (DC first)
    rooms: list[np.ndarray]  # room floor polygons, each (M, 2) in world xy

    @property
    def num_splats(self) -> int:
        return self.means.shape[0]

    @property
    def rgb(self) -> torch.Tensor:
        """View-independent colour from the SH DC term, (N, 3) in [0, 1]."""
        return (self.sh[:, 0].float() * SH_C0 + 0.5).clamp(0.0, 1.0)

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "GaussianScene":
        moved = {
            f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        }
        return replace(self, **moved)

    def pin_memory(self) -> "GaussianScene":
        pinned = {
            f.name: getattr(self, f.name).pin_memory()
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        }
        return replace(self, **pinned)


def _read_ply_header(f) -> tuple[int, list[str]]:
    num_vertices = -1
    props: list[str] = []
    line = f.readline().decode().strip()
    if line != "ply":
        raise ValueError("Not a PLY file")
    while True:
        line = f.readline().decode().strip()
        if line.startswith("format") and "binary_little_endian" not in line:
            raise ValueError(f"Unsupported PLY format: {line}")
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[-1])
        elif line.startswith("element"):
            raise ValueError(f"Unexpected PLY element: {line}")
        elif line.startswith("property"):
            dtype, name = line.split()[1:3]
            if dtype != "float":
                raise ValueError(f"Unsupported property type: {line}")
            props.append(name)
        elif line == "end_header":
            return num_vertices, props


def load_ply_gaussians(ply_path: Path, min_opacity: float = 0.0) -> dict[str, torch.Tensor]:
    """Reads a binary 3DGS PLY (all-float properties) straight into tensors.

    Much faster than plyfile for these files since the vertex block is one flat float32 array.
    """
    with open(ply_path, "rb") as f:
        num_vertices, props = _read_ply_header(f)
        data = np.fromfile(f, dtype=np.float32, count=num_vertices * len(props))
    data = data.reshape(num_vertices, len(props))
    col = {name: i for i, name in enumerate(props)}

    def take(*names: str) -> np.ndarray:
        return data[:, [col[n] for n in names]]

    means = take("x", "y", "z")
    dc = take("f_dc_0", "f_dc_1", "f_dc_2")
    rest_names = sorted((n for n in props if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
    # f_rest is stored channel-major: [R_1..R_15, G_1..G_15, B_1..B_15]
    rest = take(*rest_names).reshape(num_vertices, 3, -1).transpose(0, 2, 1)
    sh = np.concatenate([dc[:, None, :], rest], axis=1)
    quats = take("rot_0", "rot_1", "rot_2", "rot_3")
    log_scales = take("scale_0", "scale_1", "scale_2")
    opacity_logits = data[:, col["opacity"]]

    keep = np.isfinite(data).all(axis=1)
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    keep &= quat_norm[:, 0] > 1e-8
    if min_opacity > 0.0:
        keep &= opacity_logits > np.log(min_opacity / (1.0 - min_opacity))

    quats = quats / np.maximum(quat_norm, 1e-8)
    # q and -q are the same rotation; canonicalise so features are unambiguous
    quats = quats * np.where(quats[:, :1] < 0, -1.0, 1.0).astype(np.float32)

    return {
        "means": torch.from_numpy(np.ascontiguousarray(means[keep])),
        "quats": torch.from_numpy(np.ascontiguousarray(quats[keep])),
        "log_scales": torch.from_numpy(np.ascontiguousarray(log_scales[keep])),
        "opacity_logits": torch.from_numpy(np.ascontiguousarray(opacity_logits[keep])),
        "sh": torch.from_numpy(np.ascontiguousarray(sh[keep])).to(torch.float16),
    }


def load_room_polygons(scene_dir: Path) -> list[np.ndarray]:
    structure_path = scene_dir / "structure.json"
    if not structure_path.exists():
        return []
    structure = json.loads(structure_path.read_text())
    rooms = []
    for room in structure.get("rooms", []):
        poly = np.asarray(room.get("profile", []), dtype=np.float64)
        if poly.ndim == 2 and poly.shape[0] >= 3 and polygon_area(poly) > 0.5:
            rooms.append(poly[:, :2])
    return rooms


def load_scene(scene_dir: Path, min_opacity: float = 0.0) -> GaussianScene:
    gaussians = load_ply_gaussians(scene_dir / PLY_NAME, min_opacity=min_opacity)
    return GaussianScene(name=scene_dir.name, rooms=load_room_polygons(scene_dir), **gaussians)


def list_scene_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if (p / PLY_NAME).exists())


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_in_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting. points: (P, 2), poly: (M, 2) -> (P,) bool."""
    x, y = points[:, 0:1], points[:, 1:2]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return ((crosses & (x < x_int)).sum(axis=1) % 2) == 1
