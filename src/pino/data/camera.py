"""Camera sampling for rendering views inside InteriorGS rooms.

Cameras use the OpenCV convention expected by gsplat (x right, y down, z forward) and
`viewmat` is the world-to-camera transform.
"""
import math

import numpy as np
import torch

from pino.data.scene import points_in_polygon, polygon_area


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def viewmat_from_yaw_pitch(position: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """World-to-camera matrix for a Z-up world, yaw about +z (0 = looking along +x), pitch up positive."""
    forward = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w_rot = np.stack([right, down, forward], axis=1)
    viewmat = np.eye(4)
    viewmat[:3, :3] = c2w_rot.T
    viewmat[:3, 3] = -c2w_rot.T @ position
    return viewmat.astype(np.float32)


def sample_floor_points(
    rooms: list[np.ndarray], num: int, rng: np.random.Generator, fallback_bounds: np.ndarray
) -> np.ndarray:
    """Uniformly samples (num, 2) xy points inside the room polygons (area weighted).

    `fallback_bounds` ((2, 2) xy min/max) is used when the scene has no usable room layout.
    """
    if not rooms:
        return rng.uniform(fallback_bounds[0], fallback_bounds[1], size=(num, 2))
    areas = np.array([polygon_area(r) for r in rooms])
    room_ids = rng.choice(len(rooms), size=num, p=areas / areas.sum())
    points = np.empty((num, 2))
    for room_id in np.unique(room_ids):
        poly = rooms[room_id]
        slots = np.flatnonzero(room_ids == room_id)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        found = np.empty((0, 2))
        for _ in range(50):
            cand = rng.uniform(lo, hi, size=(4 * len(slots) + 16, 2))
            found = np.concatenate([found, cand[points_in_polygon(cand, poly)]])
            if len(found) >= len(slots):
                break
        if len(found) < len(slots):  # degenerate polygon, fall back to its vertices' centroid
            found = np.concatenate([found, np.repeat(poly.mean(axis=0, keepdims=True), len(slots), axis=0)])
        points[slots] = found[: len(slots)]
    return points


def sample_cameras(
    rooms: list[np.ndarray],
    num: int,
    rng: np.random.Generator,
    fallback_bounds: np.ndarray,
    height_range: tuple[float, float],
    pitch_range_deg: tuple[float, float],
) -> torch.Tensor:
    """Returns (num, 4, 4) world-to-camera matrices at random positions inside the rooms."""
    xy = sample_floor_points(rooms, num, rng, fallback_bounds)
    z = rng.uniform(*height_range, size=num)
    yaws = rng.uniform(0.0, 2.0 * math.pi, size=num)
    pitches = np.radians(rng.uniform(*pitch_range_deg, size=num))
    viewmats = [
        viewmat_from_yaw_pitch(np.array([xy[i, 0], xy[i, 1], z[i]]), yaws[i], pitches[i]) for i in range(num)
    ]
    return torch.from_numpy(np.stack(viewmats))
