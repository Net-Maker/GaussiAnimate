"""Scientific visualization for the Skelebone representation."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraView:
    """A renderer-independent camera using the HOOD Y-up convention."""

    eye: np.ndarray
    center: np.ndarray
    up: np.ndarray
    fov_degrees: float
    distance: float


def compute_y_up_camera(
    points: np.ndarray,
    *,
    azimuth_degrees: float = 12.0,
    elevation_degrees: float = 3.0,
    fov_degrees: float = 30.0,
    fit_scale: float = 1.18,
) -> CameraView:
    """Fit a slightly three-quarter front camera to a Y-up dynamic asset."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have shape [N,3] and contain at least one point")
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    extent = max(float(np.max(high - low)), 1e-6)
    distance = extent / (2.0 * math.tan(math.radians(fov_degrees) * 0.5)) * fit_scale
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    direction = np.array(
        [
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
            math.cos(azimuth) * math.cos(elevation),
        ],
        dtype=np.float64,
    )
    return CameraView(
        eye=center + direction * distance,
        center=center,
        up=np.array([0.0, 1.0, 0.0]),
        fov_degrees=float(fov_degrees),
        distance=float(distance),
    )


compute_hood_camera = compute_y_up_camera


def parent_edges(parents: np.ndarray) -> np.ndarray:
    """Convert a parent array into Open3D-compatible undirected edges."""
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    return np.asarray(
        [[int(parent), child] for child, parent in enumerate(parents) if parent >= 0],
        dtype=np.int64,
    ).reshape(-1, 2)


def save_skelebone_figure(
    points,
    curve_points,
    curve_edges,
    joints,
    parents,
    output_path,
    *,
    faces=None,
    width=760,
    height=760,
    azimuth_degrees=12.0,
    elevation_degrees=3.0,
) -> Path:
    """Render the observation-to-rig abstraction with Open3D."""
    points = np.asarray(points)
    curve_points = np.asarray(curve_points)
    curve_edges = np.asarray(curve_edges, dtype=np.int64)
    joints = np.asarray(joints)
    parents = np.asarray(parents, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape [F,3]")
    camera = compute_y_up_camera(
        points,
        azimuth_degrees=azimuth_degrees,
        elevation_degrees=elevation_degrees,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    from .open3d_renderer import render_skelebone_preview

    return render_skelebone_preview(
        points=points,
        faces=faces,
        curve_points=curve_points,
        curve_edges=curve_edges,
        joints=joints,
        joint_edges=parent_edges(parents),
        camera=camera,
        output_path=output,
        width=int(width),
        height=int(height),
    )
