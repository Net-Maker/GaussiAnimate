"""Curve-skeleton extraction and skinning transfer."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import subprocess

import numpy as np

from ._geometry import as_edges, as_points, nearest_neighbors, normalize_rows
from .mesh import TriangleMesh, write_off


@dataclass
class CurveSkeleton:
    points: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        self.points = as_points(self.points, "curve points")
        self.edges = as_edges(self.edges, len(self.points), "curve edges")


@dataclass
class Correspondence:
    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.distances = np.asarray(self.distances, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        if self.indices.ndim != 2 or self.indices.shape != self.distances.shape:
            raise ValueError("correspondence indices and distances must have shape [points, k]")
        if self.weights.shape != self.indices.shape:
            raise ValueError("correspondence weights must match indices")
        if np.any(self.indices < 0) or np.any(self.distances < 0):
            raise ValueError("correspondence indices and distances must be non-negative")
        self.weights = normalize_rows(self.weights, "correspondence weights")


@dataclass(frozen=True)
class CurveExtractionConfig:
    executable: str
    quality: float = 0.1
    medial: float = 0.25
    timeout_seconds: float = 300.0
    correspondence_neighbors: int = 3

    def __post_init__(self) -> None:
        if not self.executable:
            raise ValueError("a curve-skeleton executable path is required")
        if not 0 <= self.quality <= 1 or not 0 <= self.medial <= 1:
            raise ValueError("quality and medial must be in [0, 1]")
        if self.timeout_seconds <= 0 or self.correspondence_neighbors < 1:
            raise ValueError("timeout and correspondence_neighbors must be positive")


@dataclass
class CurveExtractionResult:
    skeleton: CurveSkeleton
    correspondence: Correspondence
    mesh_path: Path
    points_path: Path
    edges_path: Path


def _inverse_distance_weights(distances: np.ndarray) -> np.ndarray:
    weights = np.empty_like(distances, dtype=np.float64)
    exact = distances <= 1e-12
    exact_rows = exact.any(axis=1)
    if np.any(exact_rows):
        weights[exact_rows] = exact[exact_rows] / exact[exact_rows].sum(axis=1, keepdims=True)
    if np.any(~exact_rows):
        inverse = 1.0 / distances[~exact_rows]
        weights[~exact_rows] = inverse / inverse.sum(axis=1, keepdims=True)
    return weights


def vertex_curve_correspondence(vertices, curve_points, k: int = 3) -> Correspondence:
    distances, indices = nearest_neighbors(vertices, curve_points, k=k)
    return Correspondence(indices, distances, _inverse_distance_weights(distances))


def load_curve_points(path: Union[str, Path]) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="ascii") as stream:
        for line in stream:
            line = line.partition("#")[0].strip()
            if line:
                values = line.split()
                if len(values) >= 3:
                    rows.append([float(value) for value in values[:3]])
    return as_points(np.asarray(rows, dtype=np.float64), "curve points")


def load_curve_edges(path: Union[str, Path], num_points: Optional[int] = None) -> np.ndarray:
    rows = []
    with Path(path).open("r", encoding="ascii") as stream:
        for line in stream:
            line = line.partition("#")[0].strip()
            if line:
                values = line.split()
                if len(values) >= 2:
                    rows.append([int(value) for value in values[:2]])
    edges = np.asarray(rows, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    if num_points is not None:
        return as_edges(edges, num_points, "curve edges")
    if edges.ndim != 2 or edges.shape[1] != 2 or np.any(edges < 0):
        raise ValueError("curve edges must have shape [count, 2]")
    return edges


def extract_curve_skeleton(
    mesh: TriangleMesh,
    output_dir: Union[str, Path],
    config: CurveExtractionConfig,
) -> CurveExtractionResult:
    """Run the explicit CGAL/QMAT boundary without project-global path state."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = Path(config.executable).expanduser()
    if not executable.is_file():
        raise FileNotFoundError(f"curve-skeleton executable not found: {executable}")
    mesh_path = write_off(output_dir / "mesh.off", mesh)
    prefix = output_dir / "curve_skeleton"
    command = [
        str(executable), str(mesh_path), str(prefix),
        "--quality", str(config.quality), "--medial", str(config.medial),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=config.timeout_seconds, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"curve-skeleton extraction failed ({completed.returncode}): {detail}")
    points_path = Path(str(prefix) + "_points.txt")
    edges_path = Path(str(prefix) + "_edges.txt")
    if not points_path.is_file() or not edges_path.is_file():
        raise RuntimeError("curve-skeleton executable did not produce point and edge files")
    points = load_curve_points(points_path)
    skeleton = CurveSkeleton(points, load_curve_edges(edges_path, num_points=len(points)))
    correspondence = vertex_curve_correspondence(
        mesh.vertices, skeleton.points, config.correspondence_neighbors
    )
    np.savez(
        output_dir / "vertex_curve_correspondence.npz",
        indices=correspondence.indices,
        distances=correspondence.distances,
        weights=correspondence.weights,
    )
    return CurveExtractionResult(skeleton, correspondence, mesh_path, points_path, edges_path)


def align_points_to_mesh_bbox(points, mesh_vertices, ratio_bounds=(0.5, 2.0)):
    """Apply the legacy bbox fallback only when the two spaces clearly disagree."""

    points = as_points(points)
    mesh_vertices = as_points(mesh_vertices, "mesh_vertices")
    point_min, point_max = points.min(axis=0), points.max(axis=0)
    mesh_min, mesh_max = mesh_vertices.min(axis=0), mesh_vertices.max(axis=0)
    point_extent = float(np.ptp(points, axis=0).max())
    mesh_extent = float(np.ptp(mesh_vertices, axis=0).max())
    ratio = mesh_extent / max(point_extent, 1e-12)
    if ratio_bounds[0] <= ratio <= ratio_bounds[1]:
        return points.copy(), np.ones(3), np.zeros(3)
    point_range = point_max - point_min
    scale = np.divide(
        mesh_max - mesh_min,
        point_range,
        out=np.ones(3, dtype=np.float64),
        where=point_range > 1e-12,
    )
    offset = mesh_min - point_min * scale
    return points * scale + offset, scale, offset


def transfer_skinning_to_mesh(
    mesh_vertices,
    source_points,
    source_weights,
    k: int = 1,
    align_bbox: bool = False,
) -> np.ndarray:
    """Transfer outer-bone weights by hard assignment or inverse-distance KNN."""

    mesh_vertices = as_points(mesh_vertices, "mesh_vertices")
    source_points = as_points(source_points, "source_points")
    source_weights = np.asarray(source_weights, dtype=np.float64)
    if source_weights.ndim != 2:
        raise ValueError("source_weights must have shape [points, bones]")
    if source_weights.shape[0] != len(source_points) and source_weights.shape[1] == len(source_points):
        source_weights = source_weights.T
    if source_weights.shape[0] != len(source_points):
        raise ValueError("source point and weight counts differ")
    source_weights = normalize_rows(source_weights, "source_weights")
    if align_bbox:
        source_points, _, _ = align_points_to_mesh_bbox(source_points, mesh_vertices)
    distances, indices = nearest_neighbors(mesh_vertices, source_points, k=k)
    interpolation = _inverse_distance_weights(distances)
    transferred = np.einsum("nk,nkb->nb", interpolation, source_weights[indices])
    return normalize_rows(transferred)


def project_skinning_to_curve(
    mesh_weights,
    correspondence: Correspondence,
    curve: CurveSkeleton,
) -> np.ndarray:
    """Aggregate all mesh-to-curve links, then fill unsupported curve samples spatially."""

    mesh_weights = normalize_rows(mesh_weights, "mesh_weights")
    if len(mesh_weights) != len(correspondence.indices):
        raise ValueError("mesh weights and correspondence vertex counts differ")
    if correspondence.indices.size and correspondence.indices.max() >= len(curve.points):
        raise ValueError("correspondence references an out-of-range curve point")
    result = np.zeros((len(curve.points), mesh_weights.shape[1]), dtype=np.float64)
    mass = np.zeros(len(curve.points), dtype=np.float64)
    for neighbor in range(correspondence.indices.shape[1]):
        targets = correspondence.indices[:, neighbor]
        contribution = correspondence.weights[:, neighbor]
        np.add.at(result, targets, mesh_weights * contribution[:, None])
        np.add.at(mass, targets, contribution)
    supported = mass > 1e-12
    if not np.any(supported):
        raise ValueError("correspondence provides no support for the curve")
    result[supported] /= mass[supported, None]
    if np.any(~supported):
        _, nearest = nearest_neighbors(curve.points[~supported], curve.points[supported], k=1)
        result[~supported] = result[supported][nearest[:, 0]]
    return normalize_rows(result, "curve_weights")
