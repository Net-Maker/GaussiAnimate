"""Small NumPy geometry primitives shared by extraction stages."""

from typing import Tuple

import numpy as np


def as_points(value, name: str = "points") -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [count, 3]")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return points


def as_edges(value, num_points: int, name: str = "edges") -> np.ndarray:
    edges = np.asarray(value, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{name} must have shape [count, 2]")
    if edges.min() < 0 or edges.max() >= num_points:
        raise ValueError(f"{name} contains an out-of-range point index")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError(f"{name} contains a self edge")
    return edges


def normalize_rows(values, name: str = "weights") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [count, channels]")
    if not np.all(np.isfinite(array)) or np.any(array < -1e-10):
        raise ValueError(f"{name} must be finite and non-negative")
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=1, keepdims=True)
    if np.any(totals <= 1e-12):
        raise ValueError(f"each row of {name} must have positive mass")
    return array / totals


def nearest_neighbors(
    queries,
    reference,
    k: int = 1,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Euclidean KNN without requiring SciPy.

    Chunking bounds the temporary distance matrix, which is important for the
    dense point sets used by GaussiAnimate.
    """

    queries = as_points(queries, "queries")
    reference = as_points(reference, "reference")
    if len(reference) == 0:
        raise ValueError("reference must not be empty")
    if k < 1:
        raise ValueError("k must be positive")
    k = min(int(k), len(reference))
    chunk_size = max(1, int(chunk_size))
    all_indices = np.empty((len(queries), k), dtype=np.int64)
    all_distances = np.empty((len(queries), k), dtype=np.float64)

    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        delta = queries[start:stop, None, :] - reference[None, :, :]
        squared = np.einsum("qri,qri->qr", delta, delta)
        if k == len(reference):
            indices = np.argsort(squared, axis=1)[:, :k]
        else:
            indices = np.argpartition(squared, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(squared, indices, axis=1), axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
        selected = np.take_along_axis(squared, indices, axis=1)
        all_indices[start:stop] = indices
        all_distances[start:stop] = np.sqrt(np.maximum(selected, 0.0))

    return all_distances, all_indices
