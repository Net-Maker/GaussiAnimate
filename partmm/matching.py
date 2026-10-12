"""Quaternion patch distances and deterministic PartMM neighbor selection."""

from typing import Tuple

import numpy as np

from .database import build_motion_patches, motion_patch_starts


def patch_distance(query: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Mean squared SO(3) geodesic distance from one query to K patches."""
    query = np.asarray(query, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    if query.ndim != 3 or query.shape[-1] != 4:
        raise ValueError("query must have shape [patch, joints, 4]")
    if source.ndim != 4 or source.shape[1:] != query.shape:
        raise ValueError(
            "source must have shape [candidates, patch, joints, 4]"
        )
    query_norm = np.linalg.norm(query, axis=-1, keepdims=True)
    source_norm = np.linalg.norm(source, axis=-1, keepdims=True)
    if np.any(query_norm < 1e-12) or np.any(source_norm < 1e-12):
        raise ValueError("rotation patches contain a zero quaternion")
    query = query / query_norm
    source = source / source_norm
    dots = np.clip(
        np.abs(np.einsum("pjc,kpjc->kpj", query, source)),
        0.0,
        1.0,
    )
    # atan2 is better conditioned than acos for nearly identical rotations.
    angles = 2.0 * np.arctan2(
        np.sqrt(np.maximum(0.0, 1.0 - dots * dots)),
        dots,
    )
    return np.mean(angles * angles, axis=(1, 2))


def select_patch_neighbors(
    query_patch: np.ndarray,
    source_patches: np.ndarray,
    source_starts: np.ndarray,
    num_neighbors: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select deterministic KNN starts and their normalized blend weights.

    Returns the selected starts, a fixed-width padded start row for reporting,
    and one weight per selected candidate.
    """
    distance = patch_distance(query_patch, source_patches)
    neighbor_count = min(num_neighbors, len(distance))
    neighbor_indices = np.lexsort((source_starts, distance))[
        :neighbor_count
    ]
    selected_starts = source_starts[neighbor_indices]
    reported_starts = np.empty(num_neighbors, dtype=np.int64)
    reported_starts[:neighbor_count] = selected_starts
    if neighbor_count < num_neighbors:
        reported_starts[neighbor_count:] = selected_starts[-1]
    logits = (
        -(distance[neighbor_indices] - distance[neighbor_indices].min())
        / temperature
    )
    blend_weights = np.exp(logits - logits.max())
    blend_weights /= blend_weights.sum()
    return selected_starts, reported_starts, blend_weights


def frame_match_distances(
    source_rotations: np.ndarray,
    query_rotations: np.ndarray,
    part_masks: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """Absolute nearest-patch distances, averaged over covering windows.

    Unlike normalized KNN weights, these retain how far the best available
    example is from the query. Units are mean squared SO(3) radians.
    """
    source = np.asarray(source_rotations, dtype=np.float64)
    query = np.asarray(query_rotations, dtype=np.float64)
    masks = np.asarray(part_masks, dtype=bool)
    if (
        source.ndim != 3 or query.ndim != 3
        or source.shape[1:] != query.shape[1:] or source.shape[-1] != 4
        or min(len(source), len(query), patch_size) < 1
        or masks.ndim != 2 or masks.shape[1] != source.shape[1]
        or len(masks) == 0 or not masks.any(axis=1).all()
    ):
        raise ValueError("nonempty rotation sequences and joint part masks are required")
    patch = min(patch_size, len(source), len(query))
    source_starts = motion_patch_starts(len(source), patch)
    query_starts = motion_patch_starts(len(query), patch)
    distances = np.zeros((len(query), len(masks)), dtype=np.float64)
    counts = np.zeros(len(query), dtype=np.int64)
    for part, mask in enumerate(masks):
        database = build_motion_patches(source, source_starts, patch, mask)
        for start in query_starts:
            nearest = patch_distance(query[start:start + patch, mask], database).min()
            distances[start:start + patch, part] += nearest
            if part == 0:
                counts[start:start + patch] += 1
    return distances / counts[:, None]


__all__ = ["frame_match_distances", "patch_distance", "select_patch_neighbors"]
