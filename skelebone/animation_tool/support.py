"""Bounded numerical helpers for pose support analysis in the animation tool."""

import numpy as np

# Bounded query batch: keeps the [batch, source, joints] angle tensor small.
_QUERY_BATCH_SIZE = 32


def _geodesic_degrees(first, second):
    """Geodesic SO(3) angle in degrees; sign-invariant and robust near 0 and pi."""
    dot = np.clip(np.abs(np.einsum("...i,...i->...", first, second)), 0.0, 1.0)
    return np.degrees(2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot))


def _unit_rotations(rotations, name):
    """Validate and normalize scalar-first quaternions of shape [frames, joints, 4]."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError(f"{name} must have shape [frames, joints, 4]")
    if not np.isfinite(rotations).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(rotations, axis=-1)
    if (norms == 0.0).any():
        raise ValueError(f"{name} must not contain zero quaternions")
    return rotations / norms[..., None]


def pose_support_angles(source_rotations, query_rotations, part_masks):
    """Per-part support angle in degrees of each query pose against a source library.

    For each query frame and part, the distance is the minimum over source frames of
    the maximum over the part's masked joints of the SO(3) geodesic angle, so a single
    extreme joint is not diluted and each part may draw on a different source frame.
    Source and query are scalar-first quaternions of shape [frames, joints, 4];
    part_masks is boolean of shape [parts, joints]. Returns float [query_frames, parts].
    """
    source = _unit_rotations(source_rotations, "source_rotations")
    query = _unit_rotations(query_rotations, "query_rotations")
    masks = np.asarray(part_masks)
    if source.shape[0] == 0:
        raise ValueError("source_rotations must contain at least one frame")
    if source.shape[1] != query.shape[1]:
        raise ValueError("source and query must share the joint count")
    if masks.ndim != 2 or masks.dtype != bool or masks.shape[1] != source.shape[1]:
        raise ValueError("part_masks must be a boolean array of shape [parts, joints]")
    if masks.shape[0] == 0 or not masks.any(axis=1).all():
        raise ValueError("part_masks must contain parts, each selecting at least one joint")
    result = np.empty((query.shape[0], masks.shape[0]), dtype=np.float64)
    for start in range(0, query.shape[0], _QUERY_BATCH_SIZE):
        angles = _geodesic_degrees(query[start:start + _QUERY_BATCH_SIZE, None], source[None])
        for part, mask in enumerate(masks):
            result[start:start + _QUERY_BATCH_SIZE, part] = angles[:, :, mask].max(axis=-1).min(axis=-1)
    return result
