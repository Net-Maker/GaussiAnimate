"""Rigid-transform math and temporal resampling used by PartMM."""

import numpy as np

from .kinematics import (
    quaternion_to_matrix as _kinematics_quaternion_to_matrix,
)


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert proper 3x3 matrices to scalar-first quaternions."""
    flat = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    output = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = 2.0 * np.sqrt(
                    1.0
                    + rotation[0, 0]
                    - rotation[1, 1]
                    - rotation[2, 2]
                )
                quaternion = np.array(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = 2.0 * np.sqrt(
                    1.0
                    + rotation[1, 1]
                    - rotation[0, 0]
                    - rotation[2, 2]
                )
                quaternion = np.array(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(
                    1.0
                    + rotation[2, 2]
                    - rotation[0, 0]
                    - rotation[1, 1]
                )
                quaternion = np.array(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        quaternion /= np.linalg.norm(quaternion)
        output[index] = quaternion if quaternion[0] >= 0 else -quaternion
    return output.reshape(matrix.shape[:-2] + (4,))


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert scalar-first unit quaternions to rotation matrices."""
    return _kinematics_quaternion_to_matrix(quaternion)


def blend_transforms(
    transforms: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray = None,
) -> np.ndarray:
    """Average rotations and the images of optional canonical bone centers.

    A center keeps translation averaging independent of the canonical origin.
    With no centers, translations retain their historical linear average.
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if transforms.ndim < 3 or transforms.shape[-2:] != (4, 4):
        raise ValueError(
            "transforms must have shape [samples, ..., 4, 4]"
        )
    if weights.shape != (transforms.shape[0],):
        raise ValueError("weights must have one entry per transform sample")
    if (
        np.any(weights < 0)
        or not np.isfinite(weights).all()
        or weights.sum() <= 0
    ):
        raise ValueError(
            "weights must be finite, non-negative, and sum to a positive value"
        )
    weights = weights / weights.sum()
    positions = transforms[..., :3, 3]
    if centers is not None:
        centers = np.asarray(centers, dtype=np.float64)
        positions = positions + np.einsum("k...ij,...j->k...i", transforms[..., :3, :3], centers)
    translations = np.tensordot(
        weights,
        positions,
        axes=(0, 0),
    )
    quaternions = rotation_matrix_to_quaternion(
        transforms[..., :3, :3]
    )
    scatter = np.einsum(
        "k,k...i,k...j->...ij",
        weights,
        quaternions,
        quaternions,
    )
    _, eigenvectors = np.linalg.eigh(scatter)
    blended_q = eigenvectors[..., -1]
    blended_q = np.where(
        blended_q[..., :1] < 0,
        -blended_q,
        blended_q,
    )
    result = np.broadcast_to(
        np.eye(4),
        transforms.shape[1:-2] + (4, 4),
    ).copy()
    result[..., :3, :3] = quaternion_to_matrix(blended_q)
    if centers is not None:
        translations -= np.einsum("...ij,...j->...i", result[..., :3, :3], centers)
    result[..., :3, 3] = translations
    return result


def align_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> np.ndarray:
    """Return the proper rigid transform aligning two corresponding point sets."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if (
        source_points.ndim != 2
        or source_points.shape[1:] != (3,)
        or target_points.shape != source_points.shape
        or len(source_points) == 0
    ):
        raise ValueError(
            "source_points and target_points must have matching [N, 3] shapes"
        )
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (
        (source_points - source_center).T
        @ (target_points - target_center)
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    alignment = np.eye(4)
    alignment[:3, :3] = rotation
    alignment[:3, 3] = target_center - rotation @ source_center
    return alignment


def sample_indices(length: int, stride: int) -> np.ndarray:
    """Sample a temporal pyramid level while retaining the final frame."""
    indices = np.arange(0, length, stride, dtype=np.int64)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def quaternion_slerp(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Shortest-path interpolation for scalar-first unit quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    if dot > 1.0 - 1e-8:
        result = (1.0 - alpha[:, None]) * q0 + alpha[:, None] * q1
    else:
        angle = np.arccos(dot)
        scale = np.sin(angle)
        result = (
            np.sin((1.0 - alpha) * angle)[:, None] / scale * q0
            + np.sin(alpha * angle)[:, None] / scale * q1
        )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def resample_transforms(
    transforms: np.ndarray,
    sample_indices: np.ndarray,
    length: int,
    centers: np.ndarray = None,
) -> np.ndarray:
    """Interpolate an SE(3) sequence from pyramid samples to every frame."""
    if len(sample_indices) == length:
        return transforms
    target = np.arange(length)
    output = np.broadcast_to(
        np.eye(4),
        (length, transforms.shape[1], 4, 4),
    ).copy()
    positions = transforms[..., :3, 3]
    if centers is not None:
        centers = np.asarray(centers, dtype=np.float64)
        positions = positions + np.einsum("tbij,bj->tbi", transforms[..., :3, :3], centers)
    for bone in range(transforms.shape[1]):
        for axis in range(3):
            output[:, bone, axis, 3] = np.interp(
                target,
                sample_indices,
                positions[:, bone, axis],
            )
        quaternions = rotation_matrix_to_quaternion(
            transforms[:, bone, :3, :3]
        )
        for interval in range(len(sample_indices) - 1):
            start = int(sample_indices[interval])
            end = int(sample_indices[interval + 1])
            q0 = quaternions[interval]
            q1 = quaternions[interval + 1]
            if np.dot(q0, q1) < 0:
                q1 = -q1
            alpha = np.linspace(0.0, 1.0, end - start + 1)
            blended = quaternion_slerp(q0, q1, alpha)
            output[start : end + 1, bone, :3, :3] = (
                quaternion_to_matrix(blended)
            )
    if centers is not None:
        output[..., :3, 3] -= np.einsum("tbij,bj->tbi", output[..., :3, :3], centers)
    return output


__all__ = [
    "align_transform",
    "blend_transforms",
    "quaternion_slerp",
    "quaternion_to_matrix",
    "resample_transforms",
    "rotation_matrix_to_quaternion",
    "sample_indices",
]
