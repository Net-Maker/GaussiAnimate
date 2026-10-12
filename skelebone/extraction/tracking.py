"""Temporally track inner joints through a registered point sequence."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..kinematics import fit_pose_to_joint_positions
from ..types import DynamicPointSequence, PoseSequence, Skeleton
from ._geometry import as_points, nearest_neighbors


@dataclass(frozen=True)
class TrackingConfig:
    rest_frame: int = 0
    neighbors: int = 20

    def __post_init__(self) -> None:
        if self.rest_frame < 0 or self.neighbors < 1:
            raise ValueError("rest_frame must be non-negative and neighbors positive")


@dataclass
class JointTrajectories:
    positions: np.ndarray
    rest_positions: np.ndarray
    neighbor_indices: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.rest_positions = as_points(self.rest_positions, "rest_positions")
        self.neighbor_indices = np.asarray(self.neighbor_indices, dtype=np.int64)
        if self.positions.ndim != 3 or self.positions.shape[1:] != self.rest_positions.shape:
            raise ValueError("positions must have shape [frames, joints, 3]")
        if self.neighbor_indices.ndim != 2 or self.neighbor_indices.shape[0] != len(self.rest_positions):
            raise ValueError("neighbor_indices must have shape [joints, neighbors]")
        if self.fps <= 0:
            raise ValueError("fps must be positive")


@dataclass
class TrackedSkeleton:
    trajectories: JointTrajectories
    motion: PoseSequence


def _proper_rotation(source_centered: np.ndarray, target_centered: np.ndarray) -> np.ndarray:
    covariance = source_centered.T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1] *= -1
        rotation = right_t.T @ left.T
    return rotation


def track_joints(
    sequence: DynamicPointSequence,
    rest_joints,
    config: TrackingConfig = TrackingConfig(),
    reference_points: Optional[np.ndarray] = None,
) -> JointTrajectories:
    """Track medial joints by local rigid fitting around registered surface points."""

    if not isinstance(sequence, DynamicPointSequence):
        sequence = DynamicPointSequence(sequence)
    if config.rest_frame >= sequence.num_frames:
        raise ValueError("rest_frame is outside the point sequence")
    rest_joints = as_points(rest_joints, "rest_joints")
    reference = sequence.points[config.rest_frame] if reference_points is None else as_points(
        reference_points, "reference_points"
    )
    if len(reference) != sequence.num_points:
        raise ValueError("reference_points must correspond to every registered sequence point")
    _, neighbors = nearest_neighbors(rest_joints, reference, k=config.neighbors)
    trajectories = np.empty((sequence.num_frames, len(rest_joints), 3), dtype=np.float64)

    for joint, indices in enumerate(neighbors):
        rest_patch = reference[indices]
        rest_centroid = rest_patch.mean(axis=0)
        centered = rest_patch - rest_centroid
        offset = rest_joints[joint] - rest_centroid
        for frame in range(sequence.num_frames):
            if frame == config.rest_frame:
                trajectories[frame, joint] = rest_joints[joint]
                continue
            current_patch = sequence.points[frame, indices]
            current_centroid = current_patch.mean(axis=0)
            rotation = _proper_rotation(centered, current_patch - current_centroid)
            trajectories[frame, joint] = current_centroid + rotation @ offset
    return JointTrajectories(trajectories, rest_joints, neighbors, sequence.fps)


def track_skeleton(
    sequence: DynamicPointSequence,
    skeleton: Skeleton,
    config: TrackingConfig = TrackingConfig(),
    reference_points: Optional[np.ndarray] = None,
) -> TrackedSkeleton:
    trajectories = track_joints(sequence, skeleton.rest_joints, config, reference_points)
    motion = fit_pose_to_joint_positions(skeleton, trajectories.positions, fps=sequence.fps)
    return TrackedSkeleton(trajectories, motion)
