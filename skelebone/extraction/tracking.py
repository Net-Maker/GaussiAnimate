"""Temporally track inner joints through a registered point sequence."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..kinematics import fit_pose_to_joint_positions, forward_kinematics, refine_pose_to_joint_positions
from ..types import DynamicPointSequence, PoseSequence, Skeleton
from ._geometry import as_points, nearest_neighbors


@dataclass(frozen=True)
class TrackingConfig:
    rest_frame: int = 0
    neighbors: int = 20
    ik_iterations: int = 200
    ik_tolerance: float = 1e-8
    surface_constraints: bool = True

    def __post_init__(self) -> None:
        if self.rest_frame < 0 or self.neighbors < 1:
            raise ValueError("rest_frame must be non-negative and neighbors positive")
        if not isinstance(self.ik_iterations, (int, np.integer)) or self.ik_iterations < 0:
            raise ValueError("ik_iterations must be a non-negative integer")
        if not np.isfinite(self.ik_tolerance) or self.ik_tolerance <= 0:
            raise ValueError("ik_tolerance must be finite and positive")


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
    if not isinstance(sequence, DynamicPointSequence):
        sequence = DynamicPointSequence(sequence)
    trajectories = track_joints(sequence, skeleton.rest_joints, config, reference_points)
    motion = fit_pose_to_joint_positions(
        skeleton, trajectories.positions, fps=sequence.fps,
        max_iterations=config.ik_iterations, tolerance=config.ik_tolerance,
    )
    if config.surface_constraints:
        motion = refine_motion_with_surface(sequence, skeleton, trajectories.positions, motion)
    return TrackedSkeleton(trajectories, motion)


def refine_motion_with_surface(
    sequence: DynamicPointSequence,
    skeleton: Skeleton,
    tracked_positions: np.ndarray,
    motion: PoseSequence,
    *,
    iterations: int = 80,
    collision_weight: float = 1000.,
    margin: Optional[float] = None,
) -> PoseSequence:
    """Refine fixed-length IK against closed surfaces, including bone interiors."""
    faces = sequence.faces
    if faces is None or len(faces) < 4:
        return motion
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts != 2):
        return motion
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not np.isfinite(collision_weight) or collision_weight <= 0:
        raise ValueError("collision_weight must be finite and positive")
    extent = float(np.ptp(sequence.points[0], axis=0).max())
    margin = extent * .0015 if margin is None else float(margin)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    import open3d as o3d
    from ._surface_ik import SignedSurface, constrain_surface_frame

    # Virtual leaf joints sample each bone without changing the exported rig.
    children = np.flatnonzero(skeleton.parents >= 0)
    parents = skeleton.parents[children]
    fractions = np.array([.25, .5, .75])
    sample_children = np.repeat(children, len(fractions))
    sample_parents = np.repeat(parents, len(fractions))
    fraction = np.tile(fractions, len(children))[None, :, None]
    def with_samples(points):
        samples = points[..., sample_parents, :] * (1.-fraction) + points[..., sample_children, :] * fraction
        return np.concatenate([points, samples], axis=-2)

    extended = Skeleton(with_samples(skeleton.rest_joints[None])[0], np.concatenate([skeleton.parents, sample_parents]))
    rotations = np.zeros((motion.num_frames, extended.num_joints, 4))
    rotations[..., 0] = 1.
    rotations[:, :skeleton.num_joints] = motion.local_rotations
    current = PoseSequence(rotations, motion.root_translation.copy(), fps=motion.fps)
    target = with_samples(np.asarray(tracked_positions))
    goals = target.copy()
    weights = np.ones(target.shape[:2])
    scenes = []
    origins = sequence.points.mean(axis=1)
    for vertices, origin in zip(sequence.points, origins):
        scenes.append(SignedSurface(vertices-origin, faces))
    pending_frames = np.ones(motion.num_frames, dtype=bool)
    for _ in range(iterations):
        _, positions = forward_kinematics(extended, current)
        active_frames = []
        for frame, scene in enumerate(scenes):
            if not pending_frames[frame]:
                continue
            queries = o3d.core.Tensor((positions[frame]-origins[frame]).astype(np.float32))
            signed = scene.compute_signed_distance(queries, nsamples=3).numpy()
            if float(signed.max()) <= margin * .25:
                pending_frames[frame] = False
                continue
            active_frames.append(frame)
            active = signed > -margin
            if not np.any(active):
                continue
            closest = scene.compute_closest_points(queries)
            surface = closest["points"].numpy().astype(np.float64) + origins[frame]
            normal = closest["primitive_normals"].numpy()
            stable = np.abs(signed) > extent * 1e-7
            normal[stable] = (positions[frame, stable]-surface[stable]) / signed[stable, None]
            projected = surface - normal * margin
            weights[frame, active] = 1. + collision_weight
            goals[frame, active] = (target[frame, active] + collision_weight*projected[active]) / (1. + collision_weight)
        if not active_frames:
            break
        selected = np.asarray(active_frames)
        active_pose = PoseSequence(current.local_rotations[selected], current.root_translation[selected], fps=motion.fps)
        fitted = refine_pose_to_joint_positions(
            extended, goals[selected], active_pose, position_weights=weights[selected],
            max_iterations=5, tolerance=1e-12,
        )
        rotations = current.local_rotations.copy()
        translation = current.root_translation.copy()
        rotations[selected] = fitted.local_rotations
        translation[selected] = fitted.root_translation
        current = PoseSequence(rotations, translation, fps=motion.fps)
    result = PoseSequence(current.local_rotations[:, :skeleton.num_joints].copy(), current.root_translation.copy(), fps=motion.fps)
    for frame, scene in enumerate(scenes):
        single = PoseSequence(result.local_rotations[frame:frame+1], result.root_translation[frame:frame+1], fps=motion.fps)
        fitted = constrain_surface_frame(
            skeleton, tracked_positions[frame], single, scene, origins[frame],
            margin=margin*.25, extent=extent,
        )
        result.local_rotations[frame] = fitted.local_rotations[0]
        result.root_translation[frame] = fitted.root_translation[0]
    return result
