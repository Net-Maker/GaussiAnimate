"""Quaternion kinematics shared by skelebone and partmm."""

from typing import Tuple

import numpy as np

from .types import PoseSequence, Skeleton


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized scalar-first quaternions to rotation matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to scalar-first quaternions."""
    flat = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array([0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale])
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = 2.0 * np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                quaternion = np.array([(rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale])
            elif axis == 1:
                scale = 2.0 * np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                quaternion = np.array([(rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale])
            else:
                scale = 2.0 * np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                quaternion = np.array([(rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale])
        quaternion /= np.linalg.norm(quaternion)
        result[index] = quaternion if quaternion[0] >= 0 else -quaternion
    return result.reshape(matrix.shape[:-2] + (4,))


def hierarchy_order(parents: np.ndarray) -> Tuple[int, ...]:
    """Return a deterministic parent-before-child traversal."""
    parents = np.asarray(parents, dtype=np.int64)
    children = [[] for _ in range(len(parents))]
    roots = []
    for child, parent in enumerate(parents):
        if parent < 0:
            roots.append(child)
        else:
            children[int(parent)].append(child)
    order = []
    queue = list(roots)
    while queue:
        joint = queue.pop(0)
        order.append(joint)
        queue.extend(children[joint])
    if len(order) != len(parents):
        raise ValueError("parents do not define a valid forest")
    return tuple(order)


def forward_kinematics(skeleton: Skeleton, pose: PoseSequence) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate global joint transforms and positions for all frames."""
    if pose.num_joints != skeleton.num_joints:
        raise ValueError("pose and skeleton joint counts differ")
    frames, joints = pose.num_frames, skeleton.num_joints
    local_rotation = quaternion_to_matrix(pose.local_rotations)
    global_transform = np.zeros((frames, joints, 4, 4), dtype=np.float64)
    global_transform[..., 3, 3] = 1.0
    offsets = skeleton.offsets

    for joint in hierarchy_order(skeleton.parents):
        local = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
        local[:, :3, :3] = local_rotation[:, joint]
        parent = int(skeleton.parents[joint])
        if parent < 0:
            local[:, :3, 3] = skeleton.rest_joints[joint] + pose.root_translation
            global_transform[:, joint] = local
        else:
            local[:, :3, 3] = offsets[joint]
            global_transform[:, joint] = global_transform[:, parent] @ local

    return global_transform, global_transform[:, :, :3, 3]


def bind_matrices(skeleton: Skeleton) -> np.ndarray:
    """Return global bind transforms for identity local rotations."""
    identity = PoseSequence.identity(1, skeleton.num_joints)
    return forward_kinematics(skeleton, identity)[0][0]


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    cosine = np.clip(np.dot(source, target), -1.0, 1.0)
    sine = np.linalg.norm(cross)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew * ((1 - cosine) / (sine * sine))


def fit_pose_to_joint_positions(
    skeleton: Skeleton,
    joint_positions: np.ndarray,
    fps: float = 30.0,
) -> PoseSequence:
    """Fit an interpretable local-rotation sequence to tracked joint positions.

    Each joint orientation is estimated from its outgoing rest/observed bone
    directions. Branch joints use Kabsch; single-child joints use the minimum
    vector-alignment rotation. This gives a deterministic FK/BVH initialization
    that can subsequently be refined by the paper's IK objective.
    """
    positions = np.asarray(joint_positions, dtype=np.float64)
    expected_tail = (skeleton.num_joints, 3)
    if positions.ndim != 3 or positions.shape[1:] != expected_tail:
        raise ValueError(f"joint_positions must have shape [frames, {expected_tail[0]}, 3]")
    children = [[] for _ in range(skeleton.num_joints)]
    for child, parent in enumerate(skeleton.parents):
        if parent >= 0:
            children[int(parent)].append(child)
    global_rotation = np.broadcast_to(np.eye(3), (len(positions), skeleton.num_joints, 3, 3)).copy()
    for frame in range(len(positions)):
        for joint in hierarchy_order(skeleton.parents):
            child_ids = children[joint]
            if len(child_ids) == 1:
                child = child_ids[0]
                global_rotation[frame, joint] = _align_vectors(
                    skeleton.rest_joints[child] - skeleton.rest_joints[joint],
                    positions[frame, child] - positions[frame, joint],
                )
            elif len(child_ids) > 1:
                source = skeleton.rest_joints[child_ids] - skeleton.rest_joints[joint]
                target = positions[frame, child_ids] - positions[frame, joint]
                source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
                target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
                u, _, vt = np.linalg.svd(source.T @ target)
                rotation = vt.T @ u.T
                if np.linalg.det(rotation) < 0:
                    vt[-1] *= -1
                    rotation = vt.T @ u.T
                global_rotation[frame, joint] = rotation
            else:
                parent = int(skeleton.parents[joint])
                if parent >= 0:
                    global_rotation[frame, joint] = global_rotation[frame, parent]
    local_rotation = global_rotation.copy()
    for joint in hierarchy_order(skeleton.parents):
        parent = int(skeleton.parents[joint])
        if parent >= 0:
            local_rotation[:, joint] = np.swapaxes(global_rotation[:, parent], -1, -2) @ global_rotation[:, joint]
    root = int(skeleton.roots[0])
    translation = positions[:, root] - skeleton.rest_joints[root]
    return PoseSequence(matrix_to_quaternion(local_rotation), translation, fps=fps)
