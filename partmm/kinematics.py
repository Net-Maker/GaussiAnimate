"""Minimal quaternion kinematics required by standalone PartMM."""

from typing import Tuple

import numpy as np

from .types import PartMMQuery, PartMMSkeleton


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("quaternions contain a zero value")
    q = q / norms
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    flat = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quaternion = np.array([
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ])
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = 2.0 * np.sqrt(
                    1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
                )
                quaternion = np.array([
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ])
            elif axis == 1:
                scale = 2.0 * np.sqrt(
                    1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
                )
                quaternion = np.array([
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ])
            else:
                scale = 2.0 * np.sqrt(
                    1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
                )
                quaternion = np.array([
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ])
        quaternion /= np.linalg.norm(quaternion)
        result[index] = quaternion if quaternion[0] >= 0 else -quaternion
    return result.reshape(matrix.shape[:-2] + (4,))


def hierarchy_order(parents: np.ndarray) -> Tuple[int, ...]:
    parents = np.asarray(parents, dtype=np.int64)
    children = [[] for _ in parents]
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


def forward_kinematics(
    skeleton: PartMMSkeleton,
    pose: PartMMQuery,
) -> Tuple[np.ndarray, np.ndarray]:
    pose = PartMMQuery.from_pose(pose)
    if pose.num_joints != skeleton.num_joints:
        raise ValueError("pose and skeleton joint counts differ")
    frames, joints = pose.num_frames, skeleton.num_joints
    local_rotation = quaternion_to_matrix(pose.local_rotations)
    global_transform = np.zeros((frames, joints, 4, 4), dtype=np.float64)
    global_transform[..., 3, 3] = 1.0
    for joint in hierarchy_order(skeleton.parents):
        local = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
        local[:, :3, :3] = local_rotation[:, joint]
        parent = int(skeleton.parents[joint])
        if parent < 0:
            local[:, :3, 3] = skeleton.rest_joints[joint] + pose.root_translation
            global_transform[:, joint] = local
        else:
            local[:, :3, 3] = skeleton.offsets[joint]
            global_transform[:, joint] = global_transform[:, parent] @ local
    return global_transform, global_transform[:, :, :3, 3]


__all__ = [
    "forward_kinematics",
    "hierarchy_order",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
]
