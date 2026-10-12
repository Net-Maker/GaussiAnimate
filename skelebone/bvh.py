"""BVH export for inspectable temporal forward kinematics."""

from pathlib import Path
from typing import List

import numpy as np

from .kinematics import quaternion_to_matrix
from .types import PoseSequence, Skeleton


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """Return XYZ Euler angles in degrees for ``Rz @ Ry @ Rx`` matrices."""
    sy = np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    regular = sy > 1e-8
    x = np.where(regular, np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
                 np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]))
    y = np.arctan2(-matrix[..., 2, 0], sy)
    z = np.where(regular, np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]), 0.0)
    return np.rad2deg(np.stack([x, y, z], axis=-1))


def _children(parents: np.ndarray) -> List[List[int]]:
    result = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if parent >= 0:
            result[int(parent)].append(child)
    return result


def export_bvh(path, skeleton: Skeleton, pose: PoseSequence) -> Path:
    """Write one rooted skeleton and local pose sequence to a BVH file."""
    if pose.num_joints != skeleton.num_joints:
        raise ValueError("pose and skeleton joint counts differ")
    if len(skeleton.roots) != 1:
        raise ValueError("BVH export requires exactly one root")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    children = _children(skeleton.parents)
    offsets = skeleton.offsets
    channel_order = []
    lines = ["HIERARCHY"]

    def emit(joint: int, depth: int, root: bool = False) -> None:
        indent = "  " * depth
        kind = "ROOT" if root else "JOINT"
        lines.append(f"{indent}{kind} {skeleton.names[joint]}")
        lines.append(f"{indent}{{")
        off = np.zeros(3) if root else offsets[joint]
        lines.append(f"{indent}  OFFSET {off[0]:.9g} {off[1]:.9g} {off[2]:.9g}")
        if root:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Xrotation Yrotation Zrotation")
        channel_order.append(joint)
        for child in children[joint]:
            emit(child, depth + 1)
        if not children[joint]:
            lines.extend([
                f"{indent}  End Site",
                f"{indent}  {{",
                f"{indent}    OFFSET 0 0 0",
                f"{indent}  }}",
            ])
        lines.append(f"{indent}}}")

    root = int(skeleton.roots[0])
    emit(root, 0, root=True)
    euler = _matrix_to_euler_xyz(quaternion_to_matrix(pose.local_rotations))
    lines.extend(["MOTION", f"Frames: {pose.num_frames}", f"Frame Time: {1.0 / pose.fps:.9g}"])
    for frame in range(pose.num_frames):
        values = []
        for index, joint in enumerate(channel_order):
            if index == 0:
                root_position = skeleton.rest_joints[root] + pose.root_translation[frame]
                values.extend(root_position.tolist())
            values.extend(euler[frame, joint].tolist())
        lines.append(" ".join(f"{value:.9g}" for value in values))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
