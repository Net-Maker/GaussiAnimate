"""Small BVH writer used by the optional standalone pose editor."""

from pathlib import Path

import numpy as np

from .kinematics import quaternion_to_matrix
from .types import PartMMQuery, PartMMSkeleton


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    sy = np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    regular = sy > 1e-8
    x = np.where(
        regular,
        np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
        np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
    )
    y = np.arctan2(-matrix[..., 2, 0], sy)
    z = np.where(
        regular,
        np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]),
        0.0,
    )
    return np.rad2deg(np.stack([x, y, z], axis=-1))


def export_bvh(path, skeleton: PartMMSkeleton, pose: PartMMQuery) -> Path:
    pose = PartMMQuery.from_pose(pose)
    if pose.num_joints != skeleton.num_joints:
        raise ValueError("pose and skeleton joint counts differ")
    if len(skeleton.roots) != 1:
        raise ValueError("BVH export requires exactly one root")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    children = [[] for _ in skeleton.parents]
    for child, parent in enumerate(skeleton.parents):
        if parent >= 0:
            children[int(parent)].append(child)
    order = []
    lines = ["HIERARCHY"]

    def emit(joint: int, depth: int, root: bool = False) -> None:
        indent = "  " * depth
        kind = "ROOT" if root else "JOINT"
        lines.extend([f"{indent}{kind} {skeleton.names[joint]}", f"{indent}{{"])
        offset = np.zeros(3) if root else skeleton.offsets[joint]
        lines.append(
            f"{indent}  OFFSET {offset[0]:.9g} {offset[1]:.9g} {offset[2]:.9g}"
        )
        channels = (
            "6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
            if root
            else "3 Xrotation Yrotation Zrotation"
        )
        lines.append(f"{indent}  CHANNELS {channels}")
        order.append(joint)
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
    lines.extend([
        "MOTION",
        f"Frames: {pose.num_frames}",
        f"Frame Time: {1.0 / pose.fps:.9g}",
    ])
    for frame in range(pose.num_frames):
        values = []
        for index, joint in enumerate(order):
            if index == 0:
                values.extend(
                    (skeleton.rest_joints[root] + pose.root_translation[frame]).tolist()
                )
            values.extend(euler[frame, joint].tolist())
        lines.append(" ".join(f"{value:.9g}" for value in values))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = ["export_bvh"]
