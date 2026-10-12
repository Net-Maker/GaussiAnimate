"""Pose-layout contract for the rootless DressRecon SMPL skeleton adapter."""

from typing import Optional, Tuple

import numpy as np


BOB_ADAPTER_ID = "gaussianimate_dressrecon_composed_pose_bob_v2"
SMPL_JOINT_MAPPING_ID = "smpl_joints_1_23_to_rootless_links_v1"
SMPL_SOURCE_JOINTS = 24
SMPL_ROOT_JOINT_ID = 0
SMPL_BODY_JOINTS = 21
SMPL_ROOTLESS_JOINT_IDS = tuple(range(1, 24))
SMPL_ROOTLESS_JOINTS = len(SMPL_ROOTLESS_JOINT_IDS)
SMPL_ROOTLESS_POSE_DIM = SMPL_ROOTLESS_JOINTS * 3
SMPL_ROOT_HANDLING = "separate_excluded_from_rootless_fk"
SMPL_ROOT_TRANSLATION_HANDLING = "not_consumed_by_adapter"
BOB_PREDICTED_GAUSSIAN_ATTRIBUTES = ("means3D",)
BOB_EVALUATION_ATTRIBUTE_SOURCE = "target_frame"
BOB_ORACLE_EVALUATION_ATTRIBUTES = (
    "rgb_colors",
    "unnorm_rotations",
)


def split_smpl_root_and_links(
    axis_angle: np.ndarray,
    *,
    expected_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a complete 24-joint SMPL pose into root and rootless link poses.

    DressRecon's predefined SMPL skeleton drops joint 0 (pelvis). Its 23
    forward-kinematics slots therefore correspond exactly to SMPL joint IDs
    1 through 23, rather than to the first 23 entries of the source array.
    """

    values = np.asarray(axis_angle)
    expected_shape = (
        expected_frames,
        SMPL_SOURCE_JOINTS,
        3,
    )
    if (
        values.ndim != 3
        or values.shape[1:] != expected_shape[1:]
        or (
            expected_frames is not None
            and values.shape[0] != expected_frames
        )
    ):
        frame_label = (
            "frames"
            if expected_frames is None
            else str(expected_frames)
        )
        raise ValueError(
            "SMPL axis angles must have exact shape "
            f"({frame_label}, {SMPL_SOURCE_JOINTS}, 3); "
            f"received {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("SMPL axis angles must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("SMPL axis angles must be finite")

    root_axis_angle = np.asarray(
        values[:, SMPL_ROOT_JOINT_ID, :],
        dtype=np.float32,
    ).copy()
    rootless_pose = np.asarray(
        values[:, list(SMPL_ROOTLESS_JOINT_IDS), :],
        dtype=np.float32,
    ).reshape(values.shape[0], SMPL_ROOTLESS_POSE_DIM)
    return root_axis_angle, rootless_pose


def smplx_body_to_rootless_links(body_axis_angle: np.ndarray) -> np.ndarray:
    """Map the 21 SMPL-X body joints to 23 rootless SMPL links.

    SMPL-X body-pose records do not contain SMPL's two terminal hand joints,
    so the adapter appends two explicit zero rotations. Global/pelvis
    rotation is deliberately not accepted by this function.
    """

    values = np.asarray(body_axis_angle)
    if (
        values.ndim == 2
        and values.shape[1] == SMPL_BODY_JOINTS * 3
    ):
        values = values.reshape(values.shape[0], SMPL_BODY_JOINTS, 3)
    if (
        values.ndim != 3
        or values.shape[1:] != (SMPL_BODY_JOINTS, 3)
    ):
        raise ValueError(
            "SMPL-X body axis angles must have exact shape "
            f"[frames, {SMPL_BODY_JOINTS * 3}] or "
            f"[frames, {SMPL_BODY_JOINTS}, 3]; "
            f"received {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("SMPL-X body axis angles must be numeric")
    if not np.isfinite(values).all():
        raise ValueError("SMPL-X body axis angles must be finite")
    terminal_hands = np.zeros(
        (values.shape[0], 2, 3),
        dtype=values.dtype,
    )
    return np.concatenate(
        [values, terminal_hands],
        axis=1,
    ).astype(np.float32, copy=False).reshape(
        values.shape[0],
        SMPL_ROOTLESS_POSE_DIM,
    )


def reshape_rootless_pose(pose, joint_count: int, *, label: str = "pose"):
    """Return ``pose`` as ``[batch, joint_count, 3]`` with no padding."""

    shape = tuple(pose.shape)
    flat_dim = int(joint_count) * 3
    if len(shape) == 2 and shape[1] == flat_dim:
        return pose.reshape(shape[0], joint_count, 3)
    if len(shape) == 3 and shape[1:] == (joint_count, 3):
        return pose
    raise ValueError(
        f"{label} must have exact shape [batch, {flat_dim}] or "
        f"[batch, {joint_count}, 3]; received {shape}"
    )


def flatten_rootless_pose(pose, joint_count: int, *, label: str = "pose"):
    """Return a strictly validated rootless pose as ``[batch, joints * 3]``."""

    shaped = reshape_rootless_pose(
        pose,
        joint_count,
        label=label,
    )
    return shaped.reshape(shaped.shape[0], int(joint_count) * 3)
