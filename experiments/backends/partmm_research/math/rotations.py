"""Rotation representations and conversions"""
import torch
import numpy as np
from scipy.spatial.transform import Slerp, Rotation


def axis_angle_to_quaternion(axis_angle):
    """
    Convert axis-angle to quaternion

    Args:
        axis_angle: torch.Tensor or np.ndarray of shape (..., 3)

    Returns:
        quaternion: same type as input, shape (..., 4) in [w, x, y, z] format
    """
    is_torch = isinstance(axis_angle, torch.Tensor)

    if is_torch:
        device = axis_angle.device
        axis_angle = axis_angle.detach().cpu().numpy()

    # Compute angle (norm of axis-angle)
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)

    # Normalize axis (handle zero angle)
    axis = np.where(angle > 1e-8, axis_angle / angle, np.array([1, 0, 0]))

    # Compute quaternion
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis

    quaternion = np.concatenate([w, xyz], axis=-1)

    if is_torch:
        quaternion = torch.from_numpy(quaternion).float().to(device)

    return quaternion


def quaternion_to_axis_angle(quaternion):
    """
    Convert quaternion to axis-angle

    Args:
        quaternion: torch.Tensor or np.ndarray of shape (..., 4) in [w, x, y, z] format

    Returns:
        axis_angle: same type as input, shape (..., 3)
    """
    is_torch = isinstance(quaternion, torch.Tensor)

    if is_torch:
        device = quaternion.device
        quaternion = quaternion.detach().cpu().numpy()

    # Normalize quaternion
    quaternion = quaternion / (np.linalg.norm(quaternion, axis=-1, keepdims=True) + 1e-8)

    w = quaternion[..., 0:1]
    xyz = quaternion[..., 1:4]

    # Compute angle
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))

    # Compute axis (handle identity rotation)
    sin_half_angle = np.sqrt(1.0 - w * w + 1e-8)
    axis = xyz / sin_half_angle

    # Compute axis-angle
    axis_angle = angle * axis

    if is_torch:
        axis_angle = torch.from_numpy(axis_angle).float().to(device)

    return axis_angle


def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle to rotation matrix"""
    is_torch = isinstance(axis_angle, torch.Tensor)

    if is_torch:
        device = axis_angle.device
        axis_angle = axis_angle.detach().cpu().numpy()

    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, axis_angle / angle, np.array([1, 0, 0]))

    # Rodrigues formula
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    # Create skew-symmetric matrix
    K = np.zeros(axis.shape[:-1] + (3, 3))
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    I = np.eye(3)
    R = I + sin_angle[..., None] * K + (1 - cos_angle)[..., None] * (K @ K)

    if is_torch:
        R = torch.from_numpy(R).float().to(device)

    return R


def rotation_matrix_to_quaternion(matrix):
    """Convert rotation matrix to quaternion"""
    is_torch = isinstance(matrix, torch.Tensor)

    if is_torch:
        device = matrix.device
        matrix = matrix.detach().cpu().numpy()

    trace = np.trace(matrix, axis1=-2, axis2=-1)

    w = 0.5 * np.sqrt(1 + trace)
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * w)
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * w)
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * w)

    quaternion = np.stack([w, x, y, z], axis=-1)

    if is_torch:
        quaternion = torch.from_numpy(quaternion).float().to(device)

    return quaternion


def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to rotation matrix"""
    is_torch = isinstance(quaternion, torch.Tensor)

    if is_torch:
        device = quaternion.device
        quaternion = quaternion.detach().cpu().numpy()

    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    matrix = np.zeros(quaternion.shape[:-1] + (3, 3))

    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)

    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - w * x)

    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)

    if is_torch:
        matrix = torch.from_numpy(matrix).float().to(device)

    return matrix


def blend_transformations(trans_a, trans_b, alpha):
    """Blend two sets of 4x4 transformations"""
    return alpha * trans_a + (1 - alpha) * trans_b


def quaternion_slerp(quaternions, target_length):
    """
    Slerp interpolation for quaternions (支持任意数量的joints)

    Args:
        quaternions: np.ndarray [frames, n_joints, 4] in [w,x,y,z] format
        target_length: target frame count

    Returns:
        upsampled: np.ndarray [target_length, n_joints, 4]
    """
    frames, n_joints, _ = quaternions.shape

    # Original and target timestamps
    times = np.linspace(0, 1, frames)
    times_new = np.linspace(0, 1, target_length)

    # Slerp for each joint
    upsampled_quats = np.zeros((target_length, n_joints, 4))

    for joint_idx in range(n_joints):
        joint_quats = quaternions[:, joint_idx, :]  # [frames, 4] in [w,x,y,z]

        # Normalize quaternions
        norms = np.linalg.norm(joint_quats, axis=1, keepdims=True)
        joint_quats = joint_quats / (norms + 1e-8)

        # Convert to scipy format [x,y,z,w]
        joint_quats_scipy = joint_quats[:, [1, 2, 3, 0]]

        # Create Slerp interpolator
        rotations = Rotation.from_quat(joint_quats_scipy)
        slerp = Slerp(times, rotations)

        # Interpolate
        upsampled_rotations = slerp(times_new)
        upsampled_quats_scipy = upsampled_rotations.as_quat()  # [x,y,z,w]

        # Convert back to [w,x,y,z]
        upsampled_quats[:, joint_idx, :] = upsampled_quats_scipy[:, [3, 0, 1, 2]]

    return upsampled_quats
