"""Pyramid generation"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from ..math import axis_angle_to_quaternion, quaternion_to_axis_angle


def generate_pyramid(motion_data, coarse_ratio=0.2, pyr_factor=0.75):
    """
    Generate motion pyramid

    Args:
        motion_data: [1, channels, frames]
        coarse_ratio: coarsest level ratio
        pyr_factor: upsampling factor

    Returns:
        pyramid: list of tensors
        lengths: list of frame counts
    """
    original_length = motion_data.shape[-1]

    lengths = [int(np.round(original_length * coarse_ratio))]
    while lengths[-1] < original_length:
        lengths.append(int(np.round(lengths[-1] / pyr_factor)))
        if lengths[-1] == lengths[-2]:
            lengths[-1] += 1
    lengths[-1] = original_length

    pyramid = []
    for length in lengths:
        if length == original_length:
            pyramid.append(motion_data)
        else:
            resampled = upsample_smpl_with_slerp(motion_data, length)
            pyramid.append(resampled)

    return pyramid, lengths


def upsample_smpl_with_slerp(smpl_data, target_length):
    """
    Upsample SMPL data using Slerp for quaternions

    Args:
        smpl_data: [1, 99, frames] where 99 = 96 (quats) + 3 (pos)
        target_length: target frame count

    Returns:
        upsampled: [1, 99, target_length]
    """
    batch, channels, frames = smpl_data.shape
    device = smpl_data.device

    quat_data = smpl_data[:, :96, :].squeeze(0).cpu().numpy()
    other_data = smpl_data[:, 96:, :]

    quat_data = quat_data.T.reshape(frames, 24, 4)

    times = np.linspace(0, 1, frames)
    times_new = np.linspace(0, 1, target_length)

    upsampled_quats = np.zeros((target_length, 24, 4))
    for joint_idx in range(24):
        joint_quats = quat_data[:, joint_idx, :]
        norms = np.linalg.norm(joint_quats, axis=1, keepdims=True)
        joint_quats = joint_quats / (norms + 1e-8)
        joint_quats_scipy = joint_quats[:, [1, 2, 3, 0]]

        rotations = Rotation.from_quat(joint_quats_scipy)
        slerp = Slerp(times, rotations)
        interp_rotations = slerp(times_new)
        interp_quats_scipy = interp_rotations.as_quat()
        upsampled_quats[:, joint_idx, :] = interp_quats_scipy[:, [3, 0, 1, 2]]

    upsampled_quats = upsampled_quats.reshape(target_length, 96).T
    upsampled_quats = torch.from_numpy(upsampled_quats).float().to(device).unsqueeze(0)

    other_upsampled = F.interpolate(other_data, size=target_length, mode="linear", align_corners=False)

    return torch.cat([upsampled_quats, other_upsampled], dim=1)
