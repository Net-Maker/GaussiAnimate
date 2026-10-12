"""Part-based Pyramid Generation - 支持可变joint数量"""
import torch
import torch.nn.functional as F


def generate_part_pyramid(motion_data, n_joints, coarse_ratio=0.4, pyr_factor=0.75):
    """
    为part motion生成pyramid（支持任意joint数量）

    Args:
        motion_data: [1, n_channels, n_frames] where n_channels = n_joints*4 + 3
        n_joints: number of joints in this part
        coarse_ratio: coarsest level ratio
        pyr_factor: scale factor between levels

    Returns:
        pyramid: list of motion data at different resolutions
        lengths: list of frame counts
    """
    n_frames = motion_data.shape[2]

    # Compute pyramid levels
    lengths = []
    current_length = int(n_frames * coarse_ratio)

    while current_length < n_frames:
        lengths.append(current_length)
        current_length = int(current_length / pyr_factor)

    lengths.append(n_frames)

    # Build pyramid
    pyramid = []
    for target_length in lengths:
        resampled = upsample_part_motion_with_slerp(motion_data, target_length, n_joints)
        pyramid.append(resampled)

    return pyramid, lengths


def upsample_part_motion_with_slerp(motion_data, target_length, n_joints):
    """
    使用Slerp对part motion进行上采样（支持任意joint数量）

    Args:
        motion_data: [1, n_channels, frames] where n_channels = n_joints*4 + 3
        target_length: target frame count
        n_joints: number of joints

    Returns:
        upsampled: [1, n_channels, target_length]
    """
    from ..math.rotations import quaternion_slerp

    # Extract quaternions and global position
    quat_data = motion_data[:, :-3, :]  # [1, n_joints*4, frames]
    global_pos = motion_data[:, -3:, :]  # [1, 3, frames]

    # Reshape quaternions
    batch, quat_channels, frames = quat_data.shape
    quat_data_np = quat_data.squeeze(0).cpu().numpy()  # [n_joints*4, frames]
    quat_data_np = quat_data_np.T.reshape(frames, n_joints, 4)  # [frames, n_joints, 4]

    # Slerp interpolation for quaternions
    upsampled_quats = quaternion_slerp(quat_data_np, target_length)  # [target_length, n_joints, 4]

    # Reshape back
    upsampled_quats = upsampled_quats.reshape(target_length, n_joints * 4).T  # [n_joints*4, target_length]
    upsampled_quats = torch.from_numpy(upsampled_quats).float().unsqueeze(0).to(motion_data.device)

    # Linear interpolation for global position
    upsampled_pos = F.interpolate(
        global_pos,
        size=target_length,
        mode='linear',
        align_corners=False
    )

    # Concatenate
    upsampled = torch.cat([upsampled_quats, upsampled_pos], dim=1)

    return upsampled


def se3_interpolate_part(transforms, target_length):
    """
    对part的SE3 transforms进行插值（支持任意bone数量）
    参考 test_patch.py:_se3_interpolate (lines 2281-2306)

    Args:
        transforms: [frames, n_bones, 4, 4]
        target_length: target frame count

    Returns:
        interpolated: [target_length, n_bones, 4, 4]
    """
    from ..math import se3_log_map, se3_exp_map

    frames, n_bones, _, _ = transforms.shape
    device = transforms.device

    if frames == target_length:
        return transforms

    # Compute log map for all frames and bones: [frames, n_bones, 4, 4]
    log_transforms = torch.zeros_like(transforms)
    for frame_idx in range(frames):
        for bone_idx in range(n_bones):
            log_transforms[frame_idx, bone_idx] = se3_log_map(transforms[frame_idx, bone_idx])

    # Flatten se(3) matrices: [frames, n_bones, 4, 4] -> [frames, n_bones*16]
    log_flat = log_transforms.reshape(frames, n_bones * 16)

    # Linear interpolation in Lie algebra
    log_interp = F.interpolate(
        log_flat.T.unsqueeze(0),  # [1, n_bones*16, frames]
        size=target_length,
        mode='linear',
        align_corners=False
    )

    # Reshape: [1, n_bones*16, target_length] -> [target_length, n_bones, 4, 4]
    log_interp = log_interp.squeeze(0).T.reshape(target_length, n_bones, 4, 4)

    # Convert back to SE(3) using exponential map
    result = torch.zeros_like(log_interp)
    for frame_idx in range(target_length):
        for bone_idx in range(n_bones):
            result[frame_idx, bone_idx] = se3_exp_map(log_interp[frame_idx, bone_idx])

    return result
