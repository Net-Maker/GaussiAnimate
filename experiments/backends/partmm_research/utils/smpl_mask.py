"""SMPL-specific masking utilities"""
import torch
from .mask import SMPL_BODY_PARTS


def create_smpl_part_channel_masks(part_groups):
    """
    Create channel masks for SMPL parts (quaternion format)

    SMPL format: 24 joints × 4 (quaternion) + 3 (global pos) = 99 channels

    Args:
        part_groups: dict {part_name: [joint_indices]}

    Returns:
        part_masks: dict {part_name: torch.Tensor [99]}
    """
    n_joints = 24
    quat_per_joint = 4
    n_total_channels = n_joints * quat_per_joint + 3  # 99

    part_masks = {}

    for part_name, joint_indices in part_groups.items():
        mask = torch.zeros(n_total_channels)

        # Set quaternion channels for joints in this part
        for joint_idx in joint_indices:
            start_ch = joint_idx * quat_per_joint
            end_ch = start_ch + quat_per_joint
            mask[start_ch:end_ch] = 1.0

        # Global position: always 1.0
        mask[-3:] = 1.0

        part_masks[part_name] = mask

    return part_masks


def create_ssdr_part_channel_masks(part_to_bones):
    """
    Create channel masks for SSDR parts (SE3 transform format)

    SSDR format: n_bones × 16 (4x4 matrix) + 3 (global pos)

    Args:
        part_to_bones: dict {part_name: [bone_indices]}

    Returns:
        part_masks: dict {part_name: torch.Tensor [n_bones*16 + 3]}
    """
    # Find total number of bones
    all_bones = []
    for bones in part_to_bones.values():
        all_bones.extend(bones)
    n_bones = max(all_bones) + 1
    n_total_channels = n_bones * 16 + 3

    part_masks = {}

    for part_name, bone_indices in part_to_bones.items():
        mask = torch.zeros(n_total_channels)

        # Set channels for bones in this part
        for bone_idx in bone_indices:
            start_ch = bone_idx * 16
            end_ch = start_ch + 16
            mask[start_ch:end_ch] = 1.0

        # Global position: always 1.0
        mask[-3:] = 1.0

        part_masks[part_name] = mask

    return part_masks
