"""
Bone masking utilities for part-based processing

Defines body part groups and provides masking functions
"""

import torch
import numpy as np


# SMPL skeleton part groups (24 joints)
# Fine-grained parts: ~3 joints per part
# Root (0) is shared by torso and leg parts for continuity
SMPL_BODY_PARTS_FINE = {
    # Torso: spine chain (pelvis -> neck -> head)
    'torso_lower': [0, 3, 6],              # Pelvis, Spine1, Spine2 (包含root)
    'torso_upper': [0, 9, 12],             # Pelvis, Spine3, Neck (包含root)
    'head_neck': [12, 15],                 # Neck, Head (连接到torso_upper)

    # Left arm: shoulder -> hand
    'left_upper_arm': [13, 16, 18],        # L_Collar, L_Shoulder, L_Elbow
    'left_lower_arm': [18, 20, 22],        # L_Elbow, L_Wrist, L_Hand (与upper_arm重叠)

    # Right arm: shoulder -> hand
    'right_upper_arm': [14, 17, 19],       # R_Collar, R_Shoulder, R_Elbow
    'right_lower_arm': [19, 21, 23],       # R_Elbow, R_Wrist, R_Hand (与upper_arm重叠)

    # Left leg: hip -> foot
    'left_upper_leg': [0, 1, 4],           # Pelvis, L_Hip, L_Knee (包含root)
    'left_lower_leg': [4, 7, 10],          # L_Knee, L_Ankle, L_Foot (与upper_leg重叠)

    # Right leg: hip -> foot
    'right_upper_leg': [0, 2, 5],          # Pelvis, R_Hip, R_Knee (包含root)
    'right_lower_leg': [5, 8, 11],         # R_Knee, R_Ankle, R_Foot (与upper_leg重叠)
}

# Coarse-grained parts: torso + head + limbs (4-6 parts total)
# Simpler hierarchy for faster processing or rough motion transfer
SMPL_BODY_PARTS_COARSE = {
    # Torso: entire spine chain
    'torso': [0, 3, 6, 9, 12],             # Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck

    # Head
    'head': [12, 15],                       # Neck, Head (overlaps with torso at neck)

    # Left arm: entire arm chain
    'left_arm': [13, 16, 18, 20, 22],      # Collar -> Shoulder -> Elbow -> Wrist -> Hand

    # Right arm: entire arm chain
    'right_arm': [14, 17, 19, 21, 23],     # Collar -> Shoulder -> Elbow -> Wrist -> Hand

    # Left leg: entire leg chain
    'left_leg': [0, 1, 4, 7, 10],          # Pelvis, Hip -> Knee -> Ankle -> Foot (includes root)

    # Right leg: entire leg chain
    'right_leg': [0, 2, 5, 8, 11],         # Pelvis, Hip -> Knee -> Ankle -> Foot (includes root)
}

# Default: use fine-grained parts
SMPL_BODY_PARTS = SMPL_BODY_PARTS_COARSE


def map_smpl_parts_to_ssdr_bones(skeletal_binding, part_groups=SMPL_BODY_PARTS, threshold=0.3):
    """
    Map SMPL part groups to SSDR bone groups based on binding weights

    Strategy:
    1. For each SSDR bone, check its parent SMPL joints
    2. Assign bone to part if majority of binding weight goes to that part
    3. Handle boundary bones (influenced by multiple parts)

    Args:
        skeletal_binding: dict from SkeletalBinding.build()
        part_groups: dict mapping part_name -> list of SMPL joint indices
        threshold: minimum weight to consider a bone belongs to a part

    Returns:
        ssdr_part_groups: dict mapping part_name -> list of SSDR bone indices
        boundary_bones: list of bone indices influenced by multiple parts
    """
    binding_indices = skeletal_binding['indices']  # (n_bones, k)
    binding_weights = skeletal_binding['weights']  # (n_bones, k)
    n_ssdr_bones = skeletal_binding['n_ssdr_bones']

    ssdr_part_groups = {part: [] for part in part_groups}
    boundary_bones = []

    # Create inverse map: joint_idx -> part_name
    joint_to_part = {}
    for part_name, joint_indices in part_groups.items():
        for joint_idx in joint_indices:
            joint_to_part[joint_idx] = part_name

    # For each SSDR bone
    for bone_idx in range(n_ssdr_bones):
        parent_joints = binding_indices[bone_idx].cpu().numpy()
        weights = binding_weights[bone_idx].cpu().numpy()

        # Accumulate weight per part
        part_weights = {part: 0.0 for part in part_groups}
        for parent_idx, weight in zip(parent_joints, weights):
            if parent_idx in joint_to_part:
                part_name = joint_to_part[parent_idx]
                part_weights[part_name] += weight

        # Find dominant part
        max_part = max(part_weights, key=part_weights.get)
        max_weight = part_weights[max_part]

        # Check if boundary bone
        significant_parts = [p for p, w in part_weights.items() if w > threshold]
        if len(significant_parts) > 1:
            boundary_bones.append(bone_idx)

        # Assign to dominant part
        if max_weight > 0:
            ssdr_part_groups[max_part].append(bone_idx)

    print(f"\nSSMPL-to-SSDR Part Mapping:")
    for part_name, bone_indices in ssdr_part_groups.items():
        print(f"  {part_name:12s}: {len(bone_indices):3d} bones")
    print(f"  {'boundary':12s}: {len(boundary_bones):3d} bones")

    return ssdr_part_groups, boundary_bones


def create_part_channel_masks(part_groups, n_channels_per_bone=16):
    """
    Create channel-level masks for each body part

    Args:
        part_groups: dict mapping part_name -> list of bone indices
        n_channels_per_bone: channels per bone (16 for 4x4 transforms)

    Returns:
        part_masks: dict mapping part_name -> torch.Tensor mask (n_total_channels,)
    """
    # Find total number of bones
    all_bones = []
    for bones in part_groups.values():
        all_bones.extend(bones)
    n_bones = max(all_bones) + 1
    n_total_channels = n_bones * n_channels_per_bone + 3  # +3 for global position

    part_masks = {}
    for part_name, bone_indices in part_groups.items():
        mask = torch.zeros(n_total_channels)

        # Set bone channels
        for bone_idx in bone_indices:
            start_ch = bone_idx * n_channels_per_bone
            end_ch = start_ch + n_channels_per_bone
            mask[start_ch:end_ch] = 1.0

        # Global position: always 1.0 (affects all parts)
        mask[-3:] = 1.0

        part_masks[part_name] = mask

    return part_masks


def create_boundary_smoothing_weights(part_groups, boundary_bones, n_channels_per_bone=16, smooth_width=3):
    """
    Create smoothing weights for boundary bones

    Boundary bones should blend between adjacent parts

    Args:
        part_groups: dict of part groups
        boundary_bones: list of boundary bone indices
        n_channels_per_bone: channels per bone
        smooth_width: number of bones to smooth around boundary

    Returns:
        smoothing_weights: dict mapping bone_idx -> weights per part
    """
    smoothing_weights = {}

    for bone_idx in boundary_bones:
        # Find which parts influence this bone
        influencing_parts = []
        for part_name, bone_indices in part_groups.items():
            if bone_idx in bone_indices:
                influencing_parts.append(part_name)

        # Equal blending for now (can be refined based on distance)
        if len(influencing_parts) > 0:
            weight = 1.0 / len(influencing_parts)
            smoothing_weights[bone_idx] = {part: weight for part in influencing_parts}

    return smoothing_weights


# PART-BASED EXTENSION PLACEHOLDER
class PartBasedProcessor:
    """
    Processor for part-based motion synthesis

    This is a PLACEHOLDER for your part-based implementation.
    Key methods to implement:
    1. extract_part_motion()
    2. process_part_independently()
    3. merge_parts_with_smoothing()
    """

    def __init__(self, part_groups, skeletal_binding):
        self.part_groups = part_groups
        self.skeletal_binding = skeletal_binding

        # Map SMPL parts to SSDR bones
        self.ssdr_part_groups, self.boundary_bones = map_smpl_parts_to_ssdr_bones(
            skeletal_binding, part_groups
        )

        # Create channel masks
        self.part_masks = create_part_channel_masks(self.ssdr_part_groups)

        # Create boundary smoothing
        self.smoothing_weights = create_boundary_smoothing_weights(
            self.ssdr_part_groups, self.boundary_bones
        )

    def extract_part_motion(self, full_motion, part_name):
        """
        Extract channels belonging to a specific part

        Args:
            full_motion: torch.Tensor (1, n_channels, n_frames)
            part_name: str

        Returns:
            part_motion: torch.Tensor (1, n_part_channels, n_frames)
        """
        mask = self.part_masks[part_name]
        part_channels = torch.where(mask > 0)[0]
        return full_motion[:, part_channels, :]

    def merge_parts(self, part_motions, smooth_boundaries=True):
        """
        Merge part motions back into full motion

        Args:
            part_motions: dict mapping part_name -> torch.Tensor
            smooth_boundaries: whether to smooth boundary bones

        Returns:
            full_motion: torch.Tensor (1, n_channels, n_frames)
        """
        # TODO: Implement boundary-aware merging
        raise NotImplementedError("Part merging not implemented yet")
