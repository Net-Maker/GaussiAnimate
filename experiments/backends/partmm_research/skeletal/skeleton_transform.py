"""
Skeleton-Guided SSDR Transformation

Two modes:
1. Kabsch: Global optimal rotation for the entire part
2. Edge-wise: Per-bone alignment along kinematic chain (more precise)
"""

import torch
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple


from ._smpl_config import get_smpl_model_dir


# SMPL kinematic tree (parent indices)
SMPL_PARENTS = np.array([
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
     9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21
])

# Define kinematic chains for each body part (ordered from root to tip)
KINEMATIC_CHAINS = {
    'torso': [(0, 3), (3, 6), (6, 9), (9, 12)],  # Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck
    'head': [(12, 15)],  # Neck -> Head
    'left_arm': [(9, 13), (13, 16), (16, 18), (18, 20), (20, 22)],  # Spine3 -> Collar -> Shoulder -> Elbow -> Wrist -> Hand
    'right_arm': [(9, 14), (14, 17), (17, 19), (19, 21), (21, 23)],
    'left_leg': [(0, 1), (1, 4), (4, 7), (7, 10)],  # Pelvis -> Hip -> Knee -> Ankle -> Foot
    'right_leg': [(0, 2), (2, 5), (5, 8), (8, 11)],
}


def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to rotation matrix using Rodrigues formula."""
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    angle = np.maximum(angle, 1e-8)
    axis = axis_angle / angle

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    K = np.zeros((*axis.shape[:-1], 3, 3))
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    I = np.eye(3)
    R = I + sin_a[..., None] * K + (1 - cos_a[..., None]) * np.matmul(K, K)
    return R


def compute_global_transforms_from_poses(poses: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """
    Compute global rigid transform (4x4) for each joint using FK.

    Args:
        poses: (72,) SMPL axis-angle poses (24 joints * 3)
        joints: (24, 3) joint positions from SMPL FK

    Returns:
        global_transforms: (24, 4, 4) global rigid transform for each joint
    """
    from scipy.spatial.transform import Rotation

    n_joints = 24

    # Compute local rotations from poses using scipy (more robust)
    local_rotations = np.zeros((n_joints, 3, 3))
    for j in range(n_joints):
        axis_angle = poses[j*3:(j+1)*3]
        local_rotations[j] = Rotation.from_rotvec(axis_angle).as_matrix()

    # Compute global rotations by accumulating along kinematic chain
    global_rotations = np.zeros((n_joints, 3, 3))
    global_rotations[0] = local_rotations[0]  # Root is global

    for j in range(1, n_joints):
        parent = SMPL_PARENTS[j]
        global_rotations[j] = global_rotations[parent] @ local_rotations[j]

    # Build 4x4 transforms: rotation + translation (joint position)
    global_transforms = np.zeros((n_joints, 4, 4))
    for j in range(n_joints):
        global_transforms[j, :3, :3] = global_rotations[j]
        global_transforms[j, :3, 3] = joints[j]
        global_transforms[j, 3, 3] = 1.0

    return global_transforms


def compute_segment_relative_transform(
    db_transforms: np.ndarray,
    novel_transforms: np.ndarray,
    segment: Tuple[int, int]
) -> np.ndarray:
    """
    Compute the relative rigid transform for a segment from DB to Novel.

    For segment (parent_j, child_j), the segment inherits the parent joint's transform.
    Relative transform = Novel_transform @ inv(DB_transform)

    Args:
        db_transforms: (24, 4, 4) global transforms for DB pose
        novel_transforms: (24, 4, 4) global transforms for Novel pose
        segment: (parent_j, child_j) tuple

    Returns:
        relative_transform: (4, 4) relative rigid transform
    """
    parent_j, child_j = segment

    # Segment uses parent joint's transform
    T_db = db_transforms[parent_j]
    T_novel = novel_transforms[parent_j]

    # Relative transform: T_novel @ inv(T_db)
    T_db_inv = np.linalg.inv(T_db)
    relative_transform = T_novel @ T_db_inv

    return relative_transform


class SMPLForwardKinematics:
    """Simple SMPL forward kinematics to compute joint positions from poses."""

    PARENTS = SMPL_PARENTS

    def __init__(self, gender='female'):
        model_dir = get_smpl_model_dir()
        model_path = model_dir / f"SMPL_{gender.upper()}.pkl"
        if not model_path.exists():
            model_path = model_dir / "SMPL_NEUTRAL.pkl"

        with open(model_path, 'rb') as f:
            smpl_data = pickle.load(f, encoding='latin1')

        self.J_regressor = np.array(smpl_data['J_regressor'].todense())
        self.v_template = smpl_data['v_template']
        self.J = self.J_regressor @ self.v_template

    def forward(self, poses: np.ndarray, trans: np.ndarray = None) -> np.ndarray:
        if poses.ndim == 2:
            T = poses.shape[0]
            poses = poses.reshape(T, 24, 3)
        else:
            T = poses.shape[0]

        R_local = axis_angle_to_rotation_matrix(poses)
        joints = np.zeros((T, 24, 3))
        R_global = np.zeros((T, 24, 3, 3))

        for j in range(24):
            parent = self.PARENTS[j]
            if parent == -1:
                R_global[:, j] = R_local[:, j]
                joints[:, j] = self.J[j]
            else:
                R_global[:, j] = R_global[:, parent] @ R_local[:, j]
                joints[:, j] = joints[:, parent] + R_global[:, parent] @ (self.J[j] - self.J[parent])

        if trans is not None:
            joints = joints + trans[:, None, :]

        return joints


_smpl_fk_cache = {}


def get_smpl_fk(gender='female') -> SMPLForwardKinematics:
    if gender not in _smpl_fk_cache:
        _smpl_fk_cache[gender] = SMPLForwardKinematics(gender)
    return _smpl_fk_cache[gender]


def compute_joint_positions(poses: np.ndarray, trans: np.ndarray = None,
                           gender='female') -> np.ndarray:
    fk = get_smpl_fk(gender)
    return fk.forward(poses, trans)


def rotation_from_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix that rotates v1 to v2.

    Args:
        v1: (..., 3) source vectors (will be normalized)
        v2: (..., 3) target vectors (will be normalized)

    Returns:
        R: (..., 3, 3) rotation matrices
    """
    # Normalize
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)

    # Cross product gives rotation axis
    axis = torch.cross(v1, v2, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Dot product gives cos(angle)
    cos_angle = torch.sum(v1 * v2, dim=-1, keepdim=True)

    # Handle parallel vectors (no rotation needed)
    parallel_mask = axis_norm.squeeze(-1) < 1e-6

    # Normalize axis
    axis = axis / (axis_norm + 1e-8)

    # sin(angle) = |axis| before normalization
    sin_angle = axis_norm

    # Build rotation matrix using Rodrigues formula
    # R = I + sin(a)*K + (1-cos(a))*K^2
    batch_shape = v1.shape[:-1]
    device = v1.device
    dtype = v1.dtype

    # Skew-symmetric matrix K
    K = torch.zeros(*batch_shape, 3, 3, device=device, dtype=dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    I = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)

    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * torch.bmm(
        K.reshape(-1, 3, 3), K.reshape(-1, 3, 3)
    ).reshape(*batch_shape, 3, 3)

    # For parallel vectors, use identity
    if parallel_mask.any():
        R[parallel_mask] = I[parallel_mask]

    return R


def rotation_angle_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """Compute rotation angle from rotation matrix."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    return torch.acos(cos_angle)


def edge_wise_alignment(
    novel_joints: torch.Tensor,
    db_joints: torch.Tensor,
    chain: List[Tuple[int, int]],
    pivot_joint: int = None
) -> List[Tuple[Tuple[int, int], torch.Tensor]]:
    """
    Compute per-edge rotations to align database skeleton to novel skeleton.

    Args:
        novel_joints: (B, 24, 3) novel joint positions
        db_joints: (B, 24, 3) database joint positions
        chain: List of (parent, child) joint index tuples defining the kinematic chain
        pivot_joint: Optional pivot point for rotation (default: first joint of chain)

    Returns:
        List of ((parent, child), R) where R is (B, 3, 3) rotation matrix for that edge
    """
    results = []

    for parent_idx, child_idx in chain:
        # Bone direction in novel pose
        novel_bone = novel_joints[:, child_idx] - novel_joints[:, parent_idx]

        # Bone direction in database pose
        db_bone = db_joints[:, child_idx] - db_joints[:, parent_idx]

        # Compute rotation from db_bone to novel_bone
        R = rotation_from_vectors(db_bone, novel_bone)

        results.append(((parent_idx, child_idx), R))

    return results


def apply_edge_rotations_to_ssdr(
    ssdr_transforms: torch.Tensor,
    edge_rotations: List[Tuple[Tuple[int, int], torch.Tensor]],
    bone_to_joint_mapping: Dict[int, int],
    joint_to_bones: Dict[int, List[int]],
    pivot_positions: torch.Tensor = None
) -> torch.Tensor:
    """
    Apply edge-wise rotations to SSDR bone transforms.

    Each SSDR bone is rotated based on its associated skeleton joint.
    Rotations are applied cumulatively along the kinematic chain.

    Args:
        ssdr_transforms: (T, n_bones, 4, 4) SSDR transforms
        edge_rotations: List of ((parent, child), R) from edge_wise_alignment
        bone_to_joint_mapping: Maps SSDR bone index to SMPL joint index
        joint_to_bones: Maps SMPL joint index to list of SSDR bone indices
        pivot_positions: (T, 3) pivot point for rotation (optional)

    Returns:
        transformed: (T, n_bones, 4, 4) transformed SSDR
    """
    T, n_bones, _, _ = ssdr_transforms.shape
    transformed = ssdr_transforms.clone()

    # Build cumulative rotation for each joint
    # Joints downstream in the chain get all parent rotations applied
    joint_cumulative_R = {}

    cumulative_R = torch.eye(3, device=ssdr_transforms.device, dtype=ssdr_transforms.dtype)
    cumulative_R = cumulative_R.unsqueeze(0).expand(T, -1, -1).clone()

    for (parent_idx, child_idx), R in edge_rotations:
        # Accumulate rotation
        cumulative_R = torch.bmm(R, cumulative_R)

        # Store cumulative rotation for the child joint
        joint_cumulative_R[child_idx] = cumulative_R.clone()

        # Also store for parent if not already stored
        if parent_idx not in joint_cumulative_R:
            # Parent uses previous cumulative (identity for first edge)
            joint_cumulative_R[parent_idx] = torch.bmm(
                torch.inverse(R), cumulative_R
            )

    # Apply rotations to SSDR bones based on their joint association
    for bone_idx in range(n_bones):
        if bone_idx not in bone_to_joint_mapping:
            continue

        joint_idx = bone_to_joint_mapping[bone_idx]
        if joint_idx not in joint_cumulative_R:
            continue

        R_apply = joint_cumulative_R[joint_idx]

        # Apply rotation to bone transform
        R_bone = transformed[:, bone_idx, :3, :3]
        t_bone = transformed[:, bone_idx, :3, 3]

        # Rotate the bone's rotation
        R_new = torch.bmm(R_apply, R_bone)
        transformed[:, bone_idx, :3, :3] = R_new

        # Optionally rotate translation around pivot
        if pivot_positions is not None:
            t_rel = t_bone - pivot_positions
            t_rotated = torch.bmm(R_apply, t_rel.unsqueeze(-1)).squeeze(-1)
            transformed[:, bone_idx, :3, 3] = t_rotated + pivot_positions

    return transformed


class SkeletonGuidedMatcher:
    """
    Skeleton-guided transformation with two modes:
    1. 'kabsch': Global optimal rotation (original)
    2. 'edge_wise': Per-bone alignment along kinematic chain (new)
    """

    def __init__(
        self,
        part_to_joints: Dict[str, List[int]],
        part_to_bones: Dict[str, List[int]],
        enabled: bool = True,
        transform_weight: float = 1.0,
        mode: str = 'edge_wise',  # 'kabsch' or 'edge_wise'
        debug: bool = False,
        **kwargs
    ):
        self.part_to_joints = part_to_joints
        self.part_to_bones = part_to_bones
        self.enabled = enabled
        self.transform_weight = transform_weight
        self.mode = mode
        self.debug = debug

        self.novel_joint_positions = None
        self.db_joint_positions = None

        # NEW: Store poses for FK-based global transforms
        self.novel_poses = None
        self.db_poses = None

        # Build bone-to-joint mapping (will be set later)
        self.bone_to_joint = {}
        self.joint_to_bones = {}

        # NEW: bone-to-segment mapping (more accurate than bone-to-joint)
        self.bone_to_segment = {}
        self.segment_to_bones = {}

        # NEW: bone influence center positions (from weights)
        # These are the actual 3D positions of bones in rest pose
        self.bone_positions = None

    def set_joint_positions(
        self,
        novel_joint_positions: torch.Tensor,
        db_joint_positions: torch.Tensor
    ):
        self.novel_joint_positions = novel_joint_positions
        self.db_joint_positions = db_joint_positions
        if self.debug:
            print(f"  [SkeletonGuided] Set joint positions: novel={novel_joint_positions.shape}, db={db_joint_positions.shape}")
            print(f"  [SkeletonGuided] Mode: {self.mode}")

    def set_poses(
        self,
        novel_poses: np.ndarray,
        db_poses: np.ndarray
    ):
        """
        Set SMPL poses for FK-based global transform computation.

        Args:
            novel_poses: (T_novel, 72) SMPL axis-angle poses
            db_poses: (T_db, 72) SMPL axis-angle poses
        """
        self.novel_poses = novel_poses
        self.db_poses = db_poses
        if self.debug:
            print(f"  [SkeletonGuided] Set poses: novel={novel_poses.shape}, db={db_poses.shape}")

    def set_bone_joint_mapping(self, bone_to_joint: Dict[int, int]):
        """Set mapping from SSDR bone index to SMPL joint index (legacy)."""
        self.bone_to_joint = bone_to_joint
        # Build reverse mapping
        self.joint_to_bones = {}
        for bone_idx, joint_idx in bone_to_joint.items():
            if joint_idx not in self.joint_to_bones:
                self.joint_to_bones[joint_idx] = []
            self.joint_to_bones[joint_idx].append(bone_idx)

    def set_bone_segment_mapping(self, bone_to_segment: Dict[int, Tuple[int, int]]):
        """
        Set mapping from SSDR bone index to SMPL limb segment.

        Args:
            bone_to_segment: {bone_idx: (parent_joint, child_joint)}
        """
        self.bone_to_segment = bone_to_segment

        # Build reverse mapping: segment -> [bone_indices]
        self.segment_to_bones = {}
        for bone_idx, segment in bone_to_segment.items():
            if segment not in self.segment_to_bones:
                self.segment_to_bones[segment] = []
            self.segment_to_bones[segment].append(bone_idx)

        if self.debug:
            n_segments = len(self.segment_to_bones)
            print(f"  [SkeletonGuided] Set bone-to-segment mapping: {len(bone_to_segment)} bones -> {n_segments} segments")

    def set_bone_positions(self, bone_positions: torch.Tensor):
        """
        Set bone influence center positions (computed from weights).

        CRITICAL: SSDR transforms store RELATIVE displacement from rest pose,
        not absolute bone positions. We need the actual bone positions
        (computed as weighted average of vertices) to correctly transform.

        Args:
            bone_positions: (n_bones, 3) bone influence center positions in rest pose
        """
        self.bone_positions = bone_positions
        if self.debug:
            print(f"  [SkeletonGuided] Set bone positions: {bone_positions.shape}")

    def transform_matched_patches(
        self,
        novel_smpl_patches: torch.Tensor,
        db_combined_patches: torch.Tensor,
        matched_indices: torch.Tensor,
        part_name: str,
        n_smpl_channels: int,
        n_ssdr_channels: int,
        patch_size: int
    ) -> torch.Tensor:
        """Transform matched database patches using skeleton guidance."""

        if not self.enabled or part_name not in self.part_to_joints:
            return db_combined_patches[:, matched_indices, :]

        if self.novel_joint_positions is None or self.db_joint_positions is None:
            if self.debug:
                print(f"  [WARN] Joint positions not set, skipping skeleton transform")
            return db_combined_patches[:, matched_indices, :]

        part_joints = self.part_to_joints.get(part_name, [])
        part_bones = self.part_to_bones.get(part_name, [])

        if len(part_joints) < 2 or len(part_bones) == 0:
            return db_combined_patches[:, matched_indices, :]

        smpl_ch_total = n_smpl_channels * patch_size
        matched_combined = db_combined_patches[:, matched_indices, :]
        matched_smpl = matched_combined[:, :, :smpl_ch_total]
        matched_ssdr = matched_combined[:, :, smpl_ch_total:]

        # Always print for debugging
        print(f"  [SkeletonGuided] Part: {part_name}, mode: {self.mode}")
        print(f"    Joints: {part_joints}, Bones: {len(part_bones)}")
        print(f"    bone_to_segment: {len(self.bone_to_segment)} mappings")
        print(f"    novel_poses: {self.novel_poses.shape if self.novel_poses is not None else None}")
        print(f"    db_poses: {self.db_poses.shape if self.db_poses is not None else None}")

        try:
            if self.mode == 'edge_wise':
                return self._transform_edge_wise(
                    novel_smpl_patches, matched_smpl, matched_ssdr,
                    matched_indices, part_name, part_joints, part_bones,
                    n_smpl_channels, n_ssdr_channels, patch_size
                )
            else:
                return self._transform_kabsch(
                    novel_smpl_patches, matched_smpl, matched_ssdr,
                    matched_indices, part_joints, part_bones,
                    n_smpl_channels, n_ssdr_channels, patch_size
                )
        except Exception as e:
            if self.debug:
                print(f"  [WARN] Skeleton transform failed: {e}")
                import traceback
                traceback.print_exc()
            return matched_combined

    def _transform_edge_wise(
        self,
        novel_smpl_patches, matched_smpl, matched_ssdr,
        matched_indices, part_name, part_joints, part_bones,
        n_smpl_channels, n_ssdr_channels, patch_size
    ) -> torch.Tensor:
        """
        Edge-wise alignment using FK global transforms.

        Logic:
        1. Use FK to compute global rigid transform for each joint (DB and Novel)
        2. For each segment, compute relative transform = Novel @ inv(DB)
        3. Apply relative transform to bones bound to that segment

        This correctly handles cumulative rotations along the kinematic chain.
        """

        n_patches = matched_indices.shape[0]
        center_offset = patch_size // 2
        device = novel_smpl_patches.device

        # Get kinematic chain for this part
        chain = KINEMATIC_CHAINS.get(part_name, [])
        if not chain:
            if self.debug:
                print(f"    No kinematic chain defined for {part_name}")
            return torch.cat([matched_smpl, matched_ssdr], dim=-1)

        # Check if we have segment mapping
        if not self.bone_to_segment:
            if self.debug:
                print(f"    [WARN] No bone-to-segment mapping, skipping transform")
            return torch.cat([matched_smpl, matched_ssdr], dim=-1)

        # Check if we have poses for FK
        use_fk = (self.novel_poses is not None and self.db_poses is not None)
        if self.debug:
            print(f"    Using FK global transforms: {use_fk}")

        T_novel = self.novel_joint_positions.shape[0]
        T_db = self.db_joint_positions.shape[0]

        # Process each patch
        transformed_patches = []
        total_angle = 0.0
        n_transformed = 0

        for patch_idx in range(n_patches):
            novel_frame = min(patch_idx + center_offset, T_novel - 1)
            db_frame = min(matched_indices[patch_idx].item() + center_offset, T_db - 1)

            # Get joint positions for this frame
            novel_joints = self.novel_joint_positions[novel_frame]  # (24, 3)
            db_joints = self.db_joint_positions[db_frame]  # (24, 3)

            # Convert to numpy for FK computation
            novel_joints_np = novel_joints.cpu().numpy() if torch.is_tensor(novel_joints) else novel_joints
            db_joints_np = db_joints.cpu().numpy() if torch.is_tensor(db_joints) else db_joints

            # Compute segment relative transforms
            # IMPORTANT: Compute for ALL segments that bones in this part might be bound to
            segment_relative_transforms = {}

            if use_fk:
                # Use FK to get global transforms
                novel_pose = self.novel_poses[novel_frame]
                db_pose = self.db_poses[db_frame]

                novel_global = compute_global_transforms_from_poses(novel_pose, novel_joints_np)
                db_global = compute_global_transforms_from_poses(db_pose, db_joints_np)

                # Compute transforms for ALL segments that bones in this part are bound to
                segments_needed = set(chain)
                for bone_idx in part_bones:
                    seg = self.bone_to_segment.get(bone_idx)
                    if seg:
                        segments_needed.add(seg)

                for segment in segments_needed:
                    T_rel = compute_segment_relative_transform(db_global, novel_global, segment)
                    segment_relative_transforms[segment] = torch.from_numpy(T_rel.astype(np.float32)).to(device)

                    # Compute angle for logging
                    R_rel = T_rel[:3, :3]
                    angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
                    total_angle += angle
            else:
                # Fallback: use direction-based rotation (less accurate)
                novel_joints_t = novel_joints.to(device) if torch.is_tensor(novel_joints) else torch.from_numpy(novel_joints).to(device)
                db_joints_t = db_joints.to(device) if torch.is_tensor(db_joints) else torch.from_numpy(db_joints).to(device)

                # Compute transforms for ALL segments that bones in this part are bound to
                segments_needed = set(chain)
                for bone_idx in part_bones:
                    seg = self.bone_to_segment.get(bone_idx)
                    if seg:
                        segments_needed.add(seg)

                for parent_idx, child_idx in segments_needed:
                    db_bone = db_joints_t[child_idx] - db_joints_t[parent_idx]
                    novel_bone = novel_joints_t[child_idx] - novel_joints_t[parent_idx]

                    R = rotation_from_vectors(db_bone.unsqueeze(0), novel_bone.unsqueeze(0)).squeeze(0)

                    # Build 4x4 transform (rotation only, translation handled separately)
                    T_rel = torch.eye(4, device=device)
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = novel_joints_t[parent_idx] - R @ db_joints_t[parent_idx]

                    segment_relative_transforms[(parent_idx, child_idx)] = T_rel

                    angle = rotation_angle_from_matrix(R.unsqueeze(0)).item()
                    total_angle += angle

            # Get SSDR transforms for this patch
            patch_ssdr = matched_ssdr[0, patch_idx]  # (ssdr_ch * P,)
            patch_ssdr = patch_ssdr.reshape(n_ssdr_channels, patch_size)

            n_part_bones = (n_ssdr_channels - 3) // 16
            ssdr_transforms_flat = patch_ssdr[:n_part_bones * 16, :]
            ssdr_trans = patch_ssdr[n_part_bones * 16:, :]

            # Reshape to (P, n_bones, 4, 4)
            ssdr_transforms = ssdr_transforms_flat.T.reshape(patch_size, n_part_bones, 4, 4)
            transformed_ssdr = ssdr_transforms.clone()

            # IMPORTANT: part_bones contains GLOBAL bone indices
            # But in the extracted part data, bones are indexed locally (0, 1, 2, ...)
            # We need to map global indices to local indices
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(part_bones)}

            # Apply relative transform to each bone based on its bound segment
            #
            # CORRECT LOGIC:
            # 1. SSDR transforms are relative to rest pose: v' = T_ssdr @ v_rest
            # 2. DB pose bone position: bone_db = T_ssdr_db @ bone_rest
            # 3. We want: bone_novel = T_rel @ bone_db = T_rel @ (T_ssdr_db @ bone_rest)
            # 4. So: T_ssdr_new = T_rel @ T_ssdr_db (full 4x4 multiplication)
            #
            for global_bone_idx in part_bones:
                # Map global bone index to local index in this part
                local_bone_idx = global_to_local.get(global_bone_idx)
                if local_bone_idx is None or local_bone_idx >= n_part_bones:
                    continue

                # Look up which segment this bone belongs to (using global index)
                segment = self.bone_to_segment.get(global_bone_idx, None)
                if segment is None or segment not in segment_relative_transforms:
                    continue

                T_rel = segment_relative_transforms[segment]  # (4, 4)

                # Apply to all frames in patch: T_new = T_rel @ T_old
                for frame_idx in range(patch_size):
                    T_old = transformed_ssdr[frame_idx, local_bone_idx]  # (4, 4)
                    T_new = T_rel @ T_old
                    transformed_ssdr[frame_idx, local_bone_idx] = T_new

                n_transformed += 1

            # Reshape back
            transformed_flat = transformed_ssdr.reshape(patch_size, n_part_bones * 16).T
            transformed_full = torch.cat([transformed_flat, ssdr_trans], dim=0)
            transformed_patches.append(transformed_full.reshape(-1))

        # Always print transform stats for debugging
        n_segments = len(chain)
        avg_angle = total_angle / (n_patches * n_segments) if n_segments else 0
        print(f"    [SkeletonTransform] Part: {part_name}, {n_transformed} bones transformed, avg {np.degrees(avg_angle):.1f} deg/segment")

        transformed_ssdr = torch.stack(transformed_patches, dim=0).unsqueeze(0)

        # Blend if needed
        if self.transform_weight < 1.0:
            blended_ssdr = (
                self.transform_weight * transformed_ssdr +
                (1 - self.transform_weight) * matched_ssdr
            )
        else:
            blended_ssdr = transformed_ssdr

        return torch.cat([matched_smpl, blended_ssdr], dim=-1)

    def _transform_kabsch(
        self,
        novel_smpl_patches, matched_smpl, matched_ssdr,
        matched_indices, part_joints, part_bones,
        n_smpl_channels, n_ssdr_channels, patch_size
    ) -> torch.Tensor:
        """Original Kabsch-based global rotation."""

        n_patches = matched_indices.shape[0]
        center_offset = patch_size // 2
        device = novel_smpl_patches.device

        T_novel = self.novel_joint_positions.shape[0]
        T_db = self.db_joint_positions.shape[0]

        novel_positions_list = []
        db_positions_list = []

        for patch_idx in range(n_patches):
            novel_frame = min(patch_idx + center_offset, T_novel - 1)
            db_frame = min(matched_indices[patch_idx].item() + center_offset, T_db - 1)

            novel_pos = self.novel_joint_positions[novel_frame, part_joints, :]
            db_pos = self.db_joint_positions[db_frame, part_joints, :]

            novel_positions_list.append(novel_pos)
            db_positions_list.append(db_pos)

        novel_positions = torch.stack(novel_positions_list, dim=0).to(device)
        db_positions = torch.stack(db_positions_list, dim=0).to(device)

        # Kabsch
        P_centered = novel_positions - novel_positions.mean(dim=1, keepdim=True)
        Q_centered = db_positions - db_positions.mean(dim=1, keepdim=True)
        H = torch.bmm(Q_centered.transpose(1, 2), P_centered)
        U, S, Vt = torch.linalg.svd(H)
        R_rel = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

        det = torch.det(R_rel)
        mask = det < 0
        if mask.any():
            Vt_fixed = Vt.clone()
            Vt_fixed[mask, -1, :] *= -1
            R_rel[mask] = torch.bmm(Vt_fixed[mask].transpose(1, 2), U[mask].transpose(1, 2))

        avg_angle = rotation_angle_from_matrix(R_rel).mean().item()
        if self.debug:
            print(f"    Kabsch rotation: {np.degrees(avg_angle):.1f} deg")

        if avg_angle < 0.01:
            return torch.cat([matched_smpl, matched_ssdr], dim=-1)

        # Apply to SSDR
        db_ssdr_reshaped = matched_ssdr[0].reshape(n_patches, n_ssdr_channels, patch_size)
        n_part_bones = (n_ssdr_channels - 3) // 16
        ssdr_transforms_flat = db_ssdr_reshaped[:, :n_part_bones * 16, :]
        ssdr_trans = db_ssdr_reshaped[:, n_part_bones * 16:, :]

        ssdr_transforms = ssdr_transforms_flat.permute(0, 2, 1).reshape(-1, n_part_bones, 4, 4)

        R_rel_expanded = R_rel.unsqueeze(1).expand(-1, patch_size, -1, -1)
        R_rel_expanded = R_rel_expanded.reshape(n_patches * patch_size, 3, 3)

        # IMPORTANT: part_bones contains GLOBAL bone indices
        # But in the extracted part data, bones are indexed locally (0, 1, 2, ...)
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(part_bones)}

        # Apply rotation to bones
        transformed = ssdr_transforms.clone()
        for global_bone_idx in part_bones:
            local_bone_idx = global_to_local.get(global_bone_idx)
            if local_bone_idx is None or local_bone_idx >= n_part_bones:
                continue
            R_bone = transformed[:, local_bone_idx, :3, :3]
            R_new = torch.bmm(R_rel_expanded, R_bone)
            transformed[:, local_bone_idx, :3, :3] = R_new

        transformed_flat = transformed.reshape(n_patches, patch_size, n_part_bones, 16)
        transformed_flat = transformed_flat.permute(0, 2, 3, 1).reshape(n_patches, n_part_bones * 16, patch_size)
        transformed_full = torch.cat([transformed_flat, ssdr_trans], dim=1)
        transformed_ssdr = transformed_full.reshape(n_patches, -1).unsqueeze(0)

        if self.transform_weight < 1.0:
            blended_ssdr = (
                self.transform_weight * transformed_ssdr +
                (1 - self.transform_weight) * matched_ssdr
            )
        else:
            blended_ssdr = transformed_ssdr

        return torch.cat([matched_smpl, blended_ssdr], dim=-1)
