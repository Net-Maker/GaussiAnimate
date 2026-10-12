"""Part-based Skeletal Binding"""
import torch
import numpy as np
from .smpl_fk import compute_smpl_joint_transforms
from ..utils.mask import SMPL_BODY_PARTS
from scipy.spatial.transform import Rotation as R


# ============================================================================
# SMPL Limb Segments: each segment is (parent_joint, child_joint)
# SSDR bones should bind to segments (rigid bodies), NOT to joints (points)
# ============================================================================
SMPL_LIMB_SEGMENTS = {
    'torso': [
        (0, 3),   # Pelvis -> Spine1
        (3, 6),   # Spine1 -> Spine2
        (6, 9),   # Spine2 -> Spine3
        (9, 12),  # Spine3 -> Neck
    ],
    'head': [
        (12, 15),  # Neck -> Head
    ],
    'left_arm': [
        (9, 13),   # Spine3 -> L_Collar
        (13, 16),  # L_Collar -> L_Shoulder
        (16, 18),  # L_Shoulder -> L_Elbow
        (18, 20),  # L_Elbow -> L_Wrist
        (20, 22),  # L_Wrist -> L_Hand
    ],
    'right_arm': [
        (9, 14),   # Spine3 -> R_Collar
        (14, 17),  # R_Collar -> R_Shoulder
        (17, 19),  # R_Shoulder -> R_Elbow
        (19, 21),  # R_Elbow -> R_Wrist
        (21, 23),  # R_Wrist -> R_Hand
    ],
    'left_leg': [
        (0, 1),    # Pelvis -> L_Hip
        (1, 4),    # L_Hip -> L_Knee
        (4, 7),    # L_Knee -> L_Ankle
        (7, 10),   # L_Ankle -> L_Foot
    ],
    'right_leg': [
        (0, 2),    # Pelvis -> R_Hip
        (2, 5),    # R_Hip -> R_Knee
        (5, 8),    # R_Knee -> R_Ankle
        (8, 11),   # R_Ankle -> R_Foot
    ],
}

# Flatten all segments for global lookup
ALL_LIMB_SEGMENTS = []
SEGMENT_TO_PART = {}
for part_name, segments in SMPL_LIMB_SEGMENTS.items():
    for seg in segments:
        ALL_LIMB_SEGMENTS.append(seg)
        SEGMENT_TO_PART[seg] = part_name


def point_to_segment_distance(point: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
    """
    Compute distance from point(s) to line segment(s).

    Args:
        point: (..., 3) point positions
        seg_start: (..., 3) segment start positions
        seg_end: (..., 3) segment end positions

    Returns:
        distance: (...) distances
    """
    # Vector from start to end
    seg_vec = seg_end - seg_start
    seg_len_sq = torch.sum(seg_vec ** 2, dim=-1, keepdim=True)

    # Vector from start to point
    pt_vec = point - seg_start

    # Project point onto line, clamp to [0, 1] for segment
    t = torch.sum(pt_vec * seg_vec, dim=-1, keepdim=True) / (seg_len_sq + 1e-8)
    t = torch.clamp(t, 0, 1)

    # Closest point on segment
    closest = seg_start + t * seg_vec

    # Distance from point to closest point
    dist = torch.norm(point - closest, dim=-1)
    return dist


def compute_motion_similarity(smpl_motion, ssdr_motion, ssdr_rest_vertices, ssdr_skinning_weights,
                              device='cpu', gender='female', is_cloth3d=False):
    """
    Compute motion similarity between SSDR bones and SMPL joints.

    Strategy (improved to favor "direct controllers"):
    1. Extract SMPL joint LOCAL rotations over time
    2. Extract SSDR bone rotations over time (from transforms)
    3. Compute rotation change correlation (favors joints that directly control the bone)

    Key insight:
    - Position/velocity correlation can't distinguish shoulder vs elbow for forearm
    - Both cause the forearm to move!
    - But ROTATION correlation favors the joint whose local rotation matches the bone's rotation

    Args:
        smpl_motion: SMPLMotionData
        ssdr_motion: SSDRMotionData
        ssdr_rest_vertices: (V, 3) rest pose vertices
        ssdr_skinning_weights: (B, V) skinning weights
        device: torch device
        gender: 'female' or 'male'
        is_cloth3d: if True, subtract root joint

    Returns:
        similarity_matrix: (n_bones, 24) motion similarity scores (higher = more similar)
    """
    device = torch.device(device)

    # Get full motion data
    smpl_data = smpl_motion.sample().to(device)  # [1, C, T]
    ssdr_data = ssdr_motion.sample().to(device)  # [1, C, T]

    n_frames = smpl_data.shape[2]

    # Prepare weights
    if hasattr(ssdr_skinning_weights, 'toarray'):
        weights_np = ssdr_skinning_weights.toarray()
    else:
        weights_np = np.array(ssdr_skinning_weights)
    weights = torch.from_numpy(weights_np).float().to(device)
    n_bones = weights.shape[0]

    rest_verts = torch.from_numpy(ssdr_rest_vertices).float().to(device)

    print(f"  Computing motion similarity ({n_frames} frames, {n_bones} bones)...")

    # =========================================================================
    # 1. Extract SMPL joint LOCAL rotations over time
    # =========================================================================
    # SMPL data format: [1, 24*4 + 3, T] where 24*4 = quaternions
    smpl_quats = smpl_data[:, :-3, :].squeeze(0)  # [96, T]
    smpl_quats = smpl_quats.permute(1, 0).reshape(n_frames, 24, 4)  # [T, 24, 4]

    # Convert quaternions to rotation matrices for comparison
    def quat_to_rotmat(q):
        """Convert quaternion [w,x,y,z] to 3x3 rotation matrix"""
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        R = torch.zeros(q.shape[:-1] + (3, 3), device=q.device)
        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - w*z)
        R[..., 0, 2] = 2*(x*z + w*y)
        R[..., 1, 0] = 2*(x*y + w*z)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - w*x)
        R[..., 2, 0] = 2*(x*z - w*y)
        R[..., 2, 1] = 2*(y*z + w*x)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)
        return R

    smpl_rotmats = quat_to_rotmat(smpl_quats)  # [T, 24, 3, 3]

    # =========================================================================
    # 2. Extract SSDR bone rotations over time
    # =========================================================================
    ssdr_transforms_flat = ssdr_data[:, :-3, :].squeeze(0).permute(1, 0)  # [T, n_bones*16]
    ssdr_transforms = ssdr_transforms_flat.reshape(n_frames, n_bones, 4, 4)  # [T, n_bones, 4, 4]
    ssdr_rotmats = ssdr_transforms[:, :, :3, :3]  # [T, n_bones, 3, 3]

    # =========================================================================
    # 3. Compute rotation CHANGE correlation
    # =========================================================================
    # Key insight: Compare how rotations CHANGE over time
    # R_delta(t) = R(t) @ R(t-1).T  (relative rotation between frames)

    def rotation_angle(R):
        """Extract rotation angle from rotation matrix"""
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        return torch.acos(cos_angle)  # [T-1]

    # Compute rotation changes for SMPL joints
    smpl_rot_changes = torch.zeros(n_frames - 1, 24, device=device)
    for t in range(n_frames - 1):
        for j in range(24):
            R_delta = smpl_rotmats[t+1, j] @ smpl_rotmats[t, j].T
            smpl_rot_changes[t, j] = rotation_angle(R_delta)

    # Compute rotation changes for SSDR bones
    ssdr_rot_changes = torch.zeros(n_frames - 1, n_bones, device=device)
    for t in range(n_frames - 1):
        for b in range(n_bones):
            R_delta = ssdr_rotmats[t+1, b] @ ssdr_rotmats[t, b].T
            ssdr_rot_changes[t, b] = rotation_angle(R_delta)

    # Normalize rotation changes
    smpl_rot_norm = smpl_rot_changes / (smpl_rot_changes.std(dim=0, keepdim=True) + 1e-8)
    ssdr_rot_norm = ssdr_rot_changes / (ssdr_rot_changes.std(dim=0, keepdim=True) + 1e-8)

    # Compute correlation of rotation changes
    rotation_correlation = torch.zeros(n_bones, 24, device=device)

    for bone_idx in range(n_bones):
        bone_rot = ssdr_rot_norm[:, bone_idx]  # [T-1]
        bone_rot_centered = bone_rot - bone_rot.mean()
        bone_std = bone_rot.std() + 1e-8

        for joint_idx in range(24):
            joint_rot = smpl_rot_norm[:, joint_idx]  # [T-1]
            joint_rot_centered = joint_rot - joint_rot.mean()
            joint_std = joint_rot.std() + 1e-8

            # Pearson correlation of rotation changes
            corr = (bone_rot_centered * joint_rot_centered).mean()
            corr = corr / (bone_std * joint_std)
            rotation_correlation[bone_idx, joint_idx] = corr

    # =========================================================================
    # 4. Also compute position-based motion similarity (for completeness)
    # =========================================================================
    # Extract positions for position-based correlation
    smpl_joint_trajectories = []
    for t in range(n_frames):
        smpl_frame = smpl_data[:, :, t]
        joint_positions, _ = compute_smpl_joint_transforms(
            smpl_frame, device=device, gender=gender, is_cloth3d=is_cloth3d
        )
        smpl_joint_trajectories.append(joint_positions)
    smpl_trajectories = torch.stack(smpl_joint_trajectories, dim=0)  # [T, 24, 3]

    # SSDR bone positions
    ssdr_bone_trajectories = torch.zeros(n_frames, n_bones, 3, device=device)
    for bone_idx in range(n_bones):
        max_weight_idx = torch.argmax(weights[bone_idx]).item()
        rest_pos = rest_verts[max_weight_idx]
        for t in range(n_frames):
            T_mat = ssdr_transforms[t, bone_idx]
            rest_pos_homo = torch.cat([rest_pos, torch.ones(1, device=device)])
            transformed_pos = (T_mat @ rest_pos_homo)[:3]
            ssdr_bone_trajectories[t, bone_idx] = transformed_pos

    # Velocity correlation
    smpl_velocities = smpl_trajectories[1:] - smpl_trajectories[:-1]
    ssdr_velocities = ssdr_bone_trajectories[1:] - ssdr_bone_trajectories[:-1]

    smpl_vel_norm = smpl_velocities / (torch.norm(smpl_velocities, dim=2, keepdim=True) + 1e-8)
    ssdr_vel_norm = ssdr_velocities / (torch.norm(ssdr_velocities, dim=2, keepdim=True) + 1e-8)

    velocity_similarity = torch.zeros(n_bones, 24, device=device)
    for bone_idx in range(n_bones):
        bone_vel = ssdr_vel_norm[:, bone_idx, :]
        for joint_idx in range(24):
            joint_vel = smpl_vel_norm[:, joint_idx, :]
            cos_sim = (bone_vel * joint_vel).sum(dim=1)
            velocity_similarity[bone_idx, joint_idx] = cos_sim.mean()

    # =========================================================================
    # 5. Combine: Rotation correlation + Velocity similarity
    # =========================================================================
    # Rotation correlation is key for distinguishing direct vs indirect control
    # Weight rotation more heavily
    combined_similarity = 0.7 * rotation_correlation + 0.3 * velocity_similarity

    print(f"    Rotation correlation range: [{rotation_correlation.min():.3f}, {rotation_correlation.max():.3f}]")
    print(f"    Velocity similarity range: [{velocity_similarity.min():.3f}, {velocity_similarity.max():.3f}]")

    return combined_similarity.cpu().numpy()


class PartBasedSkeletalBinding:
    """
    Part-based skeletal binding: 将SSDR bones按距离分配给SMPL parts

    策略: 简单距离分配
    1. 计算每个SSDR bone到所有SMPL joints的距离
    2. 找到最近的joint
    3. 将bone分配给该joint所属的part
    """

    def __init__(self, device='cpu', part_groups=None, debug_visualize=False,
                 gender='female', is_cloth3d=True, use_motion_similarity=True,
                 motion_weight=0.5, distance_weight=0.5):
        """
        Args:
            device: torch device
            part_groups: dict {part_name: [joint_indices]} 或 None (使用默认)
            debug_visualize: if True, show SMPL joints vs SSDR bones visualization
            gender: 'female' or 'male' for SMPL model
            is_cloth3d: if True, subtract root joint position (Cloth3D data centered at pelvis)
            use_motion_similarity: if True, use motion correlation in addition to distance
            motion_weight: weight for motion similarity in combined score (0-1)
            distance_weight: weight for distance in combined score (0-1)
        """
        self.device = torch.device(device)
        self.part_groups = part_groups if part_groups is not None else SMPL_BODY_PARTS
        self.debug_visualize = debug_visualize
        self.gender = gender
        self.is_cloth3d = is_cloth3d
        self.use_motion_similarity = use_motion_similarity
        self.motion_weight = motion_weight
        self.distance_weight = distance_weight

        # 创建joint到part的反向映射
        self.joint_to_part = {}
        for part_name, joint_indices in self.part_groups.items():
            for joint_idx in joint_indices:
                if joint_idx not in self.joint_to_part:
                    self.joint_to_part[joint_idx] = part_name

    def build(self, smpl_motion, ssdr_motion, ssdr_rest_vertices, ssdr_skinning_weights):
        """
        Build part-based skeletal binding

        Binding strategy:
        - Distance-based: assign bone to nearest joint
        - Motion-based: assign bone to joint with most similar motion
        - Combined: weighted combination of both

        Returns:
            part_binding_data: dict {
                'part_to_bones': {part_name: [bone_indices]},
                'bone_to_part': {bone_idx: part_name},
                'binding_per_part': {part_name: binding_data}
            }
        """
        binding_mode = "Distance + Motion" if self.use_motion_similarity else "Distance-only"
        print("\n" + "="*70)
        print(f"  Part-based Skeletal Binding ({binding_mode})")
        print("="*70)

        # Get rest pose
        smpl_data = smpl_motion.sample().to(self.device)
        ssdr_data = ssdr_motion.sample().to(self.device)

        smpl_frame0 = smpl_data[:, :, 0]
        ssdr_frame0 = ssdr_data[:, :, 0]

        # Compute SMPL joint positions
        smpl_joint_positions, smpl_joint_rotations = compute_smpl_joint_transforms(
            smpl_frame0, device=self.device, gender=self.gender, is_cloth3d=self.is_cloth3d
        )

        # Extract SSDR bone positions
        rest_vertices = torch.from_numpy(ssdr_rest_vertices).float().to(self.device)

        if hasattr(ssdr_skinning_weights, 'toarray'):
            skinning_weights_np = ssdr_skinning_weights.toarray()
        else:
            skinning_weights_np = ssdr_skinning_weights

        skinning_weights = torch.from_numpy(skinning_weights_np).float().to(self.device)

        n_bones = skinning_weights.shape[0]
        ssdr_bone_positions = torch.zeros(n_bones, 3, device=self.device)

        # Compute bone influence center as WEIGHTED AVERAGE of vertex positions
        # This is more accurate than using single max-weight vertex
        for bone_idx in range(n_bones):
            bone_weights = skinning_weights[bone_idx]  # (n_vertices,)
            w_sum = bone_weights.sum()

            if w_sum > 1e-8:
                # Weighted average of all influenced vertices
                weighted_pos = (bone_weights.unsqueeze(1) * rest_vertices).sum(dim=0) / w_sum
                ssdr_bone_positions[bone_idx] = weighted_pos
            else:
                # Fallback to max weight vertex if weights sum to ~0
                max_weight_idx = torch.argmax(bone_weights)
            ssdr_bone_positions[bone_idx] = rest_vertices[max_weight_idx]

        # =====================================================================
        # Compute distance to LIMB SEGMENTS (not joints!)
        # Each segment is a rigid body between two joints
        # =====================================================================
        print(f"\n  Computing bone-to-segment distances ({len(ALL_LIMB_SEGMENTS)} segments)...")

        # Distance matrix: [n_bones, n_segments]
        n_segments = len(ALL_LIMB_SEGMENTS)
        segment_distances = torch.zeros(n_bones, n_segments, device=self.device)

        for seg_idx, (parent_j, child_j) in enumerate(ALL_LIMB_SEGMENTS):
            seg_start = smpl_joint_positions[parent_j:parent_j+1].expand(n_bones, -1)
            seg_end = smpl_joint_positions[child_j:child_j+1].expand(n_bones, -1)
            segment_distances[:, seg_idx] = point_to_segment_distance(
                ssdr_bone_positions, seg_start, seg_end
            )

        # Convert to similarity (closer = higher score)
        max_dist = segment_distances.max()
        segment_scores = 1.0 - (segment_distances / (max_dist + 1e-8))
        segment_scores_np = segment_scores.cpu().numpy()

        # Also compute joint-based distance for motion similarity
        distance_matrix = torch.cdist(ssdr_bone_positions, smpl_joint_positions)  # [n_bones, 24]
        distance_scores_np = (1.0 - (distance_matrix / (distance_matrix.max() + 1e-8))).cpu().numpy()

        # =====================================================================
        # Compute motion similarity scores (if enabled)
        # =====================================================================
        if self.use_motion_similarity:
            motion_scores = compute_motion_similarity(
                smpl_motion, ssdr_motion, ssdr_rest_vertices, ssdr_skinning_weights,
                device=self.device, gender=self.gender, is_cloth3d=self.is_cloth3d
            )  # [n_bones, 24], range [-1, 1]
            motion_scores = (motion_scores + 1) / 2.0
            print(f"  Binding weights: distance={self.distance_weight}, motion={self.motion_weight}")
        else:
            motion_scores = None

        # =====================================================================
        # Assignment: bind each bone to nearest LIMB SEGMENT
        # =====================================================================
        part_to_bones = {part: [] for part in self.part_groups.keys()}
        bone_to_parts = {}
        bone_to_best_joint = {}  # Keep for backward compatibility
        bone_to_segment = {}     # NEW: maps bone_idx -> (parent_joint, child_joint)
        unassigned_bones = []

        for bone_idx in range(n_bones):
            # Primary: find nearest segment by distance
            seg_scores = segment_scores_np[bone_idx]  # [n_segments]

            # Distance-only best segment (for comparison)
            dist_only_seg_idx = np.argmax(seg_scores)
            dist_only_segment = ALL_LIMB_SEGMENTS[dist_only_seg_idx]

            # If motion similarity enabled, combine with child joint's motion score
            if motion_scores is not None:
                combined_seg_scores = np.zeros(n_segments)
                for seg_idx, (parent_j, child_j) in enumerate(ALL_LIMB_SEGMENTS):
                    # Use child joint's motion score (child controls the segment)
                    motion_score = motion_scores[bone_idx, child_j]
                    combined_seg_scores[seg_idx] = (
                        self.distance_weight * seg_scores[seg_idx] +
                        self.motion_weight * motion_score
                    )
                best_seg_idx = np.argmax(combined_seg_scores)
            else:
                best_seg_idx = dist_only_seg_idx

            best_segment = ALL_LIMB_SEGMENTS[best_seg_idx]
            bone_to_segment[bone_idx] = best_segment

            # Also store child joint for backward compatibility
            bone_to_best_joint[bone_idx] = best_segment[1]  # Child joint

            # Find which part owns this segment
            best_part = SEGMENT_TO_PART.get(best_segment, None)

            # Assign to part
            assigned_parts = []
            if best_part and best_part in part_to_bones:
                part_to_bones[best_part].append(bone_idx)
                assigned_parts.append(best_part)

            if assigned_parts:
                bone_to_parts[bone_idx] = assigned_parts
            else:
                unassigned_bones.append(bone_idx)

            # Log if motion changed the assignment
            if self.use_motion_similarity and best_segment != dist_only_segment:
                print(f"    Bone {bone_idx}: {dist_only_segment} -> {best_segment} (motion-adjusted)")

        # Statistics
        overlapping_bones = [b for b, parts in bone_to_parts.items() if len(parts) > 1]

        print(f"\n  ✓ Assigned {n_bones} SSDR bones to {len(self.part_groups)} parts:")
        for part_name in self.part_groups.keys():
            n_assigned = len(part_to_bones[part_name])
            print(f"    {part_name:12s}: {n_assigned:3d} bones")

        print(f"\n  ✓ Overlapping bones: {len(overlapping_bones)} (将在合并时平均)")
        if unassigned_bones:
            print(f"    {'unassigned':12s}: {len(unassigned_bones):3d} bones")

        # 为每个part构建独立的binding
        binding_per_part = {}

        for part_name, bone_indices in part_to_bones.items():
            if not bone_indices:
                continue

            # 获取这个part的SMPL joints
            part_joint_indices = self.part_groups[part_name]

            # 为这个part构建简单的binding (最近邻)
            part_binding = self._build_part_binding(
                bone_indices,
                part_joint_indices,
                ssdr_bone_positions,
                smpl_joint_positions,
                smpl_joint_rotations,
                ssdr_frame0
            )

            binding_per_part[part_name] = part_binding

        print("="*70 + "\n")

        # 整合binding数据
        part_binding_data = {
            'part_to_bones': part_to_bones,
            'bone_to_parts': bone_to_parts,  # 记录每个bone属于哪些parts
            'bone_to_best_joint': bone_to_best_joint,  # 记录每个bone绑定到哪个joint (向后兼容)
            'bone_to_segment': bone_to_segment,  # NEW: bone_idx -> (parent_joint, child_joint)
            'binding_per_part': binding_per_part,
            'smpl_rest_joints': smpl_joint_positions,
            'ssdr_rest_bones': ssdr_bone_positions,
            'database_root_rotation': smpl_frame0[0, :4].clone(),
            'database_root_translation': smpl_frame0[0, -3:].clone(),
            'use_motion_similarity': self.use_motion_similarity,
        }

        # Print segment binding summary
        print(f"\n  Segment binding summary:")
        segment_counts = {}
        for bone_idx, seg in bone_to_segment.items():
            if seg not in segment_counts:
                segment_counts[seg] = 0
            segment_counts[seg] += 1
        for seg, count in sorted(segment_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    Segment {seg[0]:2d}->{seg[1]:2d}: {count} bones")

        return part_binding_data

    def _build_part_binding(self, bone_indices, joint_indices, ssdr_bone_positions,
                           smpl_joint_positions, smpl_joint_rotations, ssdr_frame0):
        """
        为单个part构建binding (兼容multi-parent格式)

        Returns:
            part_binding: dict - 和SkeletalBinding返回相同格式
        """
        n_bones_in_part = len(bone_indices)
        k_nearest = 4  # 保持和原来一致

        # 兼容multi-parent格式: [n_bones, k]
        binding_indices = torch.zeros(n_bones_in_part, k_nearest, dtype=torch.long, device=self.device)
        binding_weights = torch.zeros(n_bones_in_part, k_nearest, device=self.device)
        local_transforms = torch.zeros(n_bones_in_part, k_nearest, 4, 4, device=self.device)
        ssdr_rest_transforms = torch.zeros(n_bones_in_part, 4, 4, device=self.device)

        ssdr_transforms_all = ssdr_frame0[:, :-3].reshape(-1, 4, 4)

        for i, bone_idx in enumerate(bone_indices):
            bone_pos = ssdr_bone_positions[bone_idx:bone_idx+1]

            # 只考虑这个part内的joints
            part_joint_pos = smpl_joint_positions[joint_indices]
            distances = torch.cdist(bone_pos, part_joint_pos).squeeze(0)

            # Find k nearest (or all if less than k)
            k = min(k_nearest, len(joint_indices))
            k_dists, k_local_indices = torch.topk(distances, k, largest=False)
            k_global_indices = torch.tensor([joint_indices[idx] for idx in k_local_indices],
                                           dtype=torch.long, device=self.device)

            # Gaussian weights
            sigma = k_dists.median() if k_dists.median() > 1e-6 else 1.0
            weights = torch.exp(-k_dists**2 / (2 * sigma**2))
            weights = weights / weights.sum()

            # Fill k_nearest slots
            binding_indices[i, :k] = k_global_indices
            binding_weights[i, :k] = weights
            if k < k_nearest:  # Pad
                binding_indices[i, k:] = k_global_indices[0]
                binding_weights[i, k:] = 0

            # Compute local transforms for each parent
            T_child = torch.eye(4, device=self.device)
            T_child[:3, :3] = ssdr_transforms_all[bone_idx, :3, :3]
            T_child[:3, 3] = ssdr_bone_positions[bone_idx]
            ssdr_rest_transforms[i] = T_child

            for j in range(k):
                parent_joint = k_global_indices[j].item()
                T_parent = torch.eye(4, device=self.device)
                T_parent[:3, :3] = smpl_joint_rotations[parent_joint]
                T_parent[:3, 3] = smpl_joint_positions[parent_joint]

                local_transforms[i, j] = torch.inverse(T_parent) @ T_child

            # Replicate first transform for unused slots
            if k < k_nearest:
                for j in range(k, k_nearest):
                    local_transforms[i, j] = local_transforms[i, 0]

        return {
            'bone_indices': bone_indices,
            'indices': binding_indices,
            'weights': binding_weights,
            'local_transforms': local_transforms,
            'ssdr_rest_transforms': ssdr_rest_transforms,
            'k_nearest': k_nearest,
            'n_ssdr_bones': n_bones_in_part
        }

    @staticmethod
    def compute_root_alignment_transform(database_root, query_root):
        """
        计算从database root到query root的相对变换
        用于将SSDR结果从database坐标系对齐到query坐标系

        Args:
            database_root: [7] or torch.Tensor - [qw, qx, qy, qz, tx, ty, tz] 存储在binding中的database root
            query_root: [7] or torch.Tensor - [qw, qx, qy, qz, tx, ty, tz] 当前query的root

        Returns:
            transform_matrix: [4, 4] - 相对变换矩阵
            relative_rotation: [3, 3] - 相对旋转矩阵
            relative_translation: [3] - 相对平移向量
        """
        # 转换为numpy
        if torch.is_tensor(database_root):
            database_root = database_root.cpu().numpy()
        if torch.is_tensor(query_root):
            query_root = query_root.cpu().numpy()

        # 提取旋转和平移
        db_quat = database_root[:4]  # [qw, qx, qy, qz]
        db_trans = database_root[4:]  # [tx, ty, tz]

        q_quat = query_root[:4]
        q_trans = query_root[4:]

        # 四元数转旋转矩阵 (scipy需要[x,y,z,w]格式)
        db_rot = R.from_quat([db_quat[1], db_quat[2], db_quat[3], db_quat[0]]).as_matrix()
        q_rot = R.from_quat([q_quat[1], q_quat[2], q_quat[3], q_quat[0]]).as_matrix()

        # 计算相对旋转: R_relative = R_query * R_database^T
        relative_rotation = q_rot @ db_rot.T

        # 计算相对平移: t_relative = t_query - R_relative * t_database
        relative_translation = q_trans - relative_rotation @ db_trans

        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = relative_rotation
        transform_matrix[:3, 3] = relative_translation

        return transform_matrix, relative_rotation, relative_translation

    @staticmethod
    def align_ssdr_to_query_root(ssdr_vertices, database_root, query_root):
        """
        将SSDR顶点从database root坐标系对齐到query root坐标系

        Args:
            ssdr_vertices: [N, 3] numpy or torch - SSDR顶点位置
            database_root: [7] - database的root参数
            query_root: [7] - query的root参数

        Returns:
            aligned_vertices: [N, 3] - 对齐后的顶点位置（保持输入类型）
        """
        transform_matrix, rel_rot, rel_trans = PartBasedSkeletalBinding.compute_root_alignment_transform(
            database_root, query_root
        )

        # 转换为torch tensor如果需要
        if torch.is_tensor(ssdr_vertices):
            device = ssdr_vertices.device
            rel_rot = torch.from_numpy(rel_rot).float().to(device)
            rel_trans = torch.from_numpy(rel_trans).float().to(device)

            # 应用变换: v' = R * v + t
            aligned_vertices = ssdr_vertices @ rel_rot.T + rel_trans
        else:
            # numpy版本
            aligned_vertices = ssdr_vertices @ rel_rot.T + rel_trans

        return aligned_vertices

    @staticmethod
    def align_ssdr_bones_for_blending(ssdr_bone_transforms, database_root, query_root):
        """
        将SSDR bone transforms从database root坐标系对齐到query root坐标系
        用于blending时的对齐

        Args:
            ssdr_bone_transforms: [N, 4, 4] numpy or torch - SSDR bone的变换矩阵
            database_root: [7] - database的root参数
            query_root: [7] - query的root参数

        Returns:
            aligned_transforms: [N, 4, 4] - 对齐后的变换矩阵（保持输入类型）
        """
        transform_matrix, _, _ = PartBasedSkeletalBinding.compute_root_alignment_transform(
            database_root, query_root
        )

        is_torch = torch.is_tensor(ssdr_bone_transforms)

        # 转换为torch tensor如果需要
        if is_torch:
            device = ssdr_bone_transforms.device
            transform_matrix = torch.from_numpy(transform_matrix).float().to(device)

        # 应用变换: T' = T_align * T_bone
        # 这会将bone从database坐标系转换到query坐标系
        n_bones = ssdr_bone_transforms.shape[0]
        aligned_transforms = torch.zeros_like(ssdr_bone_transforms) if is_torch else np.zeros_like(ssdr_bone_transforms)

        for i in range(n_bones):
            aligned_transforms[i] = transform_matrix @ ssdr_bone_transforms[i]

        return aligned_transforms
