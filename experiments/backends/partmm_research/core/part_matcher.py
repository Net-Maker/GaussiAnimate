"""Part-based Motion Retargeting Orchestrator"""
import torch
import torch.nn.functional as F
from .part_motion_data import PartBasedSMPLMotionData, PartBasedSSDRMotionData
from ..skeletal.part_binding import PartBasedSkeletalBinding
from ..patch.part_database import PartBasedPatchDatabase
from ..patch.part_matching import PartBasedMatcher
from ..skeletal.skeleton_transform import SkeletonGuidedMatcher


class PartBasedSMPLToSSDRMatcher:
    """
    Part-based SMPL to SSDR Motion Retargeting

    完整流程:
    1. Build part-based skeletal binding
    2. Build part-based patch databases
    3. For novel motion: match and blend each part independently
    4. Merge all parts back to full motion
    """

    def __init__(self, patch_size=7, alpha=0.85, matching_mode='full', device='cpu',
                 skeleton_guided=False, skeleton_transform_weight=1.0,
                 skeleton_mode='edge_wise', skeleton_debug=False,
                 skeleton_temperature=0.5, skeleton_chain_decay=0.7,
                 debug_binding=False, gender='female', is_cloth3d=True,
                 use_motion_binding=True, motion_weight=0.5, distance_weight=0.5):
        """
        Args:
            patch_size: Patch size for motion matching
            alpha: Bidirectional matching weight
            matching_mode: 'smpl_only' or 'full'
            device: torch device
            skeleton_guided: Enable skeleton-guided SSDR transformation
            skeleton_transform_weight: Blending weight for skeleton transform (0=off, 1=full)
            skeleton_mode: 'kabsch' (global) or 'edge_wise' (per-bone alignment)
            skeleton_debug: Print debug info for skeleton transform
            skeleton_temperature: (unused, kept for compatibility)
            skeleton_chain_decay: (unused, kept for compatibility)
            debug_binding: Visualize SMPL joints vs SSDR bones for debugging
            gender: 'female' or 'male' for SMPL model
            is_cloth3d: if True, subtract root joint position (Cloth3D centered at pelvis)
            use_motion_binding: if True, use motion similarity in binding (in addition to distance)
            motion_weight: weight for motion similarity in binding score
            distance_weight: weight for distance in binding score
        """
        self.patch_size = patch_size
        self.alpha = alpha
        self.matching_mode = matching_mode
        self.device = torch.device(device)
        self.skeleton_guided = skeleton_guided
        self.skeleton_transform_weight = skeleton_transform_weight
        self.skeleton_mode = skeleton_mode
        self.skeleton_debug = skeleton_debug
        self.debug_binding = debug_binding
        self.gender = gender
        self.is_cloth3d = is_cloth3d
        self.use_motion_binding = use_motion_binding
        self.motion_weight = motion_weight
        self.distance_weight = distance_weight

        self.skeletal_binder = None
        self.part_binding_data = None
        self.part_databases = None

        self.target_smpl_motion = None
        self.target_ssdr_motion = None

        self.skeleton_transformer = None

        print(f"  Matching mode: {matching_mode}")
        if matching_mode == 'smpl_only':
            print("  → Using SMPL-only matching (ignoring SSDR in distance calculation)")
        else:
            print("  → Using full SMPL+SSDR matching")

        if skeleton_guided:
            print(f"  → Skeleton-guided transform enabled (weight={skeleton_transform_weight}, mode={skeleton_mode})")

    def build_part_binding(self, smpl_motion, ssdr_motion, ssdr_rest_vertices, ssdr_skinning_weights):
        """Step 1: Build part-based skeletal binding"""
        from ..utils.mask import SMPL_BODY_PARTS

        self.skeletal_binder = PartBasedSkeletalBinding(
            device=self.device,
            debug_visualize=self.debug_binding,
            gender=self.gender,
            is_cloth3d=self.is_cloth3d,
            use_motion_similarity=self.use_motion_binding,
            motion_weight=self.motion_weight,
            distance_weight=self.distance_weight
        )

        self.part_binding_data = self.skeletal_binder.build(
            smpl_motion,
            ssdr_motion,
            ssdr_rest_vertices,
            ssdr_skinning_weights
        )

        # Create skeleton-guided transformer if enabled
        if self.skeleton_guided:
            part_to_bones = self.part_binding_data['part_to_bones']
            self.skeleton_transformer = SkeletonGuidedMatcher(
                part_to_joints=SMPL_BODY_PARTS,
                part_to_bones=part_to_bones,
                enabled=True,
                transform_weight=self.skeleton_transform_weight,
                mode=self.skeleton_mode,
                debug=self.skeleton_debug
            )

            # Set bone-to-segment mapping (more accurate than bone-to-joint)
            bone_to_segment = self.part_binding_data.get('bone_to_segment', None)
            if bone_to_segment:
                self.skeleton_transformer.set_bone_segment_mapping(bone_to_segment)
                print(f"  → Set bone-to-segment mapping for {len(bone_to_segment)} bones")

            # Set bone positions (influence centers from weights)
            # CRITICAL: SSDR transforms store RELATIVE displacement, not absolute positions
            # We need actual bone positions to compute correct skeleton-induced deltas
            bone_positions = self.part_binding_data.get('ssdr_rest_bones', None)
            if bone_positions is not None:
                self.skeleton_transformer.set_bone_positions(bone_positions)
                print(f"  → Set bone positions for {len(bone_positions)} bones")

            print(f"  → Created SkeletonGuidedMatcher for {len(part_to_bones)} parts (mode={self.skeleton_mode})")

        return self.part_binding_data

    def build_part_databases(self, smpl_motion, ssdr_motion):
        """Step 2: Build part-based patch databases"""
        self.target_smpl_motion = smpl_motion
        self.target_ssdr_motion = ssdr_motion

        # 创建part-based motion data
        part_to_bones = self.part_binding_data['part_to_bones']

        part_smpl = PartBasedSMPLMotionData(smpl_motion)
        part_ssdr = PartBasedSSDRMotionData(ssdr_motion, part_to_bones)

        # 构建databases
        db_builder = PartBasedPatchDatabase(
            patch_size=self.patch_size,
            device=self.device
        )

        self.part_databases = db_builder.build(part_smpl, part_ssdr)

        return self.part_databases

    def retarget_motion(self, novel_smpl_motion, num_steps=5, matching_alpha=0.3,
                       coarse_ratio=0.4, pyr_factor=0.75,
                       novel_joint_positions=None, db_joint_positions=None,
                       novel_poses=None, db_poses=None,
                       **kwargs):
        """
        Step 3: Part-based retargeting with full motion_blending pipeline

        为每个part独立运行完整的pyramid + motion_blending

        Args:
            novel_smpl_motion: SMPLMotionData
            num_steps: optimization steps per pyramid level
            matching_alpha: SMPL guidance weight
            coarse_ratio: pyramid coarsest level ratio
            pyr_factor: pyramid scale factor
            novel_joint_positions: (T, 24, 3) joint positions for Kabsch (optional)
            db_joint_positions: (T, 24, 3) database joint positions for Kabsch (optional)
            novel_poses: (T, 72) SMPL poses for FK-based transform (optional)
            db_poses: (T, 72) database SMPL poses for FK-based transform (optional)
            **kwargs: additional args for motion_blending

        Returns:
            merged_ssdr_motion: [1, n_ssdr_ch, n_frames]
        """
        from ..core.matcher import SMPLToSSDRMatcher
        from ..pyramid.part_generator import generate_part_pyramid, se3_interpolate_part
        from ..utils.mask import SMPL_BODY_PARTS

        # Set joint positions for skeleton-guided transform
        if self.skeleton_transformer is not None:
            if novel_joint_positions is not None and db_joint_positions is not None:
                self.skeleton_transformer.set_joint_positions(
                    novel_joint_positions, db_joint_positions
                )
                # Set poses for FK-based global transform
                if novel_poses is not None and db_poses is not None:
                    self.skeleton_transformer.set_poses(novel_poses, db_poses)
            elif self.skeleton_debug:
                print("  [WARN] Joint positions not provided, skeleton transform may not work")

        print(f"\n{'='*70}")
        print(f"  Part-based Motion Retargeting (Full Pipeline)")
        print(f"{'='*70}")
        print(f"  Strategy: Independent pyramid optimization per part")
        print(f"  Pyramid: coarse={coarse_ratio}, factor={pyr_factor}")
        print(f"  Steps: {num_steps}, α={matching_alpha}")
        print(f"{'='*70}\n")

        part_to_bones = self.part_binding_data['part_to_bones']
        novel_part_smpl = PartBasedSMPLMotionData(novel_smpl_motion)

        # 为每个part创建独立的matcher并运行完整流程
        part_results = {}
        part_matching_info = {}  # Store matching info for visualization

        for part_name, part_db in self.part_databases.items():
            if part_name not in part_to_bones or not part_to_bones[part_name]:
                continue

            bone_indices = part_to_bones[part_name]
            if len(bone_indices) == 0:
                continue

            print(f"\n[Part: {part_name}] {len(bone_indices)} bones")

            # 1. 创建part-specific matcher
            part_matcher = SMPLToSSDRMatcher(
                patch_size=self.patch_size,
                alpha=self.alpha,
                device=self.device
            )

            # 2. 设置part的database (始终包含SMPL+SSDR，matching_mode只影响距离计算)
            smpl_patches = part_db['smpl_patches']  # [n_patches, smpl_ch*P]
            ssdr_patches = part_db['ssdr_patches']  # [n_patches, ssdr_ch*P]

            # 始终合并SMPL+SSDR patches
            combined_patches = torch.cat([smpl_patches, ssdr_patches], dim=1)  # [n_patches, total_ch*P]

            # 添加batch维度: [n_patches, ch*P] -> [1, n_patches, ch*P]
            part_matcher.combined_target_patches = combined_patches.unsqueeze(0)
            part_matcher.target_ssdr_patches = ssdr_patches.unsqueeze(0)
            part_matcher.n_smpl_channels = smpl_patches.shape[1] // self.patch_size
            part_matcher.n_ssdr_channels = ssdr_patches.shape[1] // self.patch_size

            # Part-specific weighted distance (支持可变joint数量和matching模式)
            def part_weighted_cdist(x1, x2):
                """
                x1, x2: [n, (n_smpl + n_ssdr) * P] - combined patches (flattened)

                matching_mode:
                - 'smpl_only': 只用SMPL部分计算距离
                - 'full': 用SMPL+SSDR联合计算距离
                """
                if self.matching_mode == 'smpl_only':
                    # 提取SMPL部分: x[:, :n_smpl*P]
                    n_smpl_total = part_matcher.n_smpl_channels * self.patch_size
                    smpl1 = x1[:, :n_smpl_total]
                    smpl2 = x2[:, :n_smpl_total]

                    # Reshape to [n, n_smpl, P]
                    n1, _ = smpl1.shape
                    n2, _ = smpl2.shape
                    smpl1_reshaped = smpl1.reshape(n1, part_matcher.n_smpl_channels, self.patch_size)
                    smpl2_reshaped = smpl2.reshape(n2, part_matcher.n_smpl_channels, self.patch_size)

                    # 只计算SMPL部分的距离
                    from ..math.distances import smpl_quaternion_distance
                    return smpl_quaternion_distance(
                        smpl1_reshaped, smpl2_reshaped,
                        n_joints=part_matcher.n_smpl_channels // 4,  # 4 channels per quaternion
                        patch_size=self.patch_size
                    )
                else:
                    # Full mode: 计算combined distance (输入已经是正确格式)
                    from ..math.combined_distance import combined_smpl_ssdr_distance
                    return combined_smpl_ssdr_distance(
                        x1, x2,
                        n_smpl_channels=part_matcher.n_smpl_channels,
                        n_ssdr_channels=part_matcher.n_ssdr_channels,
                        patch_size=self.patch_size,
                        rotation_weight=1.0
                    )
            part_matcher._weighted_cdist = part_weighted_cdist

            # 3. 设置skeletal binding (使用part_binding_info的完整格式)
            part_binding_info = self.part_binding_data['binding_per_part'][part_name]
            part_matcher.skeletal_binding = {
                **part_binding_info,  # 包含indices, weights, local_transforms, ssdr_rest_transforms等
                'database_root_rotation': self.part_binding_data['database_root_rotation'],
                'database_root_translation': self.part_binding_data['database_root_translation']
            }

            # 3. 提取这个part的novel SMPL motion
            novel_smpl_part = novel_part_smpl.get_part_motion(part_name, self.device)

            # 5. 计算这个part有多少joints
            part_joints = SMPL_BODY_PARTS.get(part_name, [])
            n_part_joints = len(part_joints)

            # 6. 生成part-specific pyramid
            part_pyramid, lengths = generate_part_pyramid(
                novel_smpl_part, n_part_joints, coarse_ratio, pyr_factor
            )

            print(f"  Pyramid: {' → '.join(map(str, lengths))} frames")

            # 7. Pyramid optimization（完整流程）
            synthesized_ssdr = None

            for lvl, target_length in enumerate(lengths):
                if lvl == 0:
                    # 第一层：SSDR初始化为0（motion_blending内部会做matching）
                    n_frames_init = part_pyramid[0].shape[-1]
                    n_bones = len(bone_indices)
                    init_ssdr = torch.zeros(1, n_bones * 16 + 3, n_frames_init, device=self.device)
                else:
                    # Upsample from previous level using part-aware upsampling
                    prev_length = synthesized_ssdr.shape[-1]
                    n_bones = len(bone_indices)

                    ssdr_flat = synthesized_ssdr[:, :-3, :].squeeze(0).permute(1, 0)
                    ssdr_transforms = ssdr_flat.reshape(prev_length, n_bones, 4, 4)

                    # Part-based SE3 interpolation
                    upsampled_transforms = se3_interpolate_part(ssdr_transforms, target_length)
                    upsampled_flat = upsampled_transforms.reshape(target_length, n_bones * 16)
                    upsampled_flat = upsampled_flat.permute(1, 0).unsqueeze(0)

                    # Linear interpolation for global position
                    ssdr_trans = synthesized_ssdr[:, -3:, :]
                    upsampled_trans = F.interpolate(
                        ssdr_trans, size=target_length, mode="linear", align_corners=False
                    )

                    init_ssdr = torch.cat([upsampled_flat, upsampled_trans], dim=1)

                # Motion blending（完整的优化流程）
                # smpl_only模式: 禁用mask，让distance_fn直接计算SMPL距离
                # full模式: 启用mask，通过matching_alpha控制SMPL vs SSDR权重
                synthesized_ssdr, _, losses = part_matcher.motion_blending(
                    part_pyramid[lvl],
                    num_steps=num_steps,
                    matching_alpha=matching_alpha,
                    init_ssdr=init_ssdr,
                    init_mode='upsampled',  # 总是用init_ssdr（第一层是0初始化）
                    use_smpl_mask=(self.matching_mode != 'smpl_only'),
                    verbose=True,
                    skeleton_transformer=self.skeleton_transformer,
                    part_name=part_name,
                    **kwargs
                )

                # Store matching info for the final level
                if lvl == len(lengths) - 1:
                    part_matching_info[part_name] = {
                        'matched_indices': losses.get('matched_indices'),
                        'database': part_matcher.combined_target_patches,
                        'n_smpl_channels': part_matcher.n_smpl_channels,
                        'n_ssdr_channels': part_matcher.n_ssdr_channels,
                        'bone_indices': bone_indices
                    }

            part_results[part_name] = synthesized_ssdr
            print(f"  ✓ Optimized: {synthesized_ssdr.shape}")

        # 6. 合并所有parts（对重叠区域取平均）
        print(f"\n  Merging {len(part_results)} parts (averaging overlaps)...")
        dummy_ssdr = PartBasedSSDRMotionData(self.target_ssdr_motion, part_to_bones)
        merged_ssdr = dummy_ssdr.merge_part_motions(part_results, self.device)

        print(f"  ✓ Final result: {merged_ssdr.shape}")
        print(f"{'='*70}\n")

        # Store matching info for visualization
        self.last_matching_info = {
            'part_results': part_results,
            'part_matching_info': part_matching_info,
            'part_to_bones': part_to_bones
        }

        return merged_ssdr
