"""Main SMPL to SSDR Matcher"""
import torch
import torch.nn.functional as F

from .._thirdparty_nearest_neighbor import combine_patches
from ..skeletal.binding import SkeletalBinding
from ..patch.database import build_combined_database
from ..patch.matching import combined_refinement
from ..pyramid.generator import generate_pyramid, upsample_smpl_with_slerp
from ..reconstruction.lbs import reconstruct_vertices, result_to_lbs_transforms
from ..math import blend_transformations, se3_interpolate


class SMPLToSSDRMatcher:
    """SMPL to SSDR motion retargeting matcher"""

    def __init__(self, patch_size=7, stride=1, alpha=0.85, device='cuda:0',
                 rotation_weight=1.0, translation_weight=1.0):
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.device = torch.device(device)
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

        self.skeletal_binding = None
        self.combined_target_patches = None
        self.n_smpl_channels = None
        self.n_ssdr_channels = None

    def build_skeletal_binding(self, smpl_motions, ssdr_motions, ssdr_rest_vertices, ssdr_skinning_weights):
        """Build SMPL-SSDR skeletal binding"""
        binder = SkeletalBinding(device=self.device, k_nearest=4)
        self.skeletal_binding = binder.build(
            smpl_motions[0],
            ssdr_motions[0],
            ssdr_rest_vertices,
            ssdr_skinning_weights
        )

    def build_patch_database(self, smpl_motions, ssdr_motions):
        """Build combined patch database"""
        self.combined_target_patches, self.n_smpl_channels, self.n_ssdr_channels = build_combined_database(
            smpl_motions, ssdr_motions, self.patch_size, self.stride, self.device
        )
        print(f"Built database: {self.combined_target_patches.shape[1]} patches")

    def smpl_to_ssdr_via_skeleton(self, smpl_data):
        """Initialize SSDR from SMPL via skeletal binding"""
        batch_size, n_channels, n_frames = smpl_data.shape
        n_ssdr_bones = self.skeletal_binding['n_ssdr_bones']
        k_nearest = self.skeletal_binding['k_nearest']

        binding_indices = self.skeletal_binding['indices']
        binding_weights = self.skeletal_binding['weights']
        local_transforms = self.skeletal_binding['local_transforms']
        ssdr_rest_inv = torch.inverse(self.skeletal_binding['ssdr_rest_transforms'])

        ssdr_data = torch.zeros(batch_size, n_ssdr_bones * 16 + 3, n_frames, device=self.device)

        db_root_rotation = self.skeletal_binding['database_root_rotation']
        db_root_translation = self.skeletal_binding['database_root_translation']

        for frame_idx in range(n_frames):
            smpl_frame = smpl_data[:, :, frame_idx]
            smpl_frame_aligned = smpl_frame.clone()
            smpl_frame_aligned[0, :4] = db_root_rotation
            smpl_frame_aligned[0, -3:] = db_root_translation

            from ..skeletal.smpl_fk import compute_smpl_joint_transforms
            smpl_joint_positions, smpl_joint_rotations = compute_smpl_joint_transforms(
                smpl_frame_aligned, device=self.device
            )

            smpl_global_transforms = torch.eye(4, device=self.device).unsqueeze(0).repeat(24, 1, 1)
            smpl_global_transforms[:, :3, :3] = smpl_joint_rotations
            smpl_global_transforms[:, :3, 3] = smpl_joint_positions

            parent_transforms = smpl_global_transforms[binding_indices]
            candidate_transforms = torch.matmul(parent_transforms, local_transforms)

            ssdr_global_transforms = torch.zeros(n_ssdr_bones, 4, 4, device=self.device)
            for bone_idx in range(n_ssdr_bones):
                # Weighted sum of candidate transforms
                # candidate_transforms[bone_idx]: (k, 4, 4)
                # binding_weights[bone_idx]: (k,)
                weights = binding_weights[bone_idx].unsqueeze(-1).unsqueeze(-1)  # (k, 1, 1)
                ssdr_global_transforms[bone_idx] = (candidate_transforms[bone_idx] * weights).sum(dim=0)

            ssdr_relative_transforms = torch.matmul(ssdr_global_transforms, ssdr_rest_inv)
            ssdr_data[0, :-3, frame_idx] = ssdr_relative_transforms.reshape(-1)
            ssdr_data[0, -3:, frame_idx] = db_root_translation

        return ssdr_data

    def _align_ssdr_to_query_root(self, ssdr_data, query_smpl_data, db_root_rotation, db_root_translation):
        """
        将SSDR结果从database root坐标系对齐到query root坐标系

        Args:
            ssdr_data: [1, n_ssdr_ch, n_frames] - SSDR motion在database root坐标系下
            query_smpl_data: [1, n_smpl_ch, n_frames] - query的SMPL数据
            db_root_rotation: [4] - database root的四元数
            db_root_translation: [3] - database root的平移

        Returns:
            aligned_ssdr_data: [1, n_ssdr_ch, n_frames] - 对齐到query root的SSDR数据
        """
        from ..skeletal.part_binding import PartBasedSkeletalBinding

        batch_size, n_ssdr_ch, n_frames = ssdr_data.shape
        n_bones = (n_ssdr_ch - 3) // 16

        # 准备database root [7]: [qw, qx, qy, qz, tx, ty, tz]
        database_root = torch.cat([db_root_rotation, db_root_translation])

        aligned_ssdr_data = torch.zeros_like(ssdr_data)

        for frame_idx in range(n_frames):
            # 获取query root for this frame
            query_root = torch.cat([
                query_smpl_data[0, :4, frame_idx],  # rotation
                query_smpl_data[0, -3:, frame_idx]  # translation
            ])

            # 提取SSDR bone transforms [n_bones, 4, 4]
            ssdr_frame = ssdr_data[0, :-3, frame_idx]  # [n_bones * 16]
            ssdr_transforms = ssdr_frame.reshape(n_bones, 4, 4)

            # 应用root对齐
            aligned_transforms = PartBasedSkeletalBinding.align_ssdr_bones_for_blending(
                ssdr_transforms,
                database_root,
                query_root
            )

            # 存储对齐后的transforms
            aligned_ssdr_data[0, :-3, frame_idx] = aligned_transforms.reshape(-1)

            # 对齐global translation
            ssdr_global_trans = ssdr_data[0, -3:, frame_idx]
            aligned_global_trans = PartBasedSkeletalBinding.align_ssdr_to_query_root(
                ssdr_global_trans.unsqueeze(0),
                database_root,
                query_root
            ).squeeze(0)
            aligned_ssdr_data[0, -3:, frame_idx] = aligned_global_trans

        return aligned_ssdr_data

    def _weighted_cdist(self, x1, x2):
        """Combined SMPL+SSDR geodesic distance"""
        from ..math.combined_distance import combined_smpl_ssdr_distance

        return combined_smpl_ssdr_distance(
            x1, x2,
            self.n_smpl_channels,
            self.n_ssdr_channels,
            self.patch_size,
            self.rotation_weight
        )

    def build_combined_target_database(self, smpl_motions, ssdr_motions):
        """Unsupported legacy combined-target database path.

        Not on the part-based retargeting path: it depends on
        ``core/adaptive_mask.py``, which was not vendored in this snapshot
        (see THIRD_PARTY_NOTICES.md).
        """
        raise NotImplementedError(
            "SMPLToSSDRMatcher.build_combined_target_database is the legacy "
            "combined-target path and is unsupported in this packaged "
            "research snapshot: it requires core/adaptive_mask.py, which was "
            "not vendored. Use the part-based pipeline instead "
            "(PartBasedSMPLToSSDRMatcher.build_part_databases)."
        )

    def motion_blending(self, query_smpl_data, num_steps=10, matching_alpha=0.3, verbose=False,
                       init_ssdr=None, random_mask_bones=0, mask_seed=None, refine_smpl=False,
                       init_mode='skeleton', noise_sigma=0.0, use_adaptive_mask=False, use_smpl_mask=True,
                       align_to_query_root=False, skeleton_transformer=None, part_name=None):
        """Motion blending: SMPL -> SSDR

        Args:
            align_to_query_root: 是否将SSDR结果从database root对齐到query root（默认False，因为数据已在提取时对齐）
            skeleton_transformer: Optional SkeletonGuidedMatcher for pose-guided SSDR transformation
            part_name: Body part name (required if skeleton_transformer is provided)
        """
        db_root_rotation = self.skeletal_binding['database_root_rotation']
        db_root_translation = self.skeletal_binding['database_root_translation']

        query_smpl_aligned = query_smpl_data.clone()
        query_smpl_aligned[:, :4, :] = db_root_rotation.unsqueeze(1)
        query_smpl_aligned[:, -3:, :] = db_root_translation.unsqueeze(1)

        synthesized_smpl = query_smpl_aligned.clone()

        if init_mode == 'upsampled' and init_ssdr is not None:
            synthesized_ssdr = init_ssdr
        else:
            synthesized_ssdr = self.smpl_to_ssdr_via_skeleton(query_smpl_data)

        refinement_losses = []
        last_nnf = None  # Store last matching indices for visualization
        for step in range(num_steps):
            refined_combined_patches, distances, current_patches, nnf = combined_refinement(
                synthesized_smpl,
                synthesized_ssdr,
                self.combined_target_patches,
                self.patch_size,
                self.stride,
                self.alpha,
                matching_alpha,
                self._weighted_cdist,
                use_smpl_mask,
                self.n_smpl_channels,
                self.n_ssdr_channels,
                self.device,
                skeleton_transformer=skeleton_transformer,
                part_name=part_name
            )
            last_nnf = nnf  # Keep track of the final matching

            combined_shape = (1, self.n_smpl_channels + self.n_ssdr_channels, query_smpl_data.shape[-1])

            if not use_smpl_mask:
                # smpl_only mode: confidence-weighted blending
                # confidence = 1 / (1 + dist), higher for better matches
                confidence = 1.0 / (1.0 + distances + 1e-8)  # [n_patches]

                # Expand confidence to match patch dimensions: [1, n_patches, 1]
                confidence_weights = confidence.unsqueeze(0).unsqueeze(-1)  # [1, n_patches, 1]

                # Blend: matched_patch * confidence + current_patch * (1 - confidence)
                blended_patches = (
                    refined_combined_patches * confidence_weights +
                    current_patches * (1.0 - confidence_weights)
                )

                fully_refined_combined = combine_patches(
                    combined_shape,
                    blended_patches,
                    self.patch_size,
                    self.stride,
                    loop=False
                )
            else:
                # full mode: direct replacement (original behavior)
                fully_refined_combined = combine_patches(
                    combined_shape,
                    refined_combined_patches,
                    self.patch_size,
                    self.stride,
                    loop=False
                )

            refined_smpl = fully_refined_combined[:, :self.n_smpl_channels, :]
            refined_ssdr = fully_refined_combined[:, self.n_smpl_channels:, :]

            synthesized_ssdr = refined_ssdr

            refinement_losses.append(distances.mean().item())

            if verbose and step % 2 == 0:
                avg_conf = confidence.mean().item() if not use_smpl_mask else 1.0
                print(f"    Step {step:2d}: loss={refinement_losses[-1]:.6f}, avg_conf={avg_conf:.3f}")

        # 对齐SSDR结果到query root坐标系
        if align_to_query_root:
            synthesized_ssdr = self._align_ssdr_to_query_root(
                synthesized_ssdr,
                query_smpl_data,
                db_root_rotation,
                db_root_translation
            )

        losses = {
            'refinement_losses': refinement_losses,
            'matched_indices': last_nnf  # Matching indices for visualization
        }
        return synthesized_ssdr, synthesized_smpl, losses

    def generate(self, novel_smpl_motion, coarse_ratio=0.2, pyr_factor=0.75,
                num_steps=10, matching_alpha=0.3, verbose=True):
        """Generate SSDR motion from novel SMPL motion"""
        novel_data = novel_smpl_motion.sample().to(self.device)

        novel_pyramid, lengths = generate_pyramid(novel_data, coarse_ratio, pyr_factor)

        print(f"\n{'='*70}")
        print(f"Generating SSDR motion - {len(lengths)} pyramid levels")
        print(f"{'='*70}")

        synthesized_ssdr = None
        synthesized_smpl = None

        for lvl, target_length in enumerate(lengths):
            print(f"[Level {lvl}/{len(lengths)-1}] {target_length} frames")

            if lvl > 0:
                prev_length = synthesized_ssdr.shape[-1]
                n_bones = (self.n_ssdr_channels - 3) // 16

                ssdr_transforms_flat = synthesized_ssdr[:, :-3, :]
                ssdr_transforms_flat = ssdr_transforms_flat.squeeze(0).permute(1, 0)
                ssdr_transforms = ssdr_transforms_flat.reshape(prev_length, n_bones, 4, 4)

                upsampled_transforms = se3_interpolate(ssdr_transforms, target_length)
                upsampled_flat = upsampled_transforms.reshape(target_length, n_bones * 16)
                upsampled_flat = upsampled_flat.permute(1, 0).unsqueeze(0)

                ssdr_trans = synthesized_ssdr[:, -3:, :]
                upsampled_trans = F.interpolate(ssdr_trans, size=target_length, mode="linear", align_corners=False)

                init_ssdr = torch.cat([upsampled_flat, upsampled_trans], dim=1)
                current_smpl_data = novel_pyramid[lvl]
                current_init_mode = 'upsampled'
            else:
                current_smpl_data = novel_pyramid[lvl]
                init_ssdr = None
                current_init_mode = 'skeleton'

            synthesized_ssdr, synthesized_smpl, losses = self.motion_blending(
                current_smpl_data,
                num_steps=num_steps,
                matching_alpha=matching_alpha,
                verbose=verbose,
                init_ssdr=init_ssdr,
                init_mode=current_init_mode
            )

            if verbose:
                print(f"  ✓ Loss: {losses['refinement_losses'][0]:.6f} -> {losses['refinement_losses'][-1]:.6f}\n")

        print(f"{'='*70}\n")
        return synthesized_ssdr.detach(), synthesized_smpl.detach()
