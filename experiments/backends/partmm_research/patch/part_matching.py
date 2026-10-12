"""Part-based Patch Matching"""
import torch
import torch.nn.functional as F


class PartBasedMatcher:
    """
    Part-level patch matching and blending
    """

    def __init__(self, alpha=0.85, device='cpu'):
        """
        Args:
            alpha: blending factor (0=database, 1=novel)
        """
        self.alpha = alpha
        self.device = device

    def match_and_blend(self, novel_smpl_part, part_database, num_neighbors=3):
        """
        为单个part做matching和blending

        Args:
            novel_smpl_part: [1, n_smpl_ch, n_frames] novel SMPL motion for this part
            part_database: dict from PartBasedPatchDatabase
            num_neighbors: number of nearest neighbors

        Returns:
            blended_ssdr_part: [1, n_ssdr_ch, n_frames]
        """
        patch_size = self._infer_patch_size(part_database)
        n_frames = novel_smpl_part.shape[2]
        n_patches = n_frames - patch_size + 1

        db_smpl = part_database['smpl_patches']  # [n_db_patches, smpl_ch*patch_size]
        db_ssdr = part_database['ssdr_patches']  # [n_db_patches, ssdr_ch*patch_size]

        n_smpl_ch = novel_smpl_part.shape[1]
        n_ssdr_ch = db_ssdr.shape[1] // patch_size

        # Extract novel patches
        novel_patches = []
        for i in range(n_patches):
            patch = novel_smpl_part[:, :, i:i+patch_size]
            novel_patches.append(patch.reshape(1, -1))
        novel_patches = torch.cat(novel_patches, dim=0)  # [n_patches, smpl_ch*patch_size]

        # Find nearest neighbors for each patch
        result_patches = []

        for i in range(n_patches):
            query_patch = novel_patches[i:i+1]  # [1, smpl_ch*patch_size]

            # Compute distances
            distances = torch.cdist(query_patch, db_smpl)  # [1, n_db_patches]

            # Find k nearest
            _, indices = torch.topk(distances, num_neighbors, dim=1, largest=False)

            # Get corresponding SSDR patches
            neighbor_ssdr = db_ssdr[indices[0]]  # [k, ssdr_ch*patch_size]

            # Blend: simple average
            blended_patch = neighbor_ssdr.mean(dim=0, keepdim=True)  # [1, ssdr_ch*patch_size]

            result_patches.append(blended_patch)

        # Reconstruct motion from patches (simple average overlapping regions)
        result_motion = self._reconstruct_from_patches(
            result_patches, n_frames, n_ssdr_ch, patch_size
        )

        return result_motion

    def _infer_patch_size(self, part_database):
        """从database推断patch size"""
        smpl_patches = part_database['smpl_patches']
        n_smpl_ch_flat = smpl_patches.shape[1]

        # 简单推断: 假设n_smpl_ch是合理的
        # 实际上应该从part_database存储的信息获取
        # 这里用一个heuristic
        for patch_size in range(3, 20):
            if n_smpl_ch_flat % patch_size == 0:
                return patch_size
        return 7  # default

    def _reconstruct_from_patches(self, patches, n_frames, n_channels, patch_size):
        """
        从patches重建完整motion (平均重叠区域)

        Args:
            patches: list of [1, n_ch*patch_size]
            n_frames: total frames
            n_channels: channels per frame
            patch_size: patch size

        Returns:
            motion: [1, n_channels, n_frames]
        """
        motion = torch.zeros(1, n_channels, n_frames, device=self.device)
        counts = torch.zeros(1, n_channels, n_frames, device=self.device)

        for i, patch in enumerate(patches):
            patch_reshaped = patch.reshape(n_channels, patch_size)
            motion[:, :, i:i+patch_size] += patch_reshaped
            counts[:, :, i:i+patch_size] += 1

        # Average overlapping regions
        motion = motion / (counts + 1e-8)

        return motion
