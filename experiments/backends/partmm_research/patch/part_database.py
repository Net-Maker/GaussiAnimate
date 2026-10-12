"""Part-based Patch Database"""
import torch
from tqdm import tqdm


class PartBasedPatchDatabase:
    """
    为每个part构建独立的patch database
    """

    def __init__(self, patch_size=7, device='cpu'):
        self.patch_size = patch_size
        self.device = device
        self.databases = {}  # {part_name: part_database}

    def build(self, part_based_smpl, part_based_ssdr, parts_to_include=None):
        """
        为每个part构建patch database

        Args:
            part_based_smpl: PartBasedSMPLMotionData
            part_based_ssdr: PartBasedSSDRMotionData
            parts_to_include: list of part names or None (all parts)

        Returns:
            databases: dict {part_name: {'smpl_patches': ..., 'ssdr_patches': ...}}
        """
        if parts_to_include is None:
            parts_to_include = part_based_smpl.parts

        print(f"\n{'='*70}")
        print(f"  Building Part-based Patch Database (patch_size={self.patch_size})")
        print(f"{'='*70}")

        for part_name in tqdm(parts_to_include, desc="Building databases"):
            if part_name not in part_based_smpl.parts:
                continue
            if part_name not in part_based_ssdr.parts:
                continue

            # 获取这个part的motion data
            smpl_part_motion = part_based_smpl.get_part_motion(part_name, self.device)
            ssdr_part_motion = part_based_ssdr.get_part_motion(part_name, self.device)

            # 构建patch database
            part_db = self._build_part_database(
                smpl_part_motion,
                ssdr_part_motion,
                part_name
            )

            self.databases[part_name] = part_db

        # 打印统计
        print(f"\n  ✓ Built databases for {len(self.databases)} parts:")
        for part_name, db in self.databases.items():
            n_patches = db['smpl_patches'].shape[0]
            smpl_ch = db['smpl_patches'].shape[1]
            ssdr_ch = db['ssdr_patches'].shape[1]
            print(f"    {part_name:12s}: {n_patches:4d} patches "
                  f"(SMPL: {smpl_ch:3d}ch, SSDR: {ssdr_ch:3d}ch)")
        print(f"{'='*70}\n")

        return self.databases

    def _build_part_database(self, smpl_motion, ssdr_motion, part_name):
        """
        为单个part构建database

        Args:
            smpl_motion: [1, n_smpl_ch, n_frames]
            ssdr_motion: [1, n_ssdr_ch, n_frames]

        Returns:
            part_db: dict with smpl_patches, ssdr_patches
        """
        n_frames = smpl_motion.shape[2]
        n_patches = n_frames - self.patch_size + 1

        # Extract patches
        smpl_patches = []
        ssdr_patches = []

        for i in range(n_patches):
            smpl_patch = smpl_motion[:, :, i:i+self.patch_size]
            ssdr_patch = ssdr_motion[:, :, i:i+self.patch_size]

            smpl_patches.append(smpl_patch.reshape(1, -1))
            ssdr_patches.append(ssdr_patch.reshape(1, -1))

        smpl_patches = torch.cat(smpl_patches, dim=0)  # [n_patches, smpl_ch*patch_size]
        ssdr_patches = torch.cat(ssdr_patches, dim=0)  # [n_patches, ssdr_ch*patch_size]

        return {
            'smpl_patches': smpl_patches,
            'ssdr_patches': ssdr_patches,
            'n_patches': n_patches,
            'part_name': part_name
        }

    def get_database(self, part_name):
        """获取指定part的database"""
        return self.databases.get(part_name, None)
