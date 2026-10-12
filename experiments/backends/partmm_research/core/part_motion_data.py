"""Part-based Motion Data"""
import torch
from .motion_data import SMPLMotionData, SSDRMotionData
from ..utils.mask import SMPL_BODY_PARTS
from ..utils.smpl_mask import create_smpl_part_channel_masks, create_ssdr_part_channel_masks


class PartBasedSMPLMotionData:
    """
    Part-based SMPL Motion Data

    将SMPL motion按body part分组，支持part-level操作
    """

    def __init__(self, smpl_motion_data, parts=None):
        """
        Args:
            smpl_motion_data: SMPLMotionData实例或dict
            parts: list of part names或None (使用默认SMPL_BODY_PARTS)
        """
        if isinstance(smpl_motion_data, dict):
            self.full_motion = SMPLMotionData(smpl_motion_data)
        else:
            self.full_motion = smpl_motion_data

        # 使用默认parts或自定义parts
        if parts is None:
            self.part_groups = SMPL_BODY_PARTS
            self.parts = list(SMPL_BODY_PARTS.keys())
        else:
            # 如果parts是list，从SMPL_BODY_PARTS筛选
            if isinstance(parts, list):
                self.part_groups = {k: SMPL_BODY_PARTS[k] for k in parts if k in SMPL_BODY_PARTS}
                self.parts = parts
            else:
                self.part_groups = parts
                self.parts = list(parts.keys())

        # 为每个part创建channel masks (SMPL用quaternion格式)
        self.part_masks = create_smpl_part_channel_masks(self.part_groups)

        print(f"PartBasedSMPLMotionData: {len(self.parts)} parts")
        for part_name in self.parts:
            mask = self.part_masks[part_name]
            n_active = int(mask.sum().item())
            print(f"  - {part_name:12s}: {n_active:3d} channels")

    def get_part_motion(self, part_name, device='cpu'):
        """
        获取特定part的motion data

        Args:
            part_name: part名称
            device: torch device

        Returns:
            part_motion: [1, n_part_channels, n_frames]
        """
        if part_name not in self.part_masks:
            raise ValueError(f"Unknown part: {part_name}")

        # 获取完整motion
        full_data = self.full_motion.sample().to(device)  # [1, 99, frames]

        # 应用part mask
        mask = self.part_masks[part_name].to(device)  # [99]

        # 提取part channels
        part_data = full_data * mask.unsqueeze(0).unsqueeze(2)

        # 只保留非零channels（压缩）
        active_indices = torch.where(mask > 0)[0]
        part_data_compact = part_data[:, active_indices, :]

        return part_data_compact

    def merge_part_motions(self, part_motions_dict, device='cpu'):
        """
        合并多个part motions回到完整motion

        Args:
            part_motions_dict: {part_name: part_motion_tensor}
            device: torch device

        Returns:
            merged_motion: [1, 99, n_frames]
        """
        # 获取frame数
        first_part = next(iter(part_motions_dict.values()))
        n_frames = first_part.shape[-1]

        # 初始化完整motion
        full_data = torch.zeros(1, 99, n_frames, device=device)

        # 合并每个part
        for part_name, part_motion in part_motions_dict.items():
            if part_name not in self.part_masks:
                continue

            mask = self.part_masks[part_name].to(device)
            active_indices = torch.where(mask > 0)[0]

            # 将part数据写回对应channels
            full_data[:, active_indices, :] = part_motion

        return full_data

    def get_all_parts(self, device='cpu'):
        """
        获取所有parts的motion数据

        Returns:
            dict: {part_name: part_motion}
        """
        return {
            part_name: self.get_part_motion(part_name, device)
            for part_name in self.parts
        }


class PartBasedSSDRMotionData:
    """
    Part-based SSDR Motion Data

    将SSDR motion按对应的SMPL parts分组
    """

    def __init__(self, ssdr_motion_data, smpl_part_to_ssdr_bones_mapping):
        """
        Args:
            ssdr_motion_data: SSDRMotionData实例或dict
            smpl_part_to_ssdr_bones_mapping: dict {part_name: [bone_indices]}
        """
        if isinstance(ssdr_motion_data, dict):
            self.full_motion = SSDRMotionData(ssdr_motion_data)
        else:
            self.full_motion = ssdr_motion_data

        self.part_to_bones = smpl_part_to_ssdr_bones_mapping
        self.parts = list(smpl_part_to_ssdr_bones_mapping.keys())

        print(f"PartBasedSSDRMotionData: {len(self.parts)} parts")
        for part_name in self.parts:
            n_bones = int(len(self.part_to_bones[part_name]))
            print(f"  - {part_name:12s}: {n_bones:3d} bones")

    def get_part_motion(self, part_name, device='cpu'):
        """
        获取特定part的SSDR motion

        Returns:
            part_motion: [1, n_part_bone_channels + 3, n_frames]
        """
        if part_name not in self.part_to_bones:
            raise ValueError(f"Unknown part: {part_name}")

        full_data = self.full_motion.sample().to(device)  # [1, 611, frames]
        bone_indices = self.part_to_bones[part_name]

        # 每个bone有16个channels (4x4 matrix)
        # 提取这些bone的channels + global position (最后3个)
        channel_indices = []
        for bone_idx in bone_indices:
            start_ch = bone_idx * 16
            channel_indices.extend(range(start_ch, start_ch + 16))

        # 添加global position channels (最后3个)
        channel_indices.extend([full_data.shape[1] - 3, full_data.shape[1] - 2, full_data.shape[1] - 1])

        channel_indices = torch.tensor(channel_indices, device=device)
        part_data = full_data[:, channel_indices, :]

        return part_data

    def merge_part_motions(self, part_motions_dict, device='cpu'):
        """
        合并part motions回到完整SSDR motion

        Args:
            part_motions_dict: {part_name: part_motion_tensor}

        Returns:
            merged_motion: [1, 611, n_frames]
        """
        first_part = next(iter(part_motions_dict.values()))
        n_frames = first_part.shape[-1]

        # 获取总bone数（从完整motion推断）
        full_sample = self.full_motion.sample()
        n_total_channels = full_sample.shape[1]

        full_data = torch.zeros(1, n_total_channels, n_frames, device=device)

        # 累加器和计数器，用于平均重叠区域
        bone_data_sum = torch.zeros(1, n_total_channels - 3, n_frames, device=device)  # 不含global pos
        bone_data_count = torch.zeros(1, n_total_channels - 3, 1, device=device)
        global_pos_sum = torch.zeros(1, 3, n_frames, device=device)
        global_pos_count = 0

        for part_name, part_motion in part_motions_dict.items():
            bone_indices = self.part_to_bones[part_name]

            # 写入bone channels（累加，后续会平均）
            for i, bone_idx in enumerate(bone_indices):
                start_ch_full = bone_idx * 16
                start_ch_part = i * 16
                bone_data_sum[:, start_ch_full:start_ch_full+16, :] += part_motion[:, start_ch_part:start_ch_part+16, :]
                bone_data_count[:, start_ch_full:start_ch_full+16, 0] += 1

            # Global position累加
            global_pos_sum += part_motion[:, -3:, :]
            global_pos_count += 1

        # 对重叠区域取平均
        full_data[:, :-3, :] = bone_data_sum / (bone_data_count + 1e-8)
        full_data[:, -3:, :] = global_pos_sum / global_pos_count

        return full_data

    def get_all_parts(self, device='cpu'):
        """获取所有parts的motion数据"""
        return {
            part_name: self.get_part_motion(part_name, device)
            for part_name in self.parts
        }
