"""
Motion data wrappers for SMPL and SSDR

Provides unified interface for motion representation
"""

import torch
from .._thirdparty_motion import MotionData
from ..math import axis_angle_to_quaternion


class SMPLMotionData:
    """SMPL motion data wrapper"""

    def __init__(self, smpl_data, repr="quat", use_velo=True, keep_up_pos=True, up_axis="Y_UP"):
        """
        Args:
            smpl_data: dict with 'poses' and 'trans'
                      poses: (n_frames, 72) axis-angle
                      trans: (n_frames, 3) global translation
        """
        if isinstance(smpl_data, dict):
            smpl_poses = smpl_data['poses']
            smpl_trans = smpl_data['trans']
        else:
            smpl_poses = smpl_data
            n_frames = smpl_poses.shape[0] if smpl_poses.dim() > 1 else 1
            smpl_trans = torch.zeros(n_frames, 3, device=smpl_poses.device, dtype=smpl_poses.dtype)

        if smpl_poses.dim() == 3:
            smpl_poses = smpl_poses.reshape(smpl_poses.shape[0], -1)

        # Convert axis-angle to quaternion
        n_frames = smpl_poses.shape[0]
        n_joints = smpl_poses.shape[1] // 3

        axis_angles = smpl_poses.reshape(n_frames, n_joints, 3)
        quaternions = axis_angle_to_quaternion(axis_angles)
        quaternion_data = quaternions.reshape(n_frames, -1)

        motion_data_with_pos = torch.cat([quaternion_data, smpl_trans], dim=1)

        print(f"SMPL Motion: poses {smpl_poses.shape} + trans {smpl_trans.shape} -> {motion_data_with_pos.shape} (use_velo={use_velo})")

        motion_tensor = motion_data_with_pos.permute(1, 0).unsqueeze(0)

        self.motion_data = MotionData(
            motion_tensor,
            repr=repr,
            use_velo=use_velo,
            keep_up_pos=keep_up_pos,
            up_axis=up_axis,
            padding_last=False,
            contact_id=None
        )

    def sample(self, size=None, slerp=False):
        return self.motion_data.sample(size, slerp)

    def __len__(self):
        return len(self.motion_data)


class SSDRMotionData:
    """SSDR motion data wrapper"""

    def __init__(self, ssdr_data, repr="quat", use_velo=True, keep_up_pos=True, up_axis="Y_UP"):
        """
        Args:
            ssdr_data: dict with 'transforms' and 'trans'
                      transforms: (n_frames, n_features) flattened LBS transforms
                      trans: (n_frames, 3) global translation
        """
        if isinstance(ssdr_data, dict):
            ssdr_transforms = ssdr_data['transforms']
            ssdr_trans = ssdr_data['trans']
        else:
            ssdr_transforms = ssdr_data
            n_frames = ssdr_transforms.shape[0] if ssdr_transforms.dim() > 1 else 1
            ssdr_trans = torch.zeros(n_frames, 3, device=ssdr_transforms.device, dtype=ssdr_transforms.dtype)

        if ssdr_transforms.dim() == 3:
            ssdr_transforms = ssdr_transforms.reshape(ssdr_transforms.shape[0], -1)

        global_pos = ssdr_trans
        motion_data_with_pos = torch.cat([ssdr_transforms, global_pos], dim=1)

        print(f"SSDR Motion: transforms {ssdr_transforms.shape} + trans {ssdr_trans.shape} -> {motion_data_with_pos.shape} (use_velo={use_velo})")

        motion_tensor = motion_data_with_pos.permute(1, 0).unsqueeze(0)

        self.motion_data = MotionData(
            motion_tensor,
            repr=repr,
            use_velo=use_velo,
            keep_up_pos=keep_up_pos,
            up_axis=up_axis,
            padding_last=False,
            contact_id=None
        )

    def sample(self, size=None, slerp=False):
        return self.motion_data.sample(size, slerp)

    def __len__(self):
        return len(self.motion_data)
