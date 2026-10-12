"""Patch database construction"""
import torch
from tqdm import tqdm

from .._thirdparty_nearest_neighbor import extract_patches


def build_combined_database(smpl_motions, ssdr_motions, patch_size, stride=1, device='cuda:0'):
    """
    Build combined SMPL+SSDR patch database

    Args:
        smpl_motions: list of SMPLMotionData
        ssdr_motions: list of SSDRMotionData
        patch_size: patch size
        stride: patch stride
        device: device

    Returns:
        combined_patches: [1, n_patches, n_channels * patch_size]
        n_smpl_channels: int
        n_ssdr_channels: int
    """
    combined_patches_list = []

    for smpl_motion, ssdr_motion in tqdm(zip(smpl_motions, ssdr_motions), total=len(smpl_motions)):
        smpl_data = smpl_motion.sample().to(device)
        ssdr_data = ssdr_motion.sample().to(device)

        combined_data = torch.cat([smpl_data, ssdr_data], dim=1)
        combined_patches = extract_patches(combined_data, patch_size, stride, loop=False)
        combined_patches_list.append(combined_patches)

    combined_patches = torch.cat(combined_patches_list, dim=1)
    n_smpl_channels = smpl_data.shape[1]
    n_ssdr_channels = ssdr_data.shape[1]

    return combined_patches, n_smpl_channels, n_ssdr_channels
