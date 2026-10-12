"""Patch matching with optional skeleton-guided transformation"""
import torch

from .._thirdparty_nearest_neighbor import extract_patches, combine_patches, get_NNs_Dists


def combined_refinement(smpl_data, ssdr_data, target_patches, patch_size, stride,
                       alpha, matching_alpha, distance_fn, use_smpl_mask,
                       n_smpl_channels, n_ssdr_channels, device='cuda:0',
                       skeleton_transformer=None, part_name=None):
    """
    Refine using combined SMPL+SSDR matching with optional skeleton-guided transformation.

    use_smpl_mask控制距离计算时SMPL vs SSDR的权重：
    - True: 使用weighted mask (matching_alpha控制SMPL vs SSDR权重)
      - matching_alpha=1.0: 只看SMPL (SSDR权重=0)
      - matching_alpha=0.5: SMPL和SSDR各50%权重
      - matching_alpha=0.0: 只看SSDR (SMPL权重=0)

    - False: 不使用mask，直接用distance_fn计算
      - 适用于distance_fn本身已经选择性计算（如smpl_only模式）
      - 此时matching_alpha参数不起作用

    skeleton_transformer: Optional SkeletonGuidedMatcher for pose-guided SSDR transformation
    part_name: Body part name (required if skeleton_transformer is provided)

    Returns:
        refined_patches: matched patches (始终包含SMPL+SSDR)
        distances: matching distances (per-patch)
        combined_patches: current synthetic patches (for confidence blending)
        nnf: nearest neighbor indices
    """
    # 始终构建完整的 combined data
    combined_data = torch.cat([smpl_data, ssdr_data], dim=1)
    combined_patches = extract_patches(combined_data, patch_size, stride, loop=False)

    # Extract SMPL patches for skeleton-guided transform
    smpl_patches = extract_patches(smpl_data, patch_size, stride, loop=False)

    if use_smpl_mask:
        # 创建channel mask: SMPL权重高，SSDR权重低
        n_total_channels = n_smpl_channels + n_ssdr_channels
        channel_mask = torch.zeros(1, n_total_channels, 1, device=device)
        channel_mask[:, :n_smpl_channels, :] = 1.0
        channel_mask[:, n_smpl_channels:, :] = 0.0

        # matching_alpha: SMPL的权重，(1-matching_alpha): SSDR的权重
        mask = channel_mask.expand(-1, -1, patch_size)
        mask = matching_alpha * mask + (1 - matching_alpha) * (1 - mask)
        mask_patches = extract_patches(mask, patch_size, stride, loop=False)

        nnf, distances = get_NNs_Dists(
            distance_fn,
            combined_patches.squeeze(0) * mask_patches.squeeze(0),
            target_patches.squeeze(0) * mask_patches.squeeze(0),
            alpha
        )
    else:
        # 不使用mask，distance_fn自己决定如何计算距离
        nnf, distances = get_NNs_Dists(
            distance_fn,
            combined_patches.squeeze(0),
            target_patches.squeeze(0),
            alpha
        )

    # Apply skeleton-guided transformation if enabled
    if skeleton_transformer is not None and part_name is not None:
        # Transform matched SSDR patches using skeleton guidance
        refined_patches = skeleton_transformer.transform_matched_patches(
            smpl_patches,
            target_patches,
            nnf,
            part_name,
            n_smpl_channels,
            n_ssdr_channels,
            patch_size
        )
    else:
        # 返回完整的matched patches (包含SMPL+SSDR)
        refined_patches = target_patches[:, nnf, :]

    return refined_patches, distances, combined_patches, nnf
