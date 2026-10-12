"""LBS reconstruction - Complete implementation"""
import numpy as np
import torch


def result_to_lbs_transforms(motion_result, original_ssdr_data_dict):
    """
    Convert Motion2Motion result back to LBS transformation format

    Args:
        motion_result: torch.Tensor, shape [1, channels, frames] from Motion2Motion
        original_ssdr_data_dict: dict, original SSDR data containing weights, rest_pose

    Returns:
        dict: LBS data with new transforms
    """
    # Remove batch dimension and transpose: [1, channels, frames] -> [frames, channels]
    motion_data = motion_result.squeeze(0).permute(1, 0)

    transform_data = motion_data[:, :-3]

    print(f"Converting motion result {motion_result.shape} -> transforms {transform_data.shape}")

    n_frames = transform_data.shape[0]
    n_transforms = transform_data.shape[1] // 16

    transforms = transform_data.reshape(n_frames, n_transforms, 4, 4)

    print(f"Reshaped to LBS transforms: {transforms.shape}")

    # Create new LBS data dictionary
    new_lbs_data = {
        'transformations': transforms.detach().cpu().numpy(),
        'weights': original_ssdr_data_dict['weights'],
        'rest_pose': original_ssdr_data_dict['rest_pose'],
        'reconstruction': original_ssdr_data_dict.get('reconstruction', None)
    }

    return new_lbs_data


def reconstruct_vertices(lbs_data):
    """
    Reconstruct vertices using LBS (Linear Blend Skinning)
    Fully vectorized implementation using Einstein summation

    Args:
        lbs_data: dict with 'transformations', 'weights', 'rest_pose'

    Returns:
        np.ndarray: reconstructed vertices [n_frames, n_vertices, 3]
    """
    transforms = lbs_data['transformations']  # [n_frames, n_bones, 4, 4]
    weights = lbs_data['weights']        # [n_vertices, n_bones]
    rest_pose = lbs_data['rest_pose']    # [n_vertices, 3]

    # Get dimensions first
    n_frames, n_bones, _, _ = transforms.shape
    n_vertices = rest_pose.shape[0]

    # Convert sparse matrix to dense if needed
    if hasattr(weights, 'toarray'):
        weights = weights.toarray()
    weights = np.array(weights)

    # Check weights dimension and transpose if needed
    if weights.shape[0] == n_bones and weights.shape[1] != n_bones:
        # weights is (n_bones, n_vertices), need to transpose to (n_vertices, n_bones)
        weights = weights.T
        print(f"Transposed weights from (n_bones, n_vertices) to (n_vertices, n_bones)")

    print(f"Reconstructing vertices: {n_frames} frames, {n_vertices} vertices, {n_bones} bones")
    print(f"Weights shape: {weights.shape}, dtype: {weights.dtype}")

    # Verify dimensions
    assert weights.shape == (n_vertices, n_bones), f"Expected weights shape ({n_vertices}, {n_bones}), got {weights.shape}"

    # Add homogeneous coordinate to rest pose: [n_vertices, 3] -> [n_vertices, 4]
    rest_pose_homo = np.concatenate([rest_pose, np.ones((n_vertices, 1))], axis=1)

    print("  Computing LBS using fully vectorized Einstein summation...")

    # FULLY VECTORIZED LBS COMPUTATION
    # Step 1: Compute weighted blended transforms for each vertex and frame
    # Using einsum notation: 'vb,fbij->fvij'
    blended_transforms = np.einsum('vb,fbij->fvij', weights, transforms)

    # Step 2: Apply blended transforms to rest pose vertices
    # Using einsum notation: 'fvij,vj->fvi'
    result_homo = np.einsum('fvij,vj->fvi', blended_transforms, rest_pose_homo)

    # Extract 3D coordinates (drop homogeneous coordinate)
    reconstructed_vertices = result_homo[:, :, :3]

    print(f"  ✓ Vectorized LBS complete! Output shape: {reconstructed_vertices.shape}")

    return reconstructed_vertices


def extract_ssdr_bone_positions(ssdr_data, original_ssdr_dict):
    """
    Extract SSDR bone positions from motion data

    Args:
        ssdr_data: torch.Tensor [1, 611, T] or [1, 608, T]
        original_ssdr_dict: dict with 'weights', 'rest_pose', 'transformations'

    Returns:
        bone_positions: np.ndarray [T, n_bones, 3]
    """
    print("\nExtracting SSDR bone positions from motion data...")

    # Step 1: Convert motion data to LBS format
    lbs_data = result_to_lbs_transforms(ssdr_data, original_ssdr_dict)

    # Step 2: Reconstruct vertices using LBS
    print("  Reconstructing vertices using LBS...")
    reconstructed_vertices = reconstruct_vertices(lbs_data)
    n_frames, n_vertices, _ = reconstructed_vertices.shape

    # Step 3: Extract skinning weights
    weights = lbs_data['weights']

    # Handle sparse matrix
    if hasattr(weights, 'toarray'):
        weights = weights.toarray()

    # Ensure weights shape is [n_bones, n_verts]
    if weights.shape[0] > weights.shape[1]:
        weights = weights.T

    n_bones = weights.shape[0]

    print(f"  Vertices shape: {reconstructed_vertices.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Number of bones: {n_bones}")

    # Step 4: For each bone, find vertex with maximum weight
    bone_positions = np.zeros((n_frames, n_bones, 3))

    for bone_idx in range(n_bones):
        bone_weights = weights[bone_idx]
        max_weight_idx = np.argmax(bone_weights)
        bone_positions[:, bone_idx, :] = reconstructed_vertices[:, max_weight_idx, :]

    print(f"  ✓ Extracted bone positions: {bone_positions.shape}")
    print(f"  ✓ Position range: [{bone_positions.min():.3f}, {bone_positions.max():.3f}]")

    return bone_positions
