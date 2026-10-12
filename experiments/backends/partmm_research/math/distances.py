"""
Manifold-aware geodesic distances

Implements proper distance metrics on:
- S³ (quaternion manifold)
- SE(3) (rigid transformation manifold)
"""

import torch
from .se3 import se3_log_map


def quaternion_geodesic_distance(q1, q2):
    """
    Compute geodesic distance on S³ (unit quaternion manifold)

    Distance: d(q1, q2) = arccos(|q1·q2|)

    Args:
        q1: torch.Tensor of shape (N, 4) - [w,x,y,z]
        q2: torch.Tensor of shape (M, 4) - [w,x,y,z]

    Returns:
        distances: torch.Tensor of shape (N, M)
    """
    # Normalize quaternions
    q1_norm = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-8)
    q2_norm = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-8)

    # Dot product: (N, M)
    dot_products = torch.matmul(q1_norm, q2_norm.T)

    # Geodesic distance: arccos(|dot|)
    dot_abs = torch.abs(dot_products)
    dot_clamped = torch.clamp(dot_abs, 0.0, 1.0)

    distances = torch.acos(dot_clamped)

    return distances


def se3_geodesic_distance(T1, T2, rotation_weight=1.0):
    """
    Compute geodesic distance on SE(3) manifold using log map
    VECTORIZED: no Python loops

    Distance: d(T1, T2) = rotation_weight * ||rotation||_F + ||translation||_2

    Args:
        T1: torch.Tensor of shape (N, 4, 4)
        T2: torch.Tensor of shape (M, 4, 4)
        rotation_weight: weight for rotation component

    Returns:
        distances: torch.Tensor of shape (N, M)
    """
    N = T1.shape[0]
    M = T2.shape[0]
    device = T1.device

    # Vectorized inverse of T1
    R1 = T1[:, :3, :3]
    t1 = T1[:, :3, 3]

    R1_inv = R1.transpose(-2, -1)
    t1_inv = -torch.einsum('nij,nj->ni', R1_inv, t1)

    T1_inv = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    T1_inv[:, :3, :3] = R1_inv
    T1_inv[:, :3, 3] = t1_inv

    # Vectorized relative transforms: T1^{-1} @ T2
    relative_T = torch.matmul(
        T1_inv.unsqueeze(1),  # (N, 1, 4, 4)
        T2.unsqueeze(0)       # (1, M, 4, 4)
    )  # (N, M, 4, 4)

    # Flatten for log map
    relative_T_flat = relative_T.reshape(N * M, 4, 4)
    log_T_flat = se3_log_map(relative_T_flat)
    log_T = log_T_flat.reshape(N, M, 4, 4)

    # Separate rotation and translation components
    omega_hat = log_T[..., :3, :3]
    v = log_T[..., :3, 3]

    rotation_norm = torch.norm(omega_hat.reshape(N, M, -1), dim=-1)
    translation_norm = torch.norm(v, dim=-1)

    weighted_rotation = rotation_norm * rotation_weight

    # L2 metric: sqrt(rot^2 + trans^2)
    eps = 1e-8
    distances = torch.sqrt(weighted_rotation ** 2 + translation_norm ** 2 + eps)

    return distances


def smpl_quaternion_distance(smpl1, smpl2, n_joints, patch_size):
    """
    Compute SMPL-only quaternion geodesic distance for patches

    Args:
        smpl1: [n1, n_smpl_channels, P] - query SMPL patches (quaternions + translation)
        smpl2: [n2, n_smpl_channels, P] - database SMPL patches (quaternions + translation)
        n_joints: number of SMPL joints (will be recalculated from channels)
        patch_size: P

    Returns:
        distances: [n1, n2] - geodesic distances
    """
    n1, n_smpl, P = smpl1.shape
    n2 = smpl2.shape[0]

    # SMPL格式: n_quats*4 + 3 (quaternions + translation)
    # 从channels推断quaternion数量
    n_quats = (n_smpl - 3) // 4
    quat_channels = n_quats * 4

    # 提取quaternion部分 (忽略translation)
    smpl1_quats = smpl1[:, :quat_channels, :].reshape(n1, n_quats, 4, P)  # [n1, n_quats, 4, P]
    smpl2_quats = smpl2[:, :quat_channels, :].reshape(n2, n_quats, 4, P)  # [n2, n_quats, 4, P]

    # Flatten to [n, n_quats*P, 4]
    smpl1_quats_flat = smpl1_quats.permute(0, 1, 3, 2).reshape(n1, n_quats * P, 4)
    smpl2_quats_flat = smpl2_quats.permute(0, 1, 3, 2).reshape(n2, n_quats * P, 4)

    # Normalize quaternions
    eps = 1e-8
    q1 = smpl1_quats_flat / (torch.norm(smpl1_quats_flat, dim=-1, keepdim=True) + eps)
    q2 = smpl2_quats_flat / (torch.norm(smpl2_quats_flat, dim=-1, keepdim=True) + eps)

    # Compute geodesic distance: arccos(|q1·q2|)
    q1_exp = q1[:, None, :, :]  # [n1, 1, n_quats*P, 4]
    q2_exp = q2[None, :, :, :]  # [1, n2, n_quats*P, 4]
    dot = (q1_exp * q2_exp).sum(-1)  # [n1, n2, n_quats*P]

    dot_clamped = torch.clamp(torch.abs(dot), 0.0, 1.0)
    ang = torch.acos(dot_clamped)

    # Reshape back: [n1, n2, n_quats*P] -> [n1, n2, n_quats, P]
    ang = ang.reshape(n1, n2, n_quats, P)

    # Root joint不参与距离计算
    ang[:, :, 0, :] = 0.0
    n_active_joints = n_quats - 1

    # Average over P and n_quats
    distances = torch.sqrt((ang ** 2).sum(dim=(2, 3)) / (n_active_joints * P + eps))


    return distances
