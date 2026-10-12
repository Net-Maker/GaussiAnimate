"""
SE(3) Lie group operations

SE(3) = Special Euclidean group (rigid body transformations)
- Matrices: 4x4 homogeneous transforms [R|t; 0|1]
- Lie algebra: se(3) - matrix logarithm space
"""

import torch
import torch.nn.functional as F
import numpy as np


def se3_log_map(T):
    """
    Compute matrix logarithm of SE(3) transformation

    For T ∈ SE(3), compute log(T) ∈ se(3) (Lie algebra)

    Args:
        T: torch.Tensor of shape (..., 4, 4)

    Returns:
        log_T: torch.Tensor of shape (..., 4, 4) in se(3)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Compute rotation angle
    trace_R = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(torch.clamp((trace_R - 1.0) / 2.0, -1.0, 1.0))

    log_T = torch.zeros_like(T)

    # Small angle case
    small_angle_mask = theta.abs() < 1e-4
    # Large angle case
    large_angle_mask = (torch.pi - theta).abs() < 1e-4
    # Normal case
    normal_mask = ~(small_angle_mask | large_angle_mask)

    # Normal case
    if normal_mask.any():
        theta_n = theta[normal_mask]
        R_n = R[normal_mask]
        t_n = t[normal_mask]

        coeff = theta_n / (2.0 * torch.sin(theta_n))
        omega_hat = coeff.view(-1, 1, 1) * (R_n - R_n.transpose(-2, -1))

        omega_hat_sq = torch.matmul(omega_hat, omega_hat)
        theta_sq = theta_n.view(-1, 1, 1) ** 2

        sin_half = torch.sin(theta_n / 2.0).view(-1, 1, 1)
        cos_half = torch.cos(theta_n / 2.0).view(-1, 1, 1)
        cot_half = cos_half / (sin_half + 1e-8)

        coeff2 = (1.0 / theta_sq) - (cot_half / (2.0 * theta_n.view(-1, 1, 1)))

        V_inv = torch.eye(3, device=T.device).unsqueeze(0) - omega_hat / 2.0 + coeff2 * omega_hat_sq
        v = torch.matmul(V_inv, t_n.unsqueeze(-1)).squeeze(-1)

        log_T_n = torch.zeros_like(T[normal_mask])
        log_T_n[..., :3, :3] = omega_hat
        log_T_n[..., :3, 3] = v
        log_T[normal_mask] = log_T_n

    # Small angle case
    if small_angle_mask.any():
        R_s = R[small_angle_mask]
        t_s = t[small_angle_mask]

        omega_hat_s = (R_s - R_s.transpose(-2, -1)) / 2.0
        v_s = t_s

        log_T_s = torch.zeros_like(T[small_angle_mask])
        log_T_s[..., :3, :3] = omega_hat_s
        log_T_s[..., :3, 3] = v_s
        log_T[small_angle_mask] = log_T_s

    # Large angle case
    if large_angle_mask.any():
        R_l = R[large_angle_mask]
        t_l = t[large_angle_mask]

        omega_hat_l = (R_l - R_l.transpose(-2, -1)) * torch.pi / 4.0
        v_l = t_l

        log_T_l = torch.zeros_like(T[large_angle_mask])
        log_T_l[..., :3, :3] = omega_hat_l
        log_T_l[..., :3, 3] = v_l
        log_T[large_angle_mask] = log_T_l

    return log_T


def se3_exp_map(log_T):
    """
    Compute matrix exponential of se(3) -> SE(3)
    Inverse of se3_log_map

    Args:
        log_T: torch.Tensor of shape (..., 4, 4) in se(3)

    Returns:
        T: torch.Tensor of shape (..., 4, 4) in SE(3)
    """
    omega_hat = log_T[..., :3, :3]
    v = log_T[..., :3, 3]

    # Compute rotation angle
    omega_hat_sq = torch.matmul(omega_hat, omega_hat)
    trace_omega_sq = omega_hat_sq[..., 0, 0] + omega_hat_sq[..., 1, 1] + omega_hat_sq[..., 2, 2]
    theta = torch.sqrt(torch.clamp(-trace_omega_sq / 2.0, min=0.0))

    T = torch.zeros_like(log_T)
    T[..., 3, 3] = 1.0

    small_angle_mask = theta.abs() < 1e-4
    normal_mask = ~small_angle_mask

    # Small angle case
    if small_angle_mask.any():
        omega_hat_s = omega_hat[small_angle_mask]
        v_s = v[small_angle_mask]

        R_s = torch.eye(3, device=log_T.device).unsqueeze(0) + omega_hat_s
        t_s = v_s

        T_s = torch.zeros_like(T[small_angle_mask])
        T_s[..., :3, :3] = R_s
        T_s[..., :3, 3] = t_s
        T_s[..., 3, 3] = 1.0
        T[small_angle_mask] = T_s

    # Normal case
    if normal_mask.any():
        theta_n = theta[normal_mask]
        omega_hat_n = omega_hat[normal_mask]
        v_n = v[normal_mask]

        sin_theta = torch.sin(theta_n).view(-1, 1, 1)
        cos_theta = torch.cos(theta_n).view(-1, 1, 1)
        theta_n_expanded = theta_n.view(-1, 1, 1)

        coeff1 = sin_theta / theta_n_expanded
        coeff2 = (1.0 - cos_theta) / (theta_n_expanded ** 2)

        omega_hat_sq_n = torch.matmul(omega_hat_n, omega_hat_n)

        R_n = torch.eye(3, device=log_T.device).unsqueeze(0) + coeff1 * omega_hat_n + coeff2 * omega_hat_sq_n

        coeff3 = (theta_n_expanded - sin_theta) / (theta_n_expanded ** 3)
        V_n = torch.eye(3, device=log_T.device).unsqueeze(0) + coeff2 * omega_hat_n + coeff3 * omega_hat_sq_n

        t_n = torch.matmul(V_n, v_n.unsqueeze(-1)).squeeze(-1)

        T_n = torch.zeros_like(T[normal_mask])
        T_n[..., :3, :3] = R_n
        T_n[..., :3, 3] = t_n
        T_n[..., 3, 3] = 1.0
        T[normal_mask] = T_n

    return T


def se3_interpolate(transforms, target_length):
    """
    Interpolate SE(3) transformations using log/exp maps

    CRITICAL: Linear interpolation in Lie algebra space (correct!)

    Args:
        transforms: torch.Tensor of shape (n_frames, n_bones, 4, 4)
        target_length: target number of frames

    Returns:
        interpolated: torch.Tensor of shape (target_length, n_bones, 4, 4)
    """
    n_frames, n_bones, _, _ = transforms.shape
    device = transforms.device

    if n_frames == target_length:
        return transforms

    if isinstance(transforms, np.ndarray):
        transforms = torch.from_numpy(transforms).float().to(device)

    # Compute log map for all frames
    log_transforms = torch.zeros_like(transforms)
    for frame_idx in range(n_frames):
        log_transforms[frame_idx] = se3_log_map(transforms[frame_idx])

    # Flatten and interpolate in Lie algebra
    log_flat = log_transforms.reshape(n_frames, n_bones * 16)

    log_interp = F.interpolate(
        log_flat.T.unsqueeze(0),
        size=target_length,
        mode='linear',
        align_corners=False
    )

    log_interp = log_interp.squeeze(0).T.reshape(target_length, n_bones, 4, 4)

    # Exponential map back to SE(3)
    result = torch.zeros_like(log_interp)
    for frame_idx in range(target_length):
        result[frame_idx] = se3_exp_map(log_interp[frame_idx])

    return result
