"""Combined SMPL+SSDR geodesic distance computation"""
import torch
from .distances import se3_geodesic_distance


def combined_smpl_ssdr_distance(x1, x2, n_smpl_channels, n_ssdr_channels, patch_size, rotation_weight=1.0):
    """
    Combined SMPL (SO3) + SSDR (SE3) geodesic distance

    Args:
        x1: [n1, (smpl+ssdr) * patch_size] - patches flattened
        x2: [n2, (smpl+ssdr) * patch_size]
        n_smpl_channels: SMPL channels per frame
        n_ssdr_channels: SSDR channels per frame
        patch_size: patch size
        rotation_weight: weight for SE3 rotation

    Returns:
        distances: [n1, n2]
    """
    device = x1.device
    n1, total_flat = x1.shape
    n2, _ = x2.shape
    P = patch_size

    # Total channels per frame
    n_channels_per_frame = n_smpl_channels + n_ssdr_channels

    # Verify shape
    expected_size = n_channels_per_frame * P
    if total_flat != expected_size:
        raise ValueError(f"Input size {total_flat} != expected {expected_size} = ({n_smpl_channels}+{n_ssdr_channels})*{P}")

    # Reshape: [n, (smpl+ssdr)*P] -> [n, smpl+ssdr, P]
    x1_frames = x1.reshape(n1, n_channels_per_frame, P)
    x2_frames = x2.reshape(n2, n_channels_per_frame, P)

    n_smpl = n_smpl_channels
    n_ssdr = n_ssdr_channels

    # x_frames is [n, channels, P], split by channels
    smpl1 = x1_frames[:, :n_smpl, :]  # [n1, n_smpl, P]
    ssdr1 = x1_frames[:, n_smpl:, :]  # [n1, n_ssdr, P]

    smpl2 = x2_frames[:, :n_smpl, :]  # [n2, n_smpl, P]
    ssdr2 = x2_frames[:, n_smpl:, :]  # [n2, n_ssdr, P]

    # SMPL: SO(3) quaternion geodesic
    # n_quats从smpl_channels推断：(n_quats*4 + 3) = n_smpl_channels
    n_quats = (n_smpl - 3) // 4
    quat_channels = n_quats * 4

    # smpl is [n, n_smpl, P], extract quats: [n, quat_channels, P]
    smpl1_quats = smpl1[:, :quat_channels, :].reshape(n1, n_quats, 4, P)  # [n1, n_quats, 4, P]
    smpl2_quats = smpl2[:, :quat_channels, :].reshape(n2, n_quats, 4, P)  # [n2, n_quats, 4, P]

    # Flatten: [n, n_quats, 4, P] -> [n, n_quats*P, 4]
    smpl1_quats_flat = smpl1_quats.permute(0, 1, 3, 2).reshape(n1, n_quats * P, 4)
    smpl2_quats_flat = smpl2_quats.permute(0, 1, 3, 2).reshape(n2, n_quats * P, 4)

    eps = 1e-8
    q1 = smpl1_quats_flat / (torch.norm(smpl1_quats_flat, dim=-1, keepdim=True) + eps)
    q2 = smpl2_quats_flat / (torch.norm(smpl2_quats_flat, dim=-1, keepdim=True) + eps)

    q1_exp = q1[:, None, :, :]
    q2_exp = q2[None, :, :, :]
    dot = (q1_exp * q2_exp).sum(-1)

    dot_clamped = torch.clamp(torch.abs(dot), 0.0, 1.0)
    ang = torch.acos(dot_clamped)
    # Reshape back: [n1, n2, n_quats*P] -> [n1, n2, n_quats, P]
    ang = ang.reshape(n1, n2, n_quats, P)

    # Root joint不参与距离计算
    ang[:, :, 0, :] = 0.0
    n_active_joints = n_quats - 1

    # Average over P and n_quats
    smpl_distance = torch.sqrt((ang ** 2).sum(dim=(2, 3)) / (n_active_joints * P + eps))

    # SSDR: SE(3) geodesic
    n_bones = (n_ssdr - 3) // 16

    # ssdr is [n, n_ssdr, P], extract transforms
    ssdr1_T = ssdr1[:, :n_bones * 16, :].reshape(n1, n_bones, 4, 4, P)  # [n1, n_bones, 4, 4, P]
    ssdr2_T = ssdr2[:, :n_bones * 16, :].reshape(n2, n_bones, 4, 4, P)  # [n2, n_bones, 4, 4, P]

    # Reorder: [n, n_bones, 4, 4, P] -> [n_bones, P, n, 4, 4]
    ssdr1_reorder = ssdr1_T.permute(1, 4, 0, 2, 3)
    ssdr2_reorder = ssdr2_T.permute(1, 4, 0, 2, 3)

    ssdr1_flat = ssdr1_reorder.reshape(n_bones * P, n1, 4, 4)
    ssdr2_flat = ssdr2_reorder.reshape(n_bones * P, n2, 4, 4)

    def _so3_log(R):
        tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        cos_theta = (tr - 1.0) * 0.5
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        wx = R[..., 2, 1] - R[..., 1, 2]
        wy = R[..., 0, 2] - R[..., 2, 0]
        wz = R[..., 1, 0] - R[..., 0, 1]
        w = torch.stack((wx, wy, wz), dim=-1)

        sin_theta = torch.sin(theta)
        small = theta.abs() < 1e-5

        k = torch.empty_like(theta)
        k[~small] = theta[~small] / (2.0 * sin_theta[~small])
        k[small] = 0.5

        return k.unsqueeze(-1) * w

    def _se3_pairwise_distance(T1, T2, w_rot=1.0, w_trans=1.0):
        R1 = T1[:, :3, :3]
        t1 = T1[:, :3, 3]
        R2 = T2[:, :3, :3]
        t2 = T2[:, :3, 3]

        R1T = R1.transpose(1, 2)[:, None, :, :]
        R2e = R2[None, :, :, :]
        R_rel = torch.matmul(R1T, R2e)

        t1e = t1[:, None, :]
        t2e = t2[None, :, :]
        dt = t2e - t1e
        dt_rot = torch.matmul(R1T, dt.unsqueeze(-1)).squeeze(-1)

        w = _so3_log(R_rel)

        rot_term = w_rot * torch.norm(w, dim=-1)
        trans_term = w_trans * torch.norm(dt_rot, dim=-1)

        return torch.sqrt(rot_term ** 2 + trans_term ** 2 + eps)

    total_se3_dist = torch.zeros(n1, n2, device=device)

    for bf_idx in range(n_bones * P):
        T1_bf = ssdr1_flat[bf_idx]
        T2_bf = ssdr2_flat[bf_idx]
        dist_bf = _se3_pairwise_distance(T1_bf, T2_bf, w_rot=rotation_weight, w_trans=1.0)
        total_se3_dist += dist_bf

    ssdr_distance = total_se3_dist / float(n_bones * P)

    return smpl_distance + ssdr_distance
