"""Rotation conversions used by the vendored BoB adapter."""

import torch


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle rotations to real-first quaternions.

    This avoids ``torch.sinc``, whose Jiterator path in PyTorch 1.12 cannot
    compile for Ada GPUs when the process resolves CUDA 11.6's NVRTC.
    """
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles.abs() < 1.0e-6
    safe_angles = torch.where(small, torch.ones_like(angles), angles)
    regular_scale = torch.sin(half_angles) / safe_angles
    angle_squared = angles.square()
    small_scale = 0.5 - angle_squared / 48.0 + angle_squared.square() / 3840.0
    scale = torch.where(small, small_scale, regular_scale)
    return torch.cat((torch.cos(half_angles), axis_angle * scale), dim=-1)
