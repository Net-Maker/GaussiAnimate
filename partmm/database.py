"""Pure helpers for constructing the PartMM motion-patch database."""

import numpy as np


def motion_patch_starts(length: int, patch_size: int) -> np.ndarray:
    """Return every valid starting frame for a fixed-size motion patch."""
    return np.arange(length - patch_size + 1)


def build_motion_patches(
    rotations: np.ndarray,
    starts: np.ndarray,
    patch_size: int,
    joint_mask: np.ndarray,
) -> np.ndarray:
    """Gather quaternion patches for one overlapping skeleton part."""
    return np.stack(
        [
            rotations[start : start + patch_size, joint_mask]
            for start in starts
        ]
    )


__all__ = ["build_motion_patches", "motion_patch_starts"]
