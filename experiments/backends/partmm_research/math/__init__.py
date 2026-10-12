"""
Mathematical utilities for manifold-aware operations
"""

from .rotations import (
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
    axis_angle_to_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    blend_transformations
)

from .se3 import (
    se3_log_map,
    se3_exp_map,
    se3_interpolate
)

from .distances import (
    quaternion_geodesic_distance,
    se3_geodesic_distance
)

__all__ = [
    # Rotations
    'axis_angle_to_quaternion',
    'quaternion_to_axis_angle',
    'axis_angle_to_matrix',
    'rotation_matrix_to_quaternion',
    'quaternion_to_rotation_matrix',
    'blend_transformations',
    # SE3
    'se3_log_map',
    'se3_exp_map',
    'se3_interpolate',
    # Distances
    'quaternion_geodesic_distance',
    'se3_geodesic_distance',
]
