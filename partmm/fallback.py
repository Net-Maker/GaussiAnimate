"""Skinning baseline and validation-gated PartMM candidate fallback.

``SkinningBaseline`` applies caller-supplied, precomputed robust transferred
weights; it never computes or infers weights from geometry.

``PartMMFallback`` gates a candidate deformation on caller-supplied validation
frames only. Optional distance gates are experimental; the default selects
parts by validation position error. Neither guarantees unseen test errors.
"""

from dataclasses import dataclass

import numpy as np

from .kinematics import forward_kinematics
from .types import PartMMQuery, PartMMSkeleton, _float_array

_DISTANCE_METRIC = "mean_squared_so3_radians"


@dataclass
class SkinningBaseline:
    """Linear blend skinning baseline over one reference pose.

    ``weights`` must be the caller's precomputed robust transferred weights of
    shape [points, joints]; this class validates and applies them as-is.
    """

    skeleton: PartMMSkeleton
    reference_pose: PartMMQuery
    reference_points: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.skeleton, PartMMSkeleton):
            raise ValueError("skeleton must be a PartMMSkeleton")
        self.reference_pose = PartMMQuery.from_pose(self.reference_pose)
        if self.reference_pose.num_frames != 1:
            raise ValueError("reference_pose must contain exactly one frame")
        if self.reference_pose.num_joints != self.skeleton.num_joints:
            raise ValueError("reference_pose and skeleton joint counts differ")
        self.reference_points = _float_array(
            self.reference_points, "reference_points"
        )
        if (
            self.reference_points.ndim != 2
            or len(self.reference_points) < 1
            or self.reference_points.shape[1] != 3
        ):
            raise ValueError("reference_points must have shape [points, 3]")
        self.weights = _float_array(self.weights, "weights")
        expected = (len(self.reference_points), self.skeleton.num_joints)
        if self.weights.ndim != 2 or self.weights.shape != expected:
            raise ValueError(f"weights must have shape {expected}")
        if np.any(self.weights < -1e-8):
            raise ValueError("weights must be non-negative")
        row_sum = self.weights.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 1e-12):
            raise ValueError("every point must have at least one joint influence")
        self.weights = self.weights / row_sum

    def predict(self, query) -> np.ndarray:
        """Skin ``query`` relative to the stored reference pose.

        Computes forward kinematics for the query and the reference pose and
        blends the relative transforms ``G_query @ inv(G_reference)`` over the
        reference points with the stored weights. Returns [frames, points, 3]
        world-space points.
        """
        query = PartMMQuery.from_pose(query)
        if query.num_joints != self.skeleton.num_joints:
            raise ValueError("query and skeleton joint counts differ")
        query_transforms, _ = forward_kinematics(self.skeleton, query)
        reference_transforms, _ = forward_kinematics(
            self.skeleton, self.reference_pose
        )
        relative = query_transforms @ np.linalg.inv(reference_transforms)
        homogeneous = np.concatenate(
            [
                self.reference_points,
                np.ones((len(self.reference_points), 1), dtype=np.float64),
            ],
            axis=1,
        )
        return np.einsum(
            "nb,tbij,nj->tni", self.weights, relative, homogeneous, optimize=True
        )[..., :3]


@dataclass
class PartMMFallback:
    """Validation-gated blend of candidate points over the skinning baseline.

    ``calibrate`` compares the candidate against the baseline using only the
    caller-supplied validation frames. A part is enabled when its weighted mean
    squared position error is strictly better than the baseline, and its
    default gate selects parts using validation errors. Optional distance
    policies also reject unfamiliar queries. No gate guarantees test errors.
    """

    candidate_rmse: np.ndarray
    baseline_rmse: np.ndarray
    enabled_parts: np.ndarray
    distance_limits: np.ndarray
    part_weights: np.ndarray
    distance_policy: str = "validation_error"

    def __post_init__(self) -> None:
        self.candidate_rmse = _float_array(self.candidate_rmse, "candidate_rmse")
        self.baseline_rmse = _float_array(self.baseline_rmse, "baseline_rmse")
        self.distance_limits = _float_array(self.distance_limits, "distance_limits")
        if self.candidate_rmse.ndim != 1 or len(self.candidate_rmse) < 1:
            raise ValueError("candidate_rmse must have shape [parts]")
        part_count = len(self.candidate_rmse)
        if self.baseline_rmse.shape != (part_count,):
            raise ValueError("baseline_rmse must have shape [parts]")
        if self.distance_limits.shape != (part_count,):
            raise ValueError("distance_limits must have shape [parts]")
        self.enabled_parts = np.asarray(self.enabled_parts, dtype=bool).reshape(-1)
        if len(self.enabled_parts) != part_count:
            raise ValueError("enabled_parts must have shape [parts]")
        self.part_weights = _float_array(self.part_weights, "part_weights")
        if (
            self.part_weights.ndim != 2
            or len(self.part_weights) < 1
            or self.part_weights.shape[1] != part_count
        ):
            raise ValueError("part_weights must have shape [points, parts]")
        if np.any(self.part_weights < -1e-8):
            raise ValueError("part_weights must be non-negative")
        row_sum = self.part_weights.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 1e-12):
            raise ValueError("every point must have at least one part influence")
        self.part_weights = self.part_weights / row_sum
        if self.distance_policy not in {"validation_error", "validation", "error_ratio"}:
            raise ValueError("unknown distance_policy")

    @classmethod
    def calibrate(
        cls,
        candidate_points,
        baseline_points,
        target_points,
        part_weights,
        match_distances,
        distance_quantile: float = 0.95,
        distance_policy: str = "validation_error",
    ) -> "PartMMFallback":
        """Fit the gate on caller-supplied validation frames.

        ``match_distances`` are mean squared SO(3) geodesic radians per part
        and validation frame. The weighted mean squared position error is
        taken across validation frames and points; the candidate RMSE is its
        square root.
        """
        candidate_points = _float_array(candidate_points, "candidate_points")
        baseline_points = _float_array(baseline_points, "baseline_points")
        target_points = _float_array(target_points, "target_points")
        if (
            candidate_points.ndim != 3
            or len(candidate_points) < 1
            or candidate_points.shape[2] != 3
        ):
            raise ValueError("candidate_points must have shape [frames, points, 3]")
        if baseline_points.shape != candidate_points.shape:
            raise ValueError(
                "baseline_points must have shape [frames, points, 3] "
                "matching candidate_points"
            )
        if target_points.shape != candidate_points.shape:
            raise ValueError(
                "target_points must have shape [frames, points, 3] "
                "matching candidate_points"
            )
        part_weights = _float_array(part_weights, "part_weights")
        if (
            part_weights.ndim != 2
            or part_weights.shape[0] != candidate_points.shape[1]
            or part_weights.shape[1] < 1
        ):
            raise ValueError(
                "part_weights must have shape [points, parts] "
                "matching candidate_points"
            )
        if np.any(part_weights < -1e-8):
            raise ValueError("part_weights must be non-negative")
        row_sum = part_weights.sum(axis=1, keepdims=True)
        if np.any(row_sum <= 1e-12):
            raise ValueError("every point must have at least one part influence")
        part_weights = part_weights / row_sum
        weight_mass = part_weights.sum(axis=0)
        if np.any(weight_mass <= 1e-12):
            raise ValueError("every part must carry positive weight mass")
        match_distances = _float_array(match_distances, "match_distances")
        if match_distances.shape != (
            len(candidate_points),
            part_weights.shape[1],
        ):
            raise ValueError(
                "match_distances must have shape [frames, parts] "
                "matching candidate_points"
            )
        if np.any(match_distances < -1e-8):
            raise ValueError("match_distances must be non-negative")
        distance_quantile = float(distance_quantile)
        if not 0.0 <= distance_quantile <= 1.0:
            raise ValueError("distance_quantile must be in [0, 1]")
        if distance_policy not in {"validation_error", "validation", "error_ratio"}:
            raise ValueError("unknown distance_policy")
        frames = len(candidate_points)
        candidate_error = ((target_points - candidate_points) ** 2).sum(axis=2)
        baseline_error = ((target_points - baseline_points) ** 2).sum(axis=2)
        candidate_mse = (
            candidate_error[:, :, None] * part_weights[None]
        ).sum(axis=(0, 1)) / (frames * weight_mass)
        baseline_mse = (
            baseline_error[:, :, None] * part_weights[None]
        ).sum(axis=(0, 1)) / (frames * weight_mass)
        distance_limits = np.quantile(match_distances, distance_quantile, axis=0)
        if distance_policy == "error_ratio":
            # First-order pose perturbations imply position MSE scales with
            # squared angular distance. Spend only the measured baseline margin.
            ratio = np.maximum(1.0, baseline_mse / np.maximum(candidate_mse, 1e-16))
            distance_limits = np.minimum(np.pi ** 2, distance_limits * ratio)
        return cls(
            candidate_rmse=np.sqrt(candidate_mse),
            baseline_rmse=np.sqrt(baseline_mse),
            enabled_parts=candidate_mse < baseline_mse,
            distance_limits=distance_limits,
            part_weights=part_weights,
            distance_policy=distance_policy,
        )

    def apply(self, candidate_points, baseline_points, match_distances):
        """Blend gated candidate points over the baseline for query frames.

        A part is accepted when enabled and, for optional distance policies,
        its distance is within the limit. The point blend fraction is the accepted part
        weights; returns the points and a JSON-compatible diagnostics dict.
        """
        candidate_points = _float_array(candidate_points, "candidate_points")
        baseline_points = _float_array(baseline_points, "baseline_points")
        if (
            candidate_points.ndim != 3
            or len(candidate_points) < 1
            or candidate_points.shape[2] != 3
        ):
            raise ValueError("candidate_points must have shape [frames, points, 3]")
        if baseline_points.shape != candidate_points.shape:
            raise ValueError(
                "baseline_points must have shape [frames, points, 3] "
                "matching candidate_points"
            )
        if candidate_points.shape[1] != len(self.part_weights):
            raise ValueError("candidate_points must have one row per calibrated point")
        match_distances = _float_array(match_distances, "match_distances")
        expected = (len(candidate_points), len(self.enabled_parts))
        if match_distances.shape != expected:
            raise ValueError(f"match_distances must have shape {expected}")
        if np.any(match_distances < -1e-8):
            raise ValueError("match_distances must be non-negative")
        supported = (
            np.ones_like(match_distances, dtype=bool)
            if self.distance_policy == "validation_error"
            else match_distances <= self.distance_limits[None, :]
        )
        accepted = self.enabled_parts[None, :] & supported
        accepted_float = accepted.astype(np.float64)
        fraction = accepted_float @ self.part_weights.T
        points = baseline_points + fraction[..., None] * (
            candidate_points - baseline_points
        )
        accepted_share = (
            accepted_float.mean(axis=0) * self.part_weights.mean(axis=0)
        )
        diagnostics = {
            "distance_metric": _DISTANCE_METRIC,
            "distance_policy": self.distance_policy,
            "position_error_units": "input_coordinate_units",
            "enabled_parts": [bool(value) for value in self.enabled_parts],
            "candidate_rmse": self.candidate_rmse.tolist(),
            "baseline_rmse": self.baseline_rmse.tolist(),
            "distance_limits": self.distance_limits.tolist(),
            "accepted_fractions_per_part": accepted_share.tolist(),
            "accepted_rates_per_part": accepted_float.mean(axis=0).tolist(),
            "baseline_fraction": float(np.clip(1.0 - accepted_share.sum(), 0.0, 1.0)),
            "accepted_mask": accepted.tolist(),
        }
        return points, diagnostics
