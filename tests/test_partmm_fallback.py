"""Fallback skinning and part-gated candidate blending contract tests."""

import inspect
import json
import unittest

import numpy as np

from partmm.fallback import PartMMFallback, SkinningBaseline
from partmm.kinematics import forward_kinematics, quaternion_to_matrix
from partmm.types import PartMMQuery, PartMMSkeleton


def _quaternion_z(angle):
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _skeleton():
    return PartMMSkeleton(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.5]]),
        np.array([-1, 0, 1]),
    )


def _reference_points():
    return np.array([[0.1, 0.2, 0.3], [1.5, -0.4, 0.7], [3.0, 1.0, -2.0]])


def _weights():
    return np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.0, 0.9, 0.1]])


def _pose(root_angle, root_translation):
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    rotations = np.stack([
        _quaternion_z(root_angle),
        identity,
        identity,
    ])[None]
    return PartMMQuery(rotations, np.asarray([root_translation], dtype=np.float64))


def _baseline():
    return SkinningBaseline(
        _skeleton(),
        _pose(0.3, [1.0, -2.0, 0.5]),
        _reference_points(),
        _weights(),
    )


def _target_frames():
    base = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    return np.stack([base + 0.1 * step for step in range(3)])


def _shift(shifts, frames):
    return np.stack([_target_frames()[frame] + shifts for frame in range(frames)])


def _calibrated_fallback():
    part_weights = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    candidate = _shift(np.array([0.01, 0.0, 0.0]), 3)
    candidate[:, 2] += 0.99
    baseline = _shift(np.array([0.5, 0.0, 0.0]), 3)
    baseline[:, 2] -= 0.49
    match_distances = np.array([[0.0, 0.5], [0.1, 0.5], [0.2, 0.5]])
    return PartMMFallback.calibrate(
        candidate_points=candidate,
        baseline_points=baseline,
        target_points=_target_frames(),
        part_weights=part_weights,
        match_distances=match_distances,
        distance_quantile=0.95,
        distance_policy="validation",
    )


def _single_part_fallback():
    part_weights = np.array([[1.0], [1.0], [1.0]])
    candidate = _shift(np.array([0.01, 0.0, 0.0]), 3)
    baseline = _shift(np.array([0.5, 0.0, 0.0]), 3)
    match_distances = np.array([[0.0], [0.1], [0.2]])
    return PartMMFallback.calibrate(
        candidate_points=candidate,
        baseline_points=baseline,
        target_points=_target_frames(),
        part_weights=part_weights,
        match_distances=match_distances,
        distance_quantile=0.95,
        distance_policy="validation",
    )


class SkinningBaselineTest(unittest.TestCase):
    def test_predict_matches_exact_rigid_motion_with_nonzero_root_and_pivot(self):
        baseline = _baseline()
        predicted = baseline.predict(_pose(1.0, [2.0, 0.0, 1.0]))
        rotation = (
            quaternion_to_matrix(_quaternion_z(1.0))
            @ quaternion_to_matrix(_quaternion_z(0.3)).T
        )
        pivot = np.array([1.0, -2.0, 0.5])
        expected = (
            rotation @ (_reference_points() - pivot).T
        ).T + np.array([2.0, 0.0, 1.0])
        np.testing.assert_allclose(predicted[0], expected, atol=1e-10)
        original = _reference_points()
        moved = predicted[0]
        np.testing.assert_allclose(
            np.linalg.norm(moved[:, None] - moved[None], axis=-1),
            np.linalg.norm(original[:, None] - original[None], axis=-1),
            atol=1e-10,
        )

    def test_weights_are_normalized_and_one_hot_follows_each_joint(self):
        baseline = SkinningBaseline(
            _skeleton(),
            PartMMQuery.identity(1, _skeleton().num_joints),
            _reference_points(),
            np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
        )
        np.testing.assert_allclose(baseline.weights.sum(axis=1), 1.0, atol=1e-12)
        query = PartMMQuery(
            np.stack([
                _quaternion_z(0.4),
                _quaternion_z(-0.8),
                _quaternion_z(1.1),
            ])[None],
            np.array([[0.3, -0.7, 1.2]]),
        )
        predicted = baseline.predict(query)
        query_transforms, _ = forward_kinematics(_skeleton(), query)
        reference_transforms, _ = forward_kinematics(
            _skeleton(), PartMMQuery.identity(1, _skeleton().num_joints)
        )
        relative = query_transforms[0] @ np.linalg.inv(reference_transforms[0])
        for joint in range(3):
            point = _reference_points()[joint]
            expected = relative[joint] @ np.concatenate([point, [1.0]])
            np.testing.assert_allclose(predicted[0, joint], expected[:3], atol=1e-10)

    def test_predict_and_apply_signatures_take_no_target(self):
        self.assertEqual(
            list(inspect.signature(_baseline().predict).parameters),
            ["query"],
        )
        self.assertEqual(
            list(inspect.signature(PartMMFallback.apply).parameters),
            ["self", "candidate_points", "baseline_points", "match_distances"],
        )


class PartMMFallbackTest(unittest.TestCase):
    def test_error_margin_prevents_discarding_a_much_better_candidate(self):
        target = np.zeros((3, 2, 3))
        candidate = target.copy()
        candidate[..., 0] = .01
        baseline = target.copy()
        baseline[..., 0] = .1
        guard = PartMMFallback.calibrate(candidate, baseline, target,
            np.ones((2, 1)), np.full((3, 1), .02), distance_policy='error_ratio')
        np.testing.assert_allclose(guard.distance_limits, [2.])
        result, _ = guard.apply(candidate[:1], baseline[:1], [[.5]])
        np.testing.assert_allclose(result, candidate[:1])
        result, _ = guard.apply(candidate[:1], baseline[:1], [[2.1]])
        np.testing.assert_allclose(result, baseline[:1])

    def test_default_uses_validation_error_not_unvalidated_distance_cutoff(self):
        target = np.zeros((3, 2, 3))
        candidate = target + .01
        baseline = target + .1
        guard = PartMMFallback.calibrate(candidate, baseline, target,
            np.ones((2, 1)), np.full((3, 1), .02))
        result, report = guard.apply(candidate[:1], baseline[:1], [[1.]])
        np.testing.assert_allclose(result, candidate[:1])
        self.assertEqual(report['distance_policy'], 'validation_error')

    def test_calibration_rms_and_enablement(self):
        fallback = _calibrated_fallback()
        np.testing.assert_array_equal(fallback.enabled_parts, [True, False])
        np.testing.assert_allclose(fallback.candidate_rmse[0], 0.01, atol=1e-12)
        np.testing.assert_allclose(fallback.baseline_rmse[0], 0.5, atol=1e-12)
        self.assertLess(fallback.candidate_rmse[0], fallback.baseline_rmse[0])
        self.assertGreater(fallback.candidate_rmse[1], fallback.baseline_rmse[1])
        np.testing.assert_allclose(fallback.distance_limits, [0.19, 0.5], atol=1e-12)
        np.testing.assert_allclose(fallback.part_weights.sum(axis=1), 1.0, atol=1e-12)

    def test_validation_loser_stays_fully_baseline(self):
        fallback = _calibrated_fallback()
        candidate = _shift(np.array([0.01, 0.0, 0.0]), 2)
        baseline = _shift(np.array([0.5, 0.0, 0.0]), 2)
        distances = np.array([[0.1, 0.5], [0.0, 0.5]])
        points, diagnostics = fallback.apply(candidate, baseline, distances)
        np.testing.assert_allclose(points[:, 0], candidate[:, 0], atol=1e-10)
        np.testing.assert_allclose(points[:, 2], baseline[:, 2], atol=1e-12)
        self.assertEqual(diagnostics["enabled_parts"], [True, False])
        np.testing.assert_allclose(
            diagnostics["accepted_fractions_per_part"], [0.5, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(diagnostics["baseline_fraction"], 0.5, atol=1e-12)
        self.assertEqual(
            diagnostics["accepted_mask"], [[True, False], [True, False]]
        )

    def test_winner_in_support_blends_to_candidate_and_diagnostics_are_json(self):
        fallback = _single_part_fallback()
        candidate = _shift(np.array([0.01, 0.0, 0.0]), 2)
        baseline = _shift(np.array([0.5, 0.0, 0.0]), 2)
        distances = np.array([[0.05], [0.0]])
        points, diagnostics = fallback.apply(candidate, baseline, distances)
        np.testing.assert_allclose(points, candidate, atol=1e-10)
        np.testing.assert_allclose(
            diagnostics["accepted_fractions_per_part"], [1.0], atol=1e-12
        )
        self.assertEqual(diagnostics["baseline_fraction"], 0.0)
        self.assertEqual(diagnostics["distance_metric"], "mean_squared_so3_radians")
        np.testing.assert_allclose(diagnostics["distance_limits"], [0.19])
        json.dumps(diagnostics)

    def test_out_of_support_winner_stays_fully_baseline(self):
        fallback = _single_part_fallback()
        candidate = _shift(np.array([0.01, 0.0, 0.0]), 2)
        baseline = _shift(np.array([0.5, 0.0, 0.0]), 2)
        distances = np.array([[0.3], [0.4]])
        points, diagnostics = fallback.apply(candidate, baseline, distances)
        np.testing.assert_allclose(points, baseline, atol=1e-12)
        np.testing.assert_allclose(
            diagnostics["accepted_fractions_per_part"], [0.0], atol=1e-12
        )
        self.assertEqual(diagnostics["baseline_fraction"], 1.0)
        self.assertEqual(
            diagnostics["accepted_mask"], [[False], [False]]
        )

    def test_per_part_soft_mixing_uses_part_weights(self):
        fallback = _calibrated_fallback()
        candidate = _shift(np.array([0.01, 0.0, 0.0]), 2)
        baseline = _shift(np.array([0.5, 0.0, 0.0]), 2)
        distances = np.array([[0.1, 0.5], [0.1, 0.5]])
        points, diagnostics = fallback.apply(candidate, baseline, distances)
        expected = baseline.copy()
        expected[:, 0] = candidate[:, 0]
        expected[:, 1] = baseline[:, 1] + 0.5 * (candidate[:, 1] - baseline[:, 1])
        expected[:, 2] = baseline[:, 2]
        np.testing.assert_allclose(points, expected, atol=1e-12)
        np.testing.assert_allclose(
            diagnostics["accepted_fractions_per_part"], [0.5, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(diagnostics["baseline_fraction"], 0.5, atol=1e-12)

    def test_apply_and_calibrate_reject_invalid_inputs(self):
        fallback = _calibrated_fallback()
        candidate = _shift(np.array([0.01, 0.0, 0.0]), 2)
        baseline = _shift(np.array([0.5, 0.0, 0.0]), 2)
        distances = np.array([[0.1, 0.5], [0.1, 0.5]])
        with self.assertRaises(ValueError):
            fallback.apply(candidate, baseline, distances[:, :1])
        with self.assertRaises(ValueError):
            fallback.apply(candidate, baseline, distances[:1])
        with self.assertRaises(ValueError):
            fallback.apply(candidate, baseline[:, :2], distances)
        with self.assertRaises(ValueError):
            fallback.apply(candidate, baseline, distances * np.nan)
        with self.assertRaises(ValueError):
            PartMMFallback.calibrate(
                candidate_points=candidate,
                baseline_points=baseline[:1],
                target_points=_target_frames()[:2],
                part_weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
                match_distances=distances,
            )
        with self.assertRaises(ValueError):
            PartMMFallback.calibrate(
                candidate_points=np.zeros((0, 3, 3)),
                baseline_points=np.zeros((0, 3, 3)),
                target_points=np.zeros((0, 3, 3)),
                part_weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
                match_distances=np.zeros((0, 2)),
            )
        with self.assertRaises(ValueError):
            PartMMFallback.calibrate(
                candidate_points=candidate,
                baseline_points=baseline,
                target_points=_target_frames()[:2],
                part_weights=np.array([[1.0, 0.0], [0.5, 0.5]]),
                match_distances=distances,
            )
        with self.assertRaises(ValueError):
            PartMMFallback.calibrate(
                candidate_points=candidate,
                baseline_points=baseline,
                target_points=_target_frames()[:2],
                part_weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
                match_distances=distances[:-1],
            )
        with self.assertRaises(ValueError):
            PartMMFallback.calibrate(
                candidate_points=candidate,
                baseline_points=baseline,
                target_points=_target_frames()[:2],
                part_weights=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
                match_distances=distances,
                distance_quantile=1.5,
            )
        with self.assertRaises(ValueError):
            SkinningBaseline(
                _skeleton(),
                PartMMQuery.identity(2, _skeleton().num_joints),
                _reference_points(),
                _weights(),
            )
        with self.assertRaises(ValueError):
            SkinningBaseline(
                _skeleton(),
                PartMMQuery.identity(1, _skeleton().num_joints),
                _reference_points(),
                np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]),
            )
        with self.assertRaises(ValueError):
            SkinningBaseline(
                _skeleton(),
                PartMMQuery.identity(1, _skeleton().num_joints),
                _reference_points(),
                -_weights(),
            )
        with self.assertRaises(ValueError):
            SkinningBaseline(
                _skeleton(),
                PartMMQuery.identity(1, _skeleton().num_joints),
                _reference_points()[:2],
                _weights(),
            )


if __name__ == "__main__":
    unittest.main()
