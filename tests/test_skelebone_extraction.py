import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from skelebone import DynamicPointSequence
from skelebone.io import load_bundle
from skelebone.pipeline import ExtractionConfig, SkelebonePipeline
from skelebone.extraction import (
    Correspondence,
    CurveExtractionConfig,
    CurveSkeleton,
    JointConfig,
    JointDetection,
    MeshConfig,
    SSDRConfig,
    TrackingConfig,
    TriangleMesh,
    build_kinematic_tree,
    close_boundary_loops,
    decompose_ssdr,
    detect_joints,
    extract_curve_skeleton,
    is_watertight,
    prepare_mesh,
    project_skinning_to_curve,
    read_off,
    track_joints,
    transfer_skinning_to_mesh,
    vertex_curve_correspondence,
    write_off,
)
from skelebone.extraction.ssdr import motion_cluster_labels


class _FakeDemBones:
    def load_data(self, trajectory, faces):
        self.trajectory = trajectory
        self.faces = faces

    def run_ssdr(self, num_bones, output_path):
        num_frames = self.trajectory.shape[0] // 3
        num_points = self.trajectory.shape[1]
        weights = np.zeros((num_bones, num_points), dtype=float)
        weights[np.arange(num_points) % num_bones, np.arange(num_points)] = 1.0
        transforms = np.broadcast_to(
            np.eye(4), (num_frames, num_bones, 4, 4)
        ).copy().transpose(0, 2, 1, 3)
        return weights, transforms, 0.0


class _CollapsedDemBones(_FakeDemBones):
    def run_ssdr(self, num_bones, output_path):
        return super().run_ssdr(1, output_path)


class _ConfigurableDemBones(_FakeDemBones):
    clusterThreshold = -1
    labelAssignmentMode = 0
    splitSeedNeighbors = 0
    pruneDelay = 0
    zeroGrowthPatience = 0
    pointCloudSmoothMode = 1


class _MotionInitializedDemBones(_FakeDemBones):
    def labelToWeights(self):
        labels = np.asarray(self.label, dtype=int)
        self.nB = int(labels.max()) + 1
        self.initialized_weights = np.zeros((self.nB, len(labels)), dtype=float)
        self.initialized_weights[labels, np.arange(len(labels))] = 1.0

    def computeTransFromLabel(self):
        pass

    def run_ssdr(self, num_bones, output_path):
        num_frames = self.trajectory.shape[0] // 3
        transforms = np.broadcast_to(
            np.eye(4), (num_frames, self.nB, 4, 4)
        ).copy().transpose(0, 2, 1, 3)
        return self.initialized_weights, transforms, 0.0


class PipelineTest(unittest.TestCase):
    def test_end_to_end_artifact_contract(self):
        rest = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int
        )
        sequence = DynamicPointSequence(np.stack([rest, rest]), faces=faces, fps=24)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "fake_curve.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "prefix = pathlib.Path(sys.argv[2])\n"
                "pathlib.Path(str(prefix) + '_points.txt').write_text("
                "'0 0 0\\n0.5 0 0\\n1 0 0\\n')\n"
                "pathlib.Path(str(prefix) + '_edges.txt').write_text('0 1\\n1 2\\n')\n",
                encoding="ascii",
            )
            executable.chmod(0o755)
            result = SkelebonePipeline(
                ExtractionConfig(
                    num_bones=2,
                    curve_skeleton_bin=str(executable),
                    curve_correspondence_neighbors=1,
                    joint_track_neighbors=4,
                )
            ).run(sequence, directory / "output", ssdr_backend=_FakeDemBones())
            loaded = load_bundle(result.bundle_path)
            self.assertTrue(result.bvh_path.is_file())
            self.assertEqual(result.bundle_path.name, "skelebones.npz")
            self.assertTrue(result.manifest_path.is_file())
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["surface"]["source"], "input_faces")
            self.assertEqual(manifest["main_outputs"]["skelebones"], "skelebones.npz")
            self.assertEqual(loaded.weights.shape, (4, 2))
            self.assertEqual(loaded.motion.num_frames, 2)
            for relative_path in loaded.artifacts.values():
                self.assertTrue((directory / "output" / relative_path).exists())


class RestFrameConsistencyTest(unittest.TestCase):
    def test_default_mesh_frame_resolves_to_rest_frame(self):
        ssdr, mesh, curve, joints, tracking = SkelebonePipeline(
            ExtractionConfig(rest_frame=1, curve_skeleton_bin="unused")
        )._component_configs()
        self.assertEqual(mesh.frame, 1)
        self.assertEqual(tracking.rest_frame, 1)

    def test_default_rest_frame_zero_still_resolves_mesh_frame_zero(self):
        ssdr, mesh, curve, joints, tracking = SkelebonePipeline(
            ExtractionConfig(curve_skeleton_bin="unused")
        )._component_configs()
        self.assertEqual(mesh.frame, 0)
        self.assertEqual(tracking.rest_frame, 0)

    def test_explicit_mesh_frame_must_match_rest_frame(self):
        with self.assertRaisesRegex(ValueError, "mesh_frame"):
            ExtractionConfig(rest_frame=1, mesh_frame=0)

    def test_rigid_translation_tracks_in_rest_frame_coordinates(self):
        rest = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int
        )
        translation = np.array([0.0, 10.0, 0.0])
        frames = np.stack([rest, rest + translation])
        sequence = DynamicPointSequence(frames, faces=faces, fps=24)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "fake_curve.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "lines = pathlib.Path(sys.argv[1]).read_text().splitlines()[2:]\n"
                "prefix = pathlib.Path(sys.argv[2])\n"
                "pathlib.Path(str(prefix) + '_points.txt').write_text("
                "'\\n'.join(lines[:3]) + '\\n')\n"
                "pathlib.Path(str(prefix) + '_edges.txt').write_text('0 1\\n1 2\\n')\n",
                encoding="ascii",
            )
            executable.chmod(0o755)
            SkelebonePipeline(
                ExtractionConfig(
                    num_bones=2,
                    rest_frame=1,
                    curve_skeleton_bin=str(executable),
                    curve_correspondence_neighbors=1,
                    joint_track_neighbors=4,
                )
            ).run(sequence, directory / "output", ssdr_backend=_FakeDemBones())
            trajectories = np.load(
                directory / "output" / "kinematic_tree" / "joint_trajectories.npz"
            )
            tracked = trajectories["tracked_joint_positions"]
            np.testing.assert_allclose(tracked[1], trajectories["rest_positions"])
            np.testing.assert_allclose(tracked[0], tracked[1] - translation, atol=1e-9)
            # The object sits near y=10 in the rest frame, so the rest-frame
            # joints must follow it instead of staying at the origin.
            self.assertTrue(np.all(tracked[1][:, 1] > 5.0))
            self.assertTrue(np.all(tracked[0][:, 1] < 5.0))


class SSDRExtractionTest(unittest.TestCase):
    def test_rejects_unknown_protocol(self):
        with self.assertRaisesRegex(ValueError, "protocol"):
            SSDRConfig(protocol="unpublished_hybrid")

    def test_paper_legacy_configures_exact_historical_backend_controls(self):
        points = np.zeros((2, 8, 3), dtype=float)
        backend = _ConfigurableDemBones()

        result = decompose_ssdr(
            DynamicPointSequence(points),
            SSDRConfig(
                num_bones=2,
                protocol="paper_legacy",
                initialization="backend",
            ),
            backend=backend,
        )

        self.assertEqual(backend.clusterThreshold, 3)
        self.assertEqual(backend.labelAssignmentMode, 1)
        self.assertEqual(backend.splitSeedNeighbors, 0)
        self.assertEqual(backend.pruneDelay, 0)
        self.assertEqual(backend.zeroGrowthPatience, 0)
        self.assertEqual(backend.pointCloudSmoothMode, 0)
        self.assertEqual(result.protocol, "paper_legacy")

    def test_motion_pointcloud_reports_requested_initialized_and_effective_bones(self):
        static_x = np.linspace(0.0, 4.0, 24)
        rest = np.stack(
            [static_x, np.zeros_like(static_x), np.zeros_like(static_x)], axis=1
        )
        points = np.stack([rest, rest.copy(), rest.copy()])
        points[1, 8:16, 1] += 1.0
        points[2, 8:16, 1] += 2.0
        points[1, 16:, 2] += 1.0
        points[2, 16:, 2] += 2.0

        result = decompose_ssdr(
            DynamicPointSequence(points),
            SSDRConfig(
                num_bones=3,
                protocol="motion_pointcloud",
                require_num_bones=True,
                random_seed=7,
            ),
            backend=_MotionInitializedDemBones(),
        )

        self.assertEqual(result.protocol, "motion_pointcloud")
        self.assertEqual(result.requested_bones, 3)
        self.assertEqual(result.initialized_bones, 3)
        self.assertEqual(result.num_bones, 3)
        self.assertEqual(result.effective_bones, 3)
        self.assertEqual(result.backend_name, "_MotionInitializedDemBones")
        self.assertEqual(
            result.diagnostics["backend"]["class"], "_MotionInitializedDemBones"
        )
        self.assertEqual(result.hard_bone_population.shape, (3,))
        self.assertEqual(result.soft_bone_mass.shape, (3,))
        np.testing.assert_array_equal(result.hard_bone_population, [8, 8, 8])
        np.testing.assert_allclose(result.soft_bone_mass, [8, 8, 8])

    def test_rejects_silent_bone_collapse(self):
        points = np.zeros((2, 8, 3), dtype=float)
        with self.assertRaisesRegex(RuntimeError, "returned 1 bones; requested 3"):
            decompose_ssdr(
                DynamicPointSequence(points),
                SSDRConfig(
                    num_bones=3,
                    initialization="backend",
                    require_num_bones=True,
                ),
                backend=_CollapsedDemBones(),
            )

    def test_motion_initialization_keeps_small_articulated_region(self):
        static_x = np.linspace(0.0, 4.0, 200)
        static = np.stack([static_x, np.zeros_like(static_x), np.zeros_like(static_x)], axis=1)
        moving = np.array(
            [[4.2, 0.0, 0.0], [4.3, 0.0, 0.0], [4.4, 0.0, 0.0], [4.5, 0.0, 0.0]]
        )
        rest = np.concatenate([static, moving], axis=0)
        sequence = np.stack([rest, rest.copy(), rest.copy()])
        sequence[1, -4:, 1] += 1.0
        sequence[2, -4:, 1] += 2.0

        first = motion_cluster_labels(sequence, 2, seed=7)
        second = motion_cluster_labels(sequence, 2, seed=7)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 2)
        self.assertEqual(len(np.unique(first[-4:])), 1)
        self.assertNotEqual(first[-1], first[0])

    def test_backend_adapter_and_reconstruction(self):
        points = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            ],
            dtype=float,
        )
        backend = _FakeDemBones()
        result = decompose_ssdr(
            DynamicPointSequence(points), SSDRConfig(num_bones=2), backend=backend
        )
        self.assertEqual(result.weights.shape, (4, 2))
        self.assertEqual(result.transformations.shape, (2, 2, 4, 4))
        np.testing.assert_allclose(result.reconstruction, points)
        np.testing.assert_allclose(result.weights.sum(axis=1), 1)
        self.assertEqual(backend.trajectory.shape, (6, 4))

    def test_configured_rest_frame_is_sent_to_backend_first(self):
        points = np.array(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[4, 0, 0], [5, 0, 0]],
            ],
            dtype=float,
        )
        backend = _FakeDemBones()
        decompose_ssdr(
            DynamicPointSequence(points),
            SSDRConfig(num_bones=1, rest_frame=1),
            backend=backend,
        )
        np.testing.assert_allclose(backend.trajectory[:3].T, points[1])


class MeshExtractionTest(unittest.TestCase):
    def test_boundary_loop_closure_only_caps_the_open_surface(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        open_faces = np.array(
            [[0, 2, 1], [0, 1, 3], [2, 0, 3]], dtype=int
        )
        closed = close_boundary_loops(TriangleMesh(vertices, open_faces))
        self.assertTrue(is_watertight(closed))
        self.assertEqual(len(closed.vertices), 5)
        self.assertEqual(len(closed.faces), 6)

    def test_registered_faces_and_off_round_trip(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        faces = np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int
        )
        sequence = DynamicPointSequence(vertices[None], faces=faces)
        mesh = prepare_mesh(sequence, MeshConfig())
        self.assertTrue(is_watertight(mesh))
        with tempfile.TemporaryDirectory() as directory:
            path = write_off(Path(directory) / "rest.off", mesh)
            loaded = read_off(path)
        np.testing.assert_allclose(loaded.vertices, vertices)
        np.testing.assert_array_equal(loaded.faces, faces)

    def test_open_registered_faces_fall_back_to_poisson(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
        )
        open_faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)
        expected = TriangleMesh(
            vertices,
            np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        )
        sequence = DynamicPointSequence(vertices[None], faces=open_faces)
        with mock.patch(
            "skelebone.extraction.mesh.reconstruct_poisson",
            return_value=expected,
        ) as reconstruct:
            with self.assertWarns(RuntimeWarning):
                result = prepare_mesh(sequence, MeshConfig())
        reconstruct.assert_called_once()
        self.assertIs(result, expected)


class CurveSkinningTest(unittest.TestCase):
    def test_external_curve_extractor_protocol(self):
        mesh = TriangleMesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            np.array([[0, 1, 2]]),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "fake_curve.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "prefix = pathlib.Path(sys.argv[2])\n"
                "pathlib.Path(str(prefix) + '_points.txt').write_text('0 0 0\\n1 0 0\\n')\n"
                "pathlib.Path(str(prefix) + '_edges.txt').write_text('0 1\\n')\n",
                encoding="ascii",
            )
            executable.chmod(0o755)
            result = extract_curve_skeleton(
                mesh,
                directory / "result",
                CurveExtractionConfig(str(executable), correspondence_neighbors=1),
            )
        self.assertEqual(len(result.skeleton.points), 2)
        np.testing.assert_array_equal(result.skeleton.edges, [[0, 1]])
        self.assertEqual(result.correspondence.indices.shape, (3, 1))

    def test_knn_transfer_and_curve_projection(self):
        source_points = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
        source_weights = np.eye(2)
        mesh_vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        mesh_weights = transfer_skinning_to_mesh(
            mesh_vertices, source_points, source_weights, k=2
        )
        np.testing.assert_allclose(mesh_weights[0], [1, 0])
        np.testing.assert_allclose(mesh_weights[1], [0.5, 0.5])
        np.testing.assert_allclose(mesh_weights[2], [0, 1])

        curve = CurveSkeleton(mesh_vertices, np.array([[0, 1], [1, 2]]))
        correspondence = vertex_curve_correspondence(mesh_vertices, curve.points, k=2)
        curve_weights = project_skinning_to_curve(mesh_weights, correspondence, curve)
        np.testing.assert_allclose(curve_weights, mesh_weights)

    def test_projection_fills_curve_samples_without_surface_support(self):
        curve = CurveSkeleton(
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            np.array([[0, 1], [1, 2]]),
        )
        correspondence = Correspondence(
            indices=np.array([[0], [2]]),
            distances=np.zeros((2, 1)),
            weights=np.ones((2, 1)),
        )
        result = project_skinning_to_curve(np.eye(2), correspondence, curve)
        np.testing.assert_allclose(result.sum(axis=1), 1)
        self.assertTrue(np.all(result >= 0))


class JointTreeTest(unittest.TestCase):
    def test_balanced_skinning_transition_is_a_joint(self):
        points = np.stack([np.arange(7), np.zeros(7), np.zeros(7)], axis=1)
        curve = CurveSkeleton(points, np.array([[i, i + 1] for i in range(6)]))
        first = np.array([1.0, 0.95, 0.8, 0.5, 0.2, 0.05, 0.0])
        weights = np.stack([first, 1 - first], axis=1)
        detected = detect_joints(
            curve,
            weights,
            JointConfig(score_threshold=0.2, minimum_distance=0, peak_window_hops=2),
        )
        np.testing.assert_array_equal(detected.curve_indices, [3])
        np.testing.assert_allclose(detected.scores, [1.0])

    def test_y_curve_compresses_to_an_oriented_tree(self):
        points = np.array(
            [
                [0, 0, 0], [1, 0, 0], [2, 0, 0],
                [0, 1, 0], [0, 2, 0],
                [-1, 0, 0], [-2, 0, 0],
            ],
            dtype=float,
        )
        curve = CurveSkeleton(
            points,
            np.array([[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [5, 6]]),
        )
        joints = JointDetection(points[[0]], np.array([0]), np.array([1.0]), np.ones(7))
        tree = build_kinematic_tree(curve, joints)
        self.assertEqual(tree.skeleton.num_joints, 4)
        self.assertEqual(len(tree.edges), 3)
        self.assertEqual(tree.node_types.count("endpoint"), 3)
        self.assertEqual(tree.node_types.count("joint"), 1)
        self.assertEqual(len(tree.skeleton.roots), 1)
        self.assertTrue(all(len(path) == 3 for path in tree.curve_paths))

    def test_disconnected_curve_receives_one_virtual_root(self):
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float
        )
        curve = CurveSkeleton(points, np.array([[0, 1], [2, 3]]))
        joints = JointDetection(
            np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0), np.zeros(4)
        )
        tree = build_kinematic_tree(curve, joints)
        self.assertEqual(len(tree.skeleton.roots), 1)
        self.assertEqual(tree.node_types[-1], "virtual_root")
        self.assertEqual(tree.curve_indices[-1], -1)
        self.assertEqual(len(tree.edges), tree.skeleton.num_joints - 1)


class TrackingTest(unittest.TestCase):
    def test_local_rigid_tracking_preserves_medial_offsets(self):
        rest = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float
        )
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        translation = np.array([3, -2, 0], dtype=float)
        moved = rest @ rotation.T + translation
        joints = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=float)
        tracked = track_joints(
            DynamicPointSequence(np.stack([rest, moved]), fps=24),
            joints,
            TrackingConfig(neighbors=4),
        )
        np.testing.assert_allclose(tracked.positions[0], joints)
        np.testing.assert_allclose(tracked.positions[1], joints @ rotation.T + translation)
        self.assertEqual(tracked.fps, 24)


if __name__ == "__main__":
    unittest.main()
