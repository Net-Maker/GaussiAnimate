import unittest
import importlib.util
import os
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
import partmm
import partmm.visualization as partmm_visualization


def _minimal_input():
    rotations = np.zeros((3, 2, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    transforms = np.broadcast_to(np.eye(4), (3, 1, 4, 4)).copy()
    transforms[:, 0, 0, 3] = [0.0, 0.5, 1.0]
    return partmm.PartMMInput(
        rest_joints=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        parents=np.array([-1, 0]),
        source_local_rotations=rotations,
        source_root_translation=np.zeros((3, 3)),
        bone_transforms=transforms,
        source_points=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        skinning_weights=np.ones((2, 1)),
        fps=24.0,
        joint_names=("root", "tip"),
    )


def _write_hood_tshirt_fixture(root):
    hood_root = root / "hood"
    vto_root = root / "vto"
    simulations = vto_root / "tshirt" / "simulations"
    ssdr_directory = hood_root / "output" / "vto_ssdr_results"
    reference_directory = hood_root / "output" / "precomputed_tshirt"
    simulations.mkdir(parents=True)
    ssdr_directory.mkdir(parents=True)
    reference_directory.mkdir(parents=True)

    database_cases = (
        "tshirt_shape00_01_01",
        "tshirt_shape00_02_04",
        "tshirt_shape00_02_05",
        "tshirt_shape00_05_20",
        "tshirt_shape00_07_02",
    )
    query_case = "tshirt_shape00_91_33"
    frame_count = 9
    transforms_per_motion = {}
    for case_index, case in enumerate(database_cases + (query_case,)):
        pose = np.zeros((frame_count, 72), dtype=np.float32)
        pose[:, 3 + case_index % 4] = np.linspace(0.0, 0.2, frame_count)
        vertices = np.zeros((frame_count, 4, 3), dtype=np.float32)
        vertices[:, :, 0] = np.linspace(0.0, 0.1, frame_count)[:, None]
        with (simulations / f"{case}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "pose": pose,
                    "translation": np.zeros((frame_count, 3), dtype=np.float32),
                    "vertices": vertices,
                    "faces": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
                },
                handle,
            )
        if case in database_cases:
            transforms = np.broadcast_to(
                np.eye(4), (frame_count, 2, 4, 4)
            ).copy()
            transforms[:, :, 0, 3] = np.linspace(0.0, 0.1, frame_count)[:, None]
            transforms_per_motion[case] = transforms

    np.savez_compressed(
        ssdr_directory / "tshirt_shape00_ALL_bones20.npz",
        transforms_per_motion=np.asarray(transforms_per_motion, dtype=object),
        weights=np.array(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
        ),
        rest_pose=np.array(
            [[-0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 0.0]]
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        num_bones=np.asarray(2),
    )
    rest_joints = np.stack(
        [np.array([0.0, float(joint) * 0.05, 0.0]) for joint in range(24)]
    )
    skeleton_joints = np.broadcast_to(
        rest_joints, (frame_count, 24, 3)
    ).copy()
    skeleton_joints[:, :, 2] += np.linspace(0.0, 0.08, frame_count)[:, None]
    gt_vertices = np.broadcast_to(
        np.array(
            [[-0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.0, 0.0]]
        ),
        (frame_count, 4, 3),
    ).copy()
    np.savez_compressed(
        reference_directory / f"{query_case}.npz",
        gt_vertices=gt_vertices,
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        skeleton_joints=skeleton_joints,
        seq_name=np.asarray(query_case),
        partmm_rmse=np.asarray(12.5),
    )
    return hood_root, vto_root, database_cases, query_case


class PartMMStandaloneTest(unittest.TestCase):
    def test_open3d_renderer_stays_inside_standalone_package(self):
        self.assertIsNotNone(importlib.util.find_spec("partmm.open3d_renderer"))

    def test_partmm_gif_writer_is_multiframe(self):
        frames = [
            np.full((6, 6, 3), 255, dtype=np.uint8),
            np.zeros((6, 6, 3), dtype=np.uint8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = partmm_visualization.save_animation_gif(
                Path(directory) / "preview.gif", frames, fps=8
            )
            from PIL import Image

            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 2)

    def test_outer_bone_positions_follow_retrieved_transforms(self):
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        weights = np.eye(2)
        transforms = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
        transforms[0, 0, 1, 3] = 1.0
        transforms[0, 1, 2, 3] = 3.0

        positions = partmm_visualization.outer_bone_positions(
            points, weights, transforms
        )

        np.testing.assert_allclose(
            positions, [[[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]]]
        )

    def test_public_array_and_file_contract_is_exposed(self):
        expected = {
            "PartMMInput",
            "PartMMQuery",
            "PartMMOutput",
            "load_input",
            "load_query",
            "load_output",
            "save_input",
            "save_query",
            "save_output",
        }
        self.assertEqual(expected.difference(vars(partmm)), set())

    def test_package_has_no_skelebone_runtime_imports(self):
        package_root = Path(partmm.__file__).parent
        coupled = []
        for path in package_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "from skelebone" in text or "import skelebone" in text:
                coupled.append(path.relative_to(package_root).as_posix())
        self.assertEqual(coupled, [])

    def test_direct_numpy_contract_animates_without_skelebone_types(self):
        data = _minimal_input()
        query = partmm.PartMMQuery(
            data.source_local_rotations.copy(),
            data.source_root_translation.copy(),
            data.fps,
        )
        result = partmm.animate(
            data,
            query,
            partmm.PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            ),
        )
        self.assertEqual(result.points.shape, (3, 2, 3))
        self.assertEqual(result.bone_transforms.shape, (3, 1, 4, 4))
        self.assertEqual(result.joint_positions.shape, (3, 2, 3))
        self.assertAlmostEqual(result.fps, 24.0)

    def test_npz_contract_round_trips_input_query_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = _minimal_input()
            query = partmm.PartMMQuery(
                data.source_local_rotations,
                data.source_root_translation,
                data.fps,
            )
            config = partmm.PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            )
            output = partmm.animate(data, query, config)

            partmm.save_input(root / "input.npz", data)
            partmm.save_query(root / "query.npz", query)
            partmm.save_output(root / "output.npz", output)

            loaded_input = partmm.load_input(root / "input.npz")
            loaded_query = partmm.load_query(root / "query.npz")
            loaded_output = partmm.load_output(root / "output.npz")
            np.testing.assert_allclose(loaded_input.bone_transforms, data.bone_transforms)
            np.testing.assert_allclose(loaded_query.local_rotations, query.local_rotations)
            np.testing.assert_allclose(loaded_output.points, output.points)
            self.assertEqual(set(loaded_output.matched_indices), set(output.matched_indices))

    def test_copied_package_imports_and_runs_outside_repository(self):
        package_root = Path(partmm.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            shutil.copytree(package_root, project / "partmm", ignore=shutil.ignore_patterns("__pycache__"))
            script = """
import numpy as np
import partmm

rotations = np.zeros((2, 2, 4))
rotations[..., 0] = 1.0
data = partmm.PartMMInput(
    rest_joints=np.array([[0., 0., 0.], [0., 1., 0.]]),
    parents=np.array([-1, 0]),
    source_local_rotations=rotations,
    source_root_translation=np.zeros((2, 3)),
    bone_transforms=np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy(),
    source_points=np.array([[0., 0., 0.], [0., 1., 0.]]),
    skinning_weights=np.ones((2, 1)),
)
config = partmm.PartMMConfig(num_parts=1, patch_size=1, num_neighbors=1,
                             pyramid_levels=1, align_parts=False)
result = partmm.animate(data, partmm.PartMMQuery.identity(2, 2), config)
assert result.points.shape == (2, 2, 3)
assert 'GaussiAnimate' not in str(partmm.__file__)
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project)
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=project,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_temporal_preview_writes_a_nonempty_png(self):
        from partmm.visualization import save_preview

        data = _minimal_input()
        output = partmm.animate(
            data,
            partmm.PartMMQuery(
                data.source_local_rotations,
                data.source_root_translation,
                data.fps,
            ),
            partmm.PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            preview = save_preview(
                Path(directory) / "preview.png",
                output,
                data.parents,
            )
            self.assertEqual(preview.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(preview.stat().st_size, 1000)

    def test_preview_projection_keeps_positive_y_vertical(self):
        """The Matplotlib preview camera must keep data-space +Y vertical.

        SMPL/HOOD data is Y-up, matching the explicit up=[0, 1, 0] in
        partmm.open3d_renderer._fit_camera, so a +Y segment must project
        predominantly onto the vertical screen axis.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d

        data = _minimal_input()
        output = partmm.animate(
            data,
            partmm.PartMMQuery(
                data.source_local_rotations,
                data.source_root_translation,
                data.fps,
            ),
            partmm.PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            ),
        )
        figure = partmm_visualization.plot_preview(output, data.parents)
        try:
            horizontal, vertical, _ = proj3d.proj_transform(
                np.array([0.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 0.0]),
                figure.axes[0].get_proj(),
            )
            rise = float(vertical[1] - vertical[0])
            run = float(horizontal[1] - horizontal[0])
            self.assertGreater(rise, 0.0)
            self.assertGreater(abs(rise), abs(run))
        finally:
            plt.close(figure)

    def test_preview_reports_missing_matplotlib_dependency(self):
        """plot_preview keeps raising the actionable ImportError when
        matplotlib is absent; the optional dependency stays lazy."""
        data = _minimal_input()
        output = partmm.animate(
            data,
            partmm.PartMMQuery(
                data.source_local_rotations,
                data.source_root_translation,
                data.fps,
            ),
            partmm.PartMMConfig(
                num_parts=1,
                patch_size=1,
                num_neighbors=1,
                pyramid_levels=1,
                align_parts=False,
            ),
        )
        with mock.patch.dict(
            sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}
        ):
            with self.assertRaises(ImportError) as context:
                partmm_visualization.plot_preview(output, data.parents)
        self.assertIn("matplotlib", str(context.exception))

    def test_quickstart_example_writes_portable_artifacts_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hood_root, vto_root, database_cases, query_case = (
                _write_hood_tshirt_fixture(root)
            )
            output_directory = root / "result"
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(root / ".mpl")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "partmm.examples.quickstart",
                    str(output_directory),
                    "--hood-root",
                    str(hood_root),
                    "--vto-root",
                    str(vto_root),
                    "--case",
                    query_case,
                    "--database-cases",
                    *database_cases,
                    "--num-parts",
                    "1",
                    "--patch-size",
                    "1",
                    "--neighbors",
                    "1",
                    "--pyramid-levels",
                    "1",
                    "--partition-mode",
                    "balanced_tree",
                    "--no-align",
                    "--preview-format",
                    "png",
                ],
                cwd=Path(partmm.__file__).parent.parent,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            expected = {
                "partmm_input.npz",
                "partmm_query.npz",
                "partmm_output.npz",
                "partmm_preview.png",
                "metrics.json",
            }
            self.assertEqual(
                expected.difference(path.name for path in output_directory.iterdir()),
                set(),
            )
            output = partmm.load_output(output_directory / "partmm_output.npz")
            self.assertEqual(output.points.shape[0], 9)
            self.assertEqual(output.points.shape[1], 4)
            with np.load(
                hood_root
                / "output/precomputed_tshirt"
                / f"{query_case}.npz",
                allow_pickle=False,
            ) as reference:
                np.testing.assert_allclose(
                    output.joint_positions,
                    reference["skeleton_joints"],
                )
            metrics = json.loads(
                (output_directory / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("recomputed_rmse_mm", metrics)
            self.assertEqual(metrics["historical_partmm_rmse_mm"], 12.5)


if __name__ == "__main__":
    unittest.main()
