import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.mesh.export_dt4d import export_dt4d_points
from experiments.mesh.export_vto import export_vto_points
from experiments.monocular_gaussian.prepare import export_riggs_points
from experiments.multiview_gaussian.prepare import export_d3dgs_points
from skelebone.io import load_point_sequence


class ExporterTest(unittest.TestCase):
    def test_riggs_registered_frame_export(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frames = np.arange(3 * 6 * 3, dtype=np.float32).reshape(3, 6, 3)
            for index, frame in enumerate(frames):
                np.save(directory / f"frame_{index:04d}.npy", frame)
            output = export_riggs_points(
                directory,
                directory / "points.npz",
                pattern="frame_*.npy",
                max_points=4,
                seed=7,
                fps=24,
            )
            result = np.load(output, allow_pickle=False)
            indices = result["source_indices"]
            np.testing.assert_allclose(result["points"], frames[:, indices])
            self.assertEqual(result["points"].shape, (3, 4, 3))
            self.assertEqual(float(result["fps"]), 24)

    def test_d3dgs_foreground_export(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            params = directory / "params.npz"
            means = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
            colors = np.zeros((5, 3), dtype=np.float32)
            colors[[1, 3], 0] = 1
            np.savez(params, means3D=means, seg_colors=colors)
            output = export_d3dgs_points(params, directory / "points.npz")
            sequence = load_point_sequence(output)
            self.assertEqual(sequence.points.shape, (3, 2, 3))
            np.testing.assert_allclose(sequence.points, means[:, [1, 3]])

    def test_d3dgs_high_motion_window_and_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            means = np.zeros((8, 2, 3), dtype=np.float32)
            means[4:] += 10
            params = directory / "params.npz"
            np.savez(params, means3D=means, seg_colors=np.ones((2, 3)))
            output = export_d3dgs_points(
                params,
                directory / "points.npz",
                high_motion_window=4,
                frame_stride=2,
            )
            result = np.load(output, allow_pickle=False)
            np.testing.assert_array_equal(result["source_frame_indices"], [1, 3, 4])
            self.assertEqual(float(result["fps"]), 15)

    def test_dt4d_anime_export(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            anime = directory / "motion.anime"
            rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
            faces = np.array([[0, 1, 2]], np.int32)
            offsets = np.ones((2, 3, 3), np.float32)
            with anime.open("wb") as handle:
                np.array([3, 3, 1], np.int32).tofile(handle)
                rest.tofile(handle)
                faces.tofile(handle)
                offsets.tofile(handle)
            output = export_dt4d_points(anime, directory / "points.npz", frame_stride=2)
            sequence = load_point_sequence(output)
            self.assertEqual(sequence.points.shape, (2, 3, 3))
            np.testing.assert_allclose(sequence.points[-1], rest + 1)
            np.testing.assert_array_equal(sequence.faces, faces)

    def test_vto_export_and_root_canonicalization(self):
        canonical = np.array(
            [[[1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 1, 0]]], dtype=float
        )
        world = canonical.copy()
        world[1] = canonical[1] @ np.array(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float
        )
        world[1] += [2, 0, 0]
        pose = np.zeros((2, 72), dtype=float)
        pose[1, 2] = np.pi / 2
        data = {
            "vertices": world,
            "faces": np.array([[0.0, 1.0, 1.0]], dtype=np.float32),
            "pose": pose,
            "translation": np.array([[0, 0, 0], [2, 0, 0]], dtype=float),
            "sequence": "motion_01",
            "subject": "shape00",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "simulation.pkl"
            output = Path(directory) / "sequence.npz"
            with source.open("wb") as handle:
                pickle.dump(data, handle)
            export_vto_points(source, output, pelvis_offset=[0, 0, 0])
            result = np.load(output, allow_pickle=False)
            np.testing.assert_allclose(result["points"], canonical, atol=1e-6)
            self.assertTrue(bool(result["root_canonicalized"]))
            self.assertEqual(str(result["sequence"]), "motion_01")


if __name__ == "__main__":
    unittest.main()
