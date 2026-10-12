import tempfile
import unittest
from pathlib import Path

import numpy as np

from skelebone.io import load_bundle
from experiments.mesh.migrate import migrate_prediction


class MigrationTest(unittest.TestCase):
    def test_prediction_cache_migration(self):
        frames, joints, bones, points = 4, 3, 2, 5
        rest_joints = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        local_rotation = np.broadcast_to(np.eye(3), (frames, joints, 3, 3)).copy()
        transforms = np.broadcast_to(np.eye(4), (frames, bones, 4, 4)).copy()
        weights = np.full((bones, points), 0.5)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "prediction_data.npz"
            np.savez(
                source,
                skeleton_rest_positions=rest_joints,
                skeleton_parents=np.array([-1, 0, 1]),
                skeleton_local_rotations=local_rotation,
                skeleton_root_positions=np.zeros((frames, 3)),
                ssdr_weights=weights,
                ssdr_transformations=transforms,
                ssdr_rest_pose=np.zeros((points, 3)),
            )
            migrate_prediction(source, directory / "public", source_end=3)
            bundle = load_bundle(directory / "public" / "skelebones.npz")
            self.assertEqual(bundle.motion.num_frames, 3)
            self.assertEqual(bundle.skeleton.num_joints, joints)
            self.assertEqual(bundle.num_bones, bones)
            self.assertTrue((directory / "public" / "motion.bvh").is_file())


if __name__ == "__main__":
    unittest.main()
