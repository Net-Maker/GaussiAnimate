import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.multiview_gaussian.partmm_repair import run_case


class RepairProtocolTest(unittest.TestCase):
    def test_test_geometry_cannot_change_fallback_decision_or_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/multiview_novel_pose/test/example'
            params_dir = root / 'resources/multiview/d3dgs/test/example'
            params_dir.mkdir(parents=True)
            for path in ['smpl_partmm/ssdr', 'smpl_partmm/motion',
                         'smpl_partmm/partmm/example', 'smpl_lbs/lbs']:
                (run / path).mkdir(parents=True)
            rest = np.array([[0., 0., 0.], [.1, .2, .3]])
            observed = np.broadcast_to(rest, (10, 2, 3)).copy()
            np.savez(params_dir / 'params.npz', means3D=observed)
            np.savez(run / 'smpl_partmm/ssdr/train_only.npz', rest_pose=rest,
                     weights=np.ones((1, 2)), world_transformations=np.tile(np.eye(4), (8, 1, 1, 1)))
            np.savez(run / 'smpl_partmm/motion/smpl_motion.npz', rest_joints=np.zeros((1, 3)),
                     parents=[-1], names=['root'], fps=30.,
                     local_rotations=np.tile([1., 0, 0, 0], (10, 1, 1)), root_translation=np.zeros((10, 3)))
            np.save(run / 'smpl_partmm/partmm/example/test_warped_sequence.npy', observed[8:])
            np.save(run / 'smpl_lbs/lbs/warped_sequence.npy', observed)
            np.savez(run / 'smpl_lbs/lbs/robust_weights.npz', gaussian_weights=np.ones((2, 1)))
            (run / 'smpl_lbs/lbs/metadata.json').write_text(json.dumps({'protocol': {'fit_frames': [0]}}))
            settings = dict(release_root=str(root), output_dir=str(root / 'output'), train_ratio=.8,
                            validation_ratio=.25, distance_quantile=.95, seed=42,
                            distance_policy='validation_error', save_predictions=True,
                            partmm=dict(num_parts=1, patch_size=1, num_neighbors=1, pyramid_levels=1))
            case = dict(dataset='test', id='example')
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_case(settings, case)
                prediction = np.load(root / 'output/example/test_warped_sequence.npy')
                observed[8:] += 100
                np.savez(params_dir / 'params.npz', means3D=observed)
                second = run_case(settings, case)
            np.testing.assert_array_equal(prediction, np.load(root / 'output/example/test_warped_sequence.npy'))
            self.assertEqual(first['fallback'], second['fallback'])
            self.assertEqual(first['fallback']['baseline_fraction'], 1.)
            self.assertLess(first['metrics']['guarded']['point_rmse_mm'], 1e-8)
            self.assertGreater(second['metrics']['guarded']['point_rmse_mm'], 100000)


if __name__ == '__main__':
    unittest.main()
