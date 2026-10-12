import tempfile
import unittest
from pathlib import Path

from experiments.monocular_gaussian.arap_evidence import audit_arap_evidence


class ArapEvidenceTest(unittest.TestCase):
    def test_missing_no_arap_variant_keeps_ablation_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "outputs/stage1_res/girlwalk_512_node"
            checkpoint = stage1 / "deform/iteration_40000/deform.pth"
            point_cloud = stage1 / "point_cloud/iteration_40000/point_cloud.ply"
            checkpoint.parent.mkdir(parents=True)
            point_cloud.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            point_cloud.write_bytes(b"point cloud")
            final = (
                root
                / "saved_final_results_for_skeleton"
                / "girlwalk_512_final_node"
            )
            ssdr = final / "ssdr_results_bones50.npz"
            skeleton = final / "inner_skeleton_refined/inner_skeleton.npz"
            ssdr.parent.mkdir(parents=True)
            skeleton.parent.mkdir(parents=True)
            ssdr.write_bytes(b"ssdr")
            skeleton.write_bytes(b"skeleton")

            result = audit_arap_evidence(root)
            self.assertFalse(result["all_ready"])
            self.assertEqual(
                result["variants"]["no_arap_ssdr"]["status"],
                "missing_historical_checkpoint_requires_retraining",
            )
            self.assertEqual(
                result["variants"]["arap_ssdr"]["status"],
                "historical_result_complete",
            )


if __name__ == "__main__":
    unittest.main()
