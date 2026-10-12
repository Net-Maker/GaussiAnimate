import json
import tempfile
import unittest
from pathlib import Path

from experiments.monocular_gaussian.progressive_evidence import (
    adopt_progressive_evidence,
    audit_progressive_evidence,
)


class ProgressiveEvidenceTest(unittest.TestCase):
    def _make_case(self, root: Path, percentage: int, frames: int, skeleton: bool):
        case = (
            root
            / "saved_final_results_for_skeleton"
            / "ablation_frames"
            / "trex_512_final_node"
            / f"{percentage:03d}pct"
        )
        case.mkdir(parents=True)
        summary = {
            "percentage": percentage,
            "num_frames_used": frames,
            "total_frames": 200,
            "num_points": 30000,
            "num_bones_requested": 50,
        }
        (case / "ablation_summary.json").write_text(json.dumps(summary))
        (case / "ssdr_results_bones50.npz").write_bytes(b"ssdr")
        if skeleton:
            path = case / "inner_skeleton/inner_skeleton.npz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"skeleton")

    def test_audit_distinguishes_complete_and_ssdr_only_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = (
                root
                / "saved_final_results_for_skeleton"
                / "trex_512_final_node"
                / "deformed_pointclouds"
            )
            frames.mkdir(parents=True)
            for index in range(200):
                (frames / f"frame_{index:04d}.ply").write_bytes(b"ply")
            mesh = frames.parent / "mesh/frame_0000_post.ply"
            mesh.parent.mkdir()
            mesh.write_bytes(b"mesh")
            for percentage, count in ((30, 60), (60, 120), (80, 160)):
                self._make_case(root, percentage, count, True)
            self._make_case(root, 100, 200, False)

            result = audit_progressive_evidence(root)
            self.assertFalse(result["all_ready"])
            self.assertEqual(result["complete_columns"], 3)
            self.assertEqual(result["columns"][0]["status"], "missing_exact_historical_artifact")
            self.assertEqual(
                result["columns"][-1]["status"],
                "recovered_exact_protocol_ssdr_only",
            )

            output = root / "release"
            manifest = adopt_progressive_evidence(result, root, output)
            self.assertTrue(manifest.is_file())
            self.assertTrue((output / "030pct/inner_skeleton/inner_skeleton.npz").is_file())
            self.assertFalse((output / "100pct/ssdr_results_bones50.npz").exists())


if __name__ == "__main__":
    unittest.main()
