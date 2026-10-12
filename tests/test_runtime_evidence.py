import json
import tempfile
import unittest
from pathlib import Path

from experiments.mesh.runtime import load_runtime_detail


class RuntimeEvidenceTest(unittest.TestCase):
    def _record(self):
        build_binding = 7.0
        build_databases = 1.0
        ssdr = 2.0
        fk = 0.5
        retarget = 1.5
        reconstruct = 0.03
        offline = build_binding + build_databases + ssdr
        online = fk + retarget + reconstruct
        return {
            "config": {
                "garment": "tshirt",
                "n_train": 5,
                "n_train_frames": 840,
                "n_test_frames": 203,
                "target_seq": "tshirt_shape00_91_33",
                "num_bones": 20,
            },
            "timing": {
                "build_part_binding_s": build_binding,
                "build_part_databases_s": build_databases,
                "smpl_fk_s": fk,
                "retarget_motion_s": retarget,
                "reconstruct_s": reconstruct,
                "ssdr_decomposition_s": ssdr,
                "offline_total_s": offline,
                "online_total_s": online,
                "end_to_end_s": offline + online,
                "online_per_frame_ms": online * 1000.0 / 203,
            },
        }

    def test_validates_protocol_and_recomputes_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(self._record()), encoding="utf-8")
            result = load_runtime_detail(path)
        self.assertEqual(result["config"]["n_test_frames"], 203)
        self.assertAlmostEqual(result["metrics"]["online_total_s"], 2.03)

    def test_rejects_inconsistent_timing(self):
        record = self._record()
        record["timing"]["online_total_s"] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent totals"):
                load_runtime_detail(path)


if __name__ == "__main__":
    unittest.main()
