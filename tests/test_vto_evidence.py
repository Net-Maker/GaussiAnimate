import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.mesh.vto_evidence import audit_vto_evidence


class VtoEvidenceTest(unittest.TestCase):
    def _write_case(self, root: Path, garment: str, case: str):
        gt = np.zeros((2, 3, 3), dtype=np.float32)
        prediction = np.full_like(gt, 0.001)
        rmse = float(np.linalg.norm(prediction - gt, axis=-1).mean() * 1000.0)
        np.savez_compressed(
            root / f"{case}.npz",
            gt_vertices=gt,
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            seq_name=case,
            garment=garment,
            mlp_vertices=prediction,
            gru_vertices=prediction,
            fullmm_vertices=prediction,
            partmm_vertices=prediction,
            mlp_rmse=rmse,
            gru_rmse=rmse,
            fullmm_rmse=rmse,
            partmm_rmse=rmse,
        )
        return {"seq_name": case, "frames": 2}

    def test_validates_arrays_and_recomputes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {"tshirt": root / "tshirt", "dress": root / "dress"}
            cases = {"tshirt": ("tshirt_case",), "dress": ("dress_case",)}
            for garment, garment_root in roots.items():
                garment_root.mkdir()
                row = self._write_case(garment_root, garment, cases[garment][0])
                (garment_root / "summary.json").write_text(json.dumps([row]))
            paper = {
                garment: {method: 1.7 for method in ("mlp", "gru", "fullmm", "partmm")}
                for garment in roots
            }
            with patch(
                "experiments.mesh.vto_evidence.EXPECTED_CASES",
                cases,
            ):
                with patch(
                    "experiments.mesh.vto_evidence.PAPER_RMSE_MM",
                    paper,
                ):
                    result = audit_vto_evidence(roots)
            self.assertTrue(result["all_arrays_valid"])
            self.assertTrue(result["all_paper_rows_reproduced"])


if __name__ == "__main__":
    unittest.main()
