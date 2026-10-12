import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.mesh.dt4d_evidence import audit_dt4d_evidence


class Dt4dEvidenceTest(unittest.TestCase):
    def test_rejects_a_non_frozen_instance_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"results": []}))
            digest = __import__("hashlib").sha256(
                path.read_bytes()
            ).hexdigest()
            with patch(
                "experiments.mesh.dt4d_evidence.EXPECTED_SHA256",
                digest,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "26 D4D instances",
                ):
                    audit_dt4d_evidence(path)


if __name__ == "__main__":
    unittest.main()
