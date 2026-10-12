import unittest

from experiments.multiview_gaussian.historical_evidence import (
    ACTORSHQ_CASES,
    PAPER,
    _actorshq_declared_subset_audit,
    _paper_check,
)


class MultiviewHistoricalEvidenceTest(unittest.TestCase):
    def test_printed_precision_check_is_metric_specific(self):
        result = _paper_check(
            {"psnr": 16.6585, "ssim": 0.7221, "lpips": 0.2594},
            {"psnr": 16.66, "ssim": 0.722, "lpips": 0.259},
        )
        self.assertTrue(result["within_printed_precision"])

    def test_declared_eight_case_audit_checks_all_subsets(self):
        bob_rows = {}
        lbs_rows = {}
        for case in ACTORSHQ_CASES:
            bob_rows[case] = {
                "oracle": dict(PAPER["actorshq"]["d3dgs_gt"]),
                "bob": dict(PAPER["actorshq"]["smpl_bob"]),
            }
            lbs_rows[case] = {
                "lbs": dict(PAPER["actorshq"]["smpl_lbs"])
            }
        result = _actorshq_declared_subset_audit(bob_rows, lbs_rows)
        self.assertEqual(result["tested_eight_sequence_subsets"], 45)
        self.assertEqual(
            len(result["subsets_matching_all_three_printed_rows"]),
            45,
        )
        self.assertTrue(result["ten_sequence_aggregate_matches"])


if __name__ == "__main__":
    unittest.main()
