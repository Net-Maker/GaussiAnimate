import unittest
from pathlib import Path

from experiments.audit import audit_catalog
from experiments.paper import expand_jobs, load_paper_config, run_job


class ExperimentAuditTest(unittest.TestCase):
    def test_catalog_explicitly_reports_not_ready(self):
        report = audit_catalog(Path("experiments/catalog.yaml"))
        self.assertFalse(report["ready"])
        self.assertEqual(report["targets"], 10)
        self.assertFalse(report["missing_configs"])
        self.assertGreater(report["state_counts"]["blocked"], 0)
        self.assertEqual(
            {record["target_id"] for record in report["records"]},
            {
                "arap_ssdr_ablation",
                "d4d_animal_animation",
                "human_novel_pose",
                "multiview_d3dgs_reconstruction",
                "partmm_database_scaling",
                "partmm_patch_length",
                "partmm_runtime",
                "progressive_skeletonization",
                "synthetic_novel_view",
                "vto_garment_animation",
            },
        )
        records = {
            record["target_id"]: record for record in report["records"]
        }
        self.assertEqual(
            records["synthetic_novel_view"]["target"],
            "Table 2 / unseen-view synthetic reconstruction",
        )
        self.assertEqual(
            records["human_novel_pose"]["target"],
            "Table 3 / unseen-pose clothed-human rendering",
        )
        self.assertEqual(
            records["partmm_runtime"]["target"],
            "Fig. R.2 / PartMM runtime",
        )
        d4d_blockers = " ".join(
            records["d4d_animal_animation"]["blockers"]
        )
        self.assertIn("rounds to 53.4", d4d_blockers)
        self.assertIn("inverse-distance-to-bone", d4d_blockers)
        self.assertIn("nearest-neighbor retrieval", d4d_blockers)

    def test_non_novel_view_audit_excludes_only_table_2(self):
        report = audit_catalog(
            Path("experiments/catalog.yaml"),
            exclude_target_ids={"synthetic_novel_view"},
        )
        self.assertEqual(report["targets"], 9)
        self.assertEqual(report["excluded_target_ids"], ["synthetic_novel_view"])
        self.assertNotIn(
            "synthetic_novel_view",
            {record["target_id"] for record in report["records"]},
        )

    def test_human_partmm_uses_the_public_train_only_pipeline(self):
        config = load_paper_config(
            "experiments/multiview_gaussian/novel_pose.yaml"
        )
        job = next(
            job
            for job in expand_jobs(config, methods=["smpl_partmm"])
            if job.case["id"] == "dna_0012_11"
        )
        dry_run = run_job(job, dry_run=True)
        self.assertEqual(dry_run["mode"], "external")
        stage_ids = [stage["id"] for stage in dry_run["stages"]]
        self.assertEqual(
            stage_ids,
            [
                "fit_ssdr",
                "export_smpl_motion",
                "animate",
                "evaluate",
                "collect",
            ],
        )
        fit_command = dry_run["stages"][0]["command"]
        animate_command = dry_run["stages"][2]["command"]
        self.assertIn("--train-ratio", fit_command)
        self.assertIn("0.8", [str(value) for value in fit_command])
        self.assertIn(
            "experiments.multiview_gaussian.paper_partmm",
            animate_command,
        )
        self.assertIn("--implementation-root", animate_command)
        self.assertIn("paper_spec", animate_command)

    def test_human_lbs_uses_public_robust_all_gaussian_pipeline(self):
        config = load_paper_config(
            "experiments/multiview_gaussian/novel_pose.yaml"
        )
        job = next(
            job
            for job in expand_jobs(config, methods=["smpl_lbs"])
            if job.case["id"] == "dna_0012_11"
        )
        stages = run_job(job, dry_run=True)["stages"]
        self.assertEqual(
            [stage["id"] for stage in stages],
            ["export_smpl_motion", "fit", "evaluate", "collect"],
        )
        fit_command = stages[1]["command"]
        self.assertIn(
            "experiments.multiview_gaussian.human_lbs",
            fit_command,
        )
        self.assertNotIn("lbs_drive_gaussians.py", fit_command)
        self.assertIn("--target-triangles", fit_command)
        self.assertTrue(
            any(
                output.endswith("/point_indices.npy")
                for output in stages[1]["outputs"]
            )
        )

    def test_human_bob_disables_gt_filled_point_filtering(self):
        config = load_paper_config(
            "experiments/multiview_gaussian/novel_pose.yaml"
        )
        job = next(
            job
            for job in expand_jobs(config, methods=["smpl_bob"])
            if job.case["id"] == "dna_0012_11"
        )
        stages = run_job(job, dry_run=True)["stages"]
        self.assertEqual(
            [stage["id"] for stage in stages],
            ["export_smpl_motion", "fit", "evaluate", "collect"],
        )
        command = stages[1]["command"]
        self.assertIn(
            "experiments.multiview_gaussian.human_bob",
            command,
        )
        self.assertNotIn("--max_points", command)
        self.assertNotIn("--opacity_threshold", command)
        protocol = job.method["protocol"]
        self.assertEqual(protocol["pose_joint_range"], [1, 24])
        self.assertEqual(protocol["pose_dimensions"], 69)
        self.assertEqual(protocol["skeleton_links"], 23)
        self.assertEqual(
            protocol["root_handling"],
            "separate_excluded_from_rootless_fk",
        )
        self.assertEqual(
            protocol["predicted_gaussian_attributes"],
            ["means3D"],
        )
        self.assertEqual(
            protocol["oracle_evaluation_attributes"],
            ["rgb_colors", "unnorm_rotations"],
        )
        self.assertTrue(
            any(
                output.endswith("/point_indices.npy")
                for output in stages[1]["outputs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
