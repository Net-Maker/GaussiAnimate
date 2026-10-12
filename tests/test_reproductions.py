import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.paper import expand_jobs, load_paper_config, run_job
from experiments.reproductions.sync import (
    sync_bob_availability,
    sync_cross_method,
    sync_deform2dgs_splits,
    sync_dressrecon_original,
    sync_human_novel_pose,
)


ROOT = Path("experiments/reproductions")
RESULTS = ROOT / "results"


class ReproductionConfigTest(unittest.TestCase):
    def test_all_reproduction_configs_validate_and_render(self):
        configs = []
        for path in sorted(ROOT.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if raw.get("schema_version") == 2:
                configs.append(path)
        self.assertEqual(len(configs), 5)
        expected_jobs = {
            "dressrecon_original.yaml": 28,
            "human_novel_pose.yaml": 72,
            "dressrecon_bob_recompute.yaml": 18,
            "cross_method_skeletonization.yaml": 8,
            "deform2dgs_observation_split.yaml": 2,
        }
        for path in configs:
            config = load_paper_config(path)
            jobs = expand_jobs(config)
            self.assertEqual(len(jobs), expected_jobs[path.name], path)
            for job in jobs:
                self.assertTrue(run_job(job, dry_run=True)["stages"])

    def test_checked_in_snapshots_have_no_machine_paths(self):
        for path in RESULTS.glob("*.json"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("/home/", text, path)
            self.assertNotIn("/Users/", text, path)


class ReproductionEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.human = json.loads(
            (RESULTS / "human_novel_pose.json").read_text(encoding="utf-8")
        )
        cls.cross = json.loads(
            (RESULTS / "cross_method_skeletonization.json").read_text(
                encoding="utf-8"
            )
        )

    def test_bob_matches_paper_and_covers_all_human_sequences(self):
        dna = self.human["by_dataset"]["dna_rendering"]
        actors = self.human["by_dataset"]["actorshq"]
        self.assertEqual(dna["sequence_count"], 8)
        self.assertEqual(actors["sequence_count"], 10)
        self.assertAlmostEqual(
            dna["methods"]["smpl_bob"]["metrics"]["psnr"],
            23.028511859476566,
        )
        self.assertAlmostEqual(
            actors["methods"]["smpl_bob"]["metrics"]["psnr"],
            16.65850118001302,
        )
        self.assertEqual(
            self.human["paper_consistency"]["actorshq"]["smpl_bob"]["status"],
            "matches_rounded_paper",
        )

    def test_ssdr_proxy_is_never_reported_as_partmm(self):
        record = self.human["paper_consistency"]["actorshq"]["smpl_partmm"]
        self.assertEqual(record["status"], "invalid_proxy_mismatch")
        self.assertEqual(record["measured_method"], "ssdr_reconstruction")
        self.assertLess(record["delta"]["psnr"], -1.7)

    def test_cross_method_failures_are_not_discarded(self):
        self.assertEqual(self.cross["total_runs"], 65)
        self.assertEqual(self.cross["status_counts"]["timeout"], 21)
        self.assertEqual(self.cross["status_counts"]["partial_ssdr_only"], 3)
        accounted = sum(self.cross["status_counts"].values())
        self.assertEqual(accounted, self.cross["total_runs"])

    def test_sync_is_deterministic_against_installed_sources(self):
        sources = {
            "dress": Path(
                "~/research/code/dressrecon/batch_evaluation_results.json"
            ).expanduser(),
            "bob": Path(
                "~/research/GA/Dynamic3DGaussians/server_output/"
                "bob_eval_v2/summary.json"
            ).expanduser(),
            "lbs_dna": Path(
                "~/research/GA/Dynamic3DGaussians/output/"
                "lbs_eval_v2/summary.json"
            ).expanduser(),
            "lbs_actors": Path(
                "~/research/GA/Dynamic3DGaussians/server_output/"
                "lbs_eval/summary.json"
            ).expanduser(),
            "ssdr": Path(
                "~/research/GA/Dynamic3DGaussians/server_output/"
                "ssdr_eval/summary.json"
            ).expanduser(),
            "cross": Path(
                "~/research/GA/Dynamic3DGaussians/output/"
                "cross_method_pointcloud_pipeline_full/"
                "cache_first_summary.csv"
            ).expanduser(),
            "splits": Path("~/research/code/Deform-2DGS/tempexp").expanduser(),
            "actors_params": Path(
                "~/research/GA/Dynamic3DGaussians/output/actorshq_exp_iso"
            ).expanduser(),
            "annotations": Path(
                "~/research/data/DNA-Rendering/"
                "data_used_in_4K4D/annotations"
            ).expanduser(),
            "actors_smplx": Path(
                "~/research/GA/Dynamic3DGaussians/actorshq_smplx"
            ).expanduser(),
            "d3dgs_data": Path(
                "~/research/GA/Dynamic3DGaussians/data"
            ).expanduser(),
        }
        if not all(path.exists() for path in sources.values()):
            self.skipTest("local reproduction sources are not installed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sync_dressrecon_original(
                sources["dress"], output / "dressrecon_original.json"
            )
            sync_human_novel_pose(
                sources["bob"],
                sources["lbs_dna"],
                sources["lbs_actors"],
                sources["ssdr"],
                output / "human_novel_pose.json",
            )
            sync_cross_method(
                sources["cross"],
                output / "cross_method_skeletonization.json",
            )
            sync_deform2dgs_splits(
                sources["splits"],
                output / "deform2dgs_observation_split.json",
            )
            sync_bob_availability(
                sources["bob"].parents[1] / "dna_exp_1x",
                sources["actors_params"],
                sources["annotations"],
                sources["actors_smplx"],
                sources["d3dgs_data"],
                output / "bob_recompute_availability.json",
            )
            for checked_in in RESULTS.glob("*.json"):
                regenerated = output / checked_in.name
                self.assertEqual(
                    checked_in.read_bytes(), regenerated.read_bytes(), checked_in
                )


if __name__ == "__main__":
    unittest.main()
