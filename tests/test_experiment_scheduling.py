import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import paper
from experiments.scheduling import (
    GpuScheduler,
    configure_visible_gpu,
    method_requires_gpu,
    stage_requires_gpu,
)


class GpuSchedulerTest(unittest.TestCase):
    def test_child_gpu_argument_preserves_parent_assignment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configure_visible_gpu(3), "3")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "7"},
            clear=True,
        ):
            self.assertEqual(configure_visible_gpu(0), "7")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

    def test_round_robin_assignment_preserves_cli_behavior(self):
        scheduler = GpuScheduler(["2", "5"])
        self.assertEqual(
            [scheduler.gpu_for(index) for index in range(5)],
            ["2", "5", "2", "5", "2"],
        )
        self.assertIsNone(GpuScheduler([]).gpu_for(0))

    def test_different_gpus_and_cpu_jobs_can_overlap(self):
        for scheduler, indices in (
            (GpuScheduler(["0", "1"]), (0, 1)),
            (GpuScheduler([]), (0, 1)),
        ):
            barrier = threading.Barrier(2)

            def occupy(index):
                with scheduler.slot(index):
                    barrier.wait(timeout=1.0)

            with self.subTest(gpus=[scheduler.gpu_for(i) for i in indices]):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(occupy, index) for index in indices]
                    for future in futures:
                        future.result(timeout=2.0)

    def test_gpu_stage_detection_covers_explicit_cuda_and_lpips(self):
        self.assertTrue(
            stage_requires_gpu(
                {"type": "command", "command": ["fit", "--device", "cuda"]}
            )
        )
        self.assertTrue(
            stage_requires_gpu(
                {"type": "command", "command": ["evaluate", "--gpu", "0"]}
            )
        )
        self.assertTrue(
            stage_requires_gpu(
                {"type": "image_metrics", "metrics": ["psnr", "lpips"]}
            )
        )
        self.assertFalse(
            stage_requires_gpu(
                {
                    "type": "image_metrics",
                    "metrics": ["psnr", "lpips"],
                    "device": "cpu",
                }
            )
        )
        method = {
            "stages": [
                {"id": "prepare", "type": "command", "command": ["prepare"]},
                {
                    "id": "evaluate",
                    "type": "command",
                    "command": ["evaluate", "--gpu", "0"],
                },
            ]
        }
        self.assertFalse(method_requires_gpu(method, ["prepare"]))
        self.assertTrue(method_requires_gpu(method, ["evaluate"]))

    def test_parallel_gpu_jobs_without_gpu_list_are_rejected(self):
        method = {
            "id": "gpu_method",
            "stages": [
                {
                    "id": "fit",
                    "type": "command",
                    "command": ["fit", "--device", "cuda"],
                }
            ],
        }
        jobs = [
            SimpleNamespace(
                key=f"dataset/case{index}/gpu_method",
                output_root=Path("/tmp"),
                method=method,
            )
            for index in range(2)
        ]
        argv = [
            "gaussianimate-paper",
            "experiment.yaml",
            "--workers",
            "2",
        ]
        config = {
            "experiment": {"id": "test"},
            "datasets": [{"id": "dataset"}],
            "methods": [method],
        }
        with patch.object(sys, "argv", argv), \
                patch.object(paper, "load_paper_config", return_value=config), \
                patch.object(paper, "expand_jobs", return_value=jobs), \
                patch.object(paper, "run_job") as run_job, \
                patch.object(paper, "aggregate_run"):
            with self.assertRaisesRegex(
                ValueError,
                "parallel GPU stages require --gpus",
            ):
                paper.main()
        run_job.assert_not_called()

    def test_paper_cli_serializes_jobs_assigned_to_one_gpu(self):
        method = {
            "id": "method",
            "stages": [
                {
                    "id": "fit",
                    "type": "command",
                    "command": ["fit", "--device", "cuda"],
                }
            ],
        }
        jobs = [
            SimpleNamespace(
                key=f"dataset/case/method{index}",
                output_root=Path("/tmp"),
                method=method,
            )
            for index in range(4)
        ]
        first_entered = threading.Event()
        release_first = threading.Event()
        later_entered = threading.Event()
        errors = []
        calls = []
        call_lock = threading.Lock()

        def fake_run_job(job, selected_stages, force, gpu):
            with call_lock:
                call_index = len(calls)
                calls.append((job.key, gpu))
            if call_index == 0:
                first_entered.set()
                if not release_first.wait(timeout=2.0):
                    raise TimeoutError("test did not release first GPU job")
            else:
                later_entered.set()
            return {}

        def invoke():
            try:
                paper.main()
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        argv = [
            "gaussianimate-paper",
            "experiment.yaml",
            "--workers",
            "4",
            "--gpus",
            "0",
        ]
        config = {
            "experiment": {"id": "test"},
            "datasets": [{"id": "dataset"}],
            "methods": [method],
        }
        with patch.object(sys, "argv", argv), \
                patch.object(paper, "load_paper_config", return_value=config), \
                patch.object(paper, "expand_jobs", return_value=jobs), \
                patch.object(paper, "run_job", side_effect=fake_run_job), \
                patch.object(paper, "aggregate_run"):
            thread = threading.Thread(target=invoke)
            thread.start()
            self.assertTrue(first_entered.wait(timeout=1.0))
            self.assertFalse(later_entered.wait(timeout=0.1))
            release_first.set()
            thread.join(timeout=3.0)

        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(len(calls), len(jobs))
        self.assertEqual({gpu for _, gpu in calls}, {"0"})

    def test_cpu_job_does_not_wait_for_gpu_slot(self):
        gpu_method = {
            "id": "gpu",
            "stages": [
                {
                    "id": "fit",
                    "type": "command",
                    "command": ["fit", "--device", "cuda"],
                }
            ],
        }
        cpu_method = {
            "id": "cpu",
            "stages": [
                {
                    "id": "prepare",
                    "type": "command",
                    "command": ["prepare"],
                }
            ],
        }
        jobs = [
            SimpleNamespace(
                key="dataset/gpu0/gpu",
                output_root=Path("/tmp"),
                method=gpu_method,
            ),
            SimpleNamespace(
                key="dataset/gpu1/gpu",
                output_root=Path("/tmp"),
                method=gpu_method,
            ),
            SimpleNamespace(
                key="dataset/cpu/cpu",
                output_root=Path("/tmp"),
                method=cpu_method,
            ),
        ]
        first_gpu_entered = threading.Event()
        release_first_gpu = threading.Event()
        second_gpu_entered = threading.Event()
        cpu_entered = threading.Event()

        def fake_run_job(job, selected_stages, force, gpu):
            if job.method["id"] == "cpu":
                self.assertIsNone(gpu)
                cpu_entered.set()
            elif not first_gpu_entered.is_set():
                first_gpu_entered.set()
                release_first_gpu.wait(timeout=2.0)
            else:
                second_gpu_entered.set()
            return {}

        argv = [
            "gaussianimate-paper",
            "experiment.yaml",
            "--workers",
            "3",
            "--gpus",
            "0",
        ]
        config = {
            "experiment": {"id": "test"},
            "datasets": [{"id": "dataset"}],
            "methods": [gpu_method, cpu_method],
        }
        errors = []

        def invoke():
            try:
                paper.main()
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        with patch.object(sys, "argv", argv), \
                patch.object(paper, "load_paper_config", return_value=config), \
                patch.object(paper, "expand_jobs", return_value=jobs), \
                patch.object(paper, "run_job", side_effect=fake_run_job), \
                patch.object(paper, "aggregate_run"):
            thread = threading.Thread(target=invoke)
            thread.start()
            self.assertTrue(first_gpu_entered.wait(timeout=1.0))
            self.assertTrue(cpu_entered.wait(timeout=1.0))
            self.assertFalse(second_gpu_entered.wait(timeout=0.1))
            release_first_gpu.set()
            thread.join(timeout=3.0)

        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertTrue(second_gpu_entered.is_set())


if __name__ == "__main__":
    unittest.main()
