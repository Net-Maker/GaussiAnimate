"""CLI and compatibility facade for the modular experiment engine."""

import os

# Respect explicit user settings while keeping NumPy/PyTorch extension imports
# stable on high-core-count experiment hosts.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

import argparse
import concurrent.futures
import json
from contextlib import nullcontext
from typing import Sequence

from .engine import (
    Job,
    adopt_legacy_stages,
    expand_jobs,
    load_paper_config,
    render_template,
    run_job,
    stage_fingerprint,
    validate_compiled_config,
    validate_paper_config,
)
from .report import aggregate_run
from .scheduling import GpuScheduler, method_requires_gpu


__all__ = [
    "Job",
    "adopt_legacy_stages",
    "expand_jobs",
    "load_paper_config",
    "render_template",
    "run_job",
    "stage_fingerprint",
    "validate_compiled_config",
    "validate_paper_config",
]


def _print_jobs(jobs: Sequence[Job]) -> None:
    for job in jobs:
        stages = ",".join(
            stage["id"] for stage in job.method["stages"]
        )
        print(
            f"{job.key} mode={job.method.get('mode', 'native')} "
            f"stages={stages}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a GaussiAnimate paper experiment"
    )
    parser.add_argument("config")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--case", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--stage", action="append")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--adopt-legacy",
        action="store_true",
        help=(
            "fingerprint selected legacy stage records only when a current "
            "completed downstream stage proves their outputs were consumed"
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_paper_config(
        args.config,
        require_environment=False,
    )
    jobs = expand_jobs(
        config,
        args.dataset,
        args.case,
        args.method,
    )
    if not jobs:
        raise ValueError("filters selected no experiment jobs")
    if args.validate_only:
        print(
            f"valid: {config['experiment']['id']} "
            f"datasets={len(config['datasets'])} "
            f"methods={len(config['methods'])} "
            f"jobs={len(jobs)}"
        )
        return
    if args.list:
        _print_jobs(jobs)
        return
    if args.dry_run:
        for job in jobs:
            print(
                json.dumps(
                    run_job(job, args.stage, dry_run=True),
                    indent=2,
                )
            )
        return
    if args.adopt_legacy:
        if not args.stage:
            raise ValueError("--adopt-legacy requires at least one --stage")
        if args.force:
            raise ValueError("--adopt-legacy cannot be combined with --force")
        for job in jobs:
            manifest = adopt_legacy_stages(job, args.stage)
            print(f"{manifest['status']}: {job.key}")
        aggregate_run(
            jobs[0].output_root,
            expected_jobs=expand_jobs(config),
        )
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")

    gpus = [
        item.strip()
        for item in args.gpus.split(",")
        if item.strip()
    ]
    if args.workers > 1 and len(jobs) > 1 and not gpus:
        gpu_jobs = [
            job.key
            for job in jobs
            if method_requires_gpu(job.method, args.stage)
        ]
        if gpu_jobs:
            preview = ", ".join(gpu_jobs[:3])
            if len(gpu_jobs) > 3:
                preview += f", ... ({len(gpu_jobs)} jobs)"
            raise ValueError(
                "parallel GPU stages require --gpus; "
                f"selected GPU jobs: {preview}"
            )
    gpu_scheduler = GpuScheduler(gpus)
    requires_gpu = [
        method_requires_gpu(job.method, args.stage)
        for job in jobs
    ]
    gpu_ordinals = {}
    next_gpu_ordinal = 0
    for index, required in enumerate(requires_gpu):
        if required:
            gpu_ordinals[index] = next_gpu_ordinal
            next_gpu_ordinal += 1
    failures = []

    def execute(index_job):
        index, job = index_job
        slot = (
            gpu_scheduler.slot(gpu_ordinals[index])
            if requires_gpu[index]
            else nullcontext(None)
        )
        with slot as gpu:
            return job, run_job(
                job,
                args.stage,
                args.force,
                gpu=gpu,
            )

    if args.workers == 1:
        for index_job in enumerate(jobs):
            try:
                job, _ = execute(index_job)
                print(f"complete: {job.key}")
            except Exception as exc:
                job = index_job[1]
                failures.append((job.key, repr(exc)))
                print(f"failed: {job.key}: {exc}")
                if args.fail_fast:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(execute, item): item[1]
                for item in enumerate(jobs)
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                    print(f"complete: {job.key}")
                except Exception as exc:
                    failures.append((job.key, repr(exc)))
                    print(f"failed: {job.key}: {exc}")
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        break

    aggregate_run(
        jobs[0].output_root,
        expected_jobs=expand_jobs(config),
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} experiment jobs failed; "
            "see per-job manifests"
        )


if __name__ == "__main__":
    main()
