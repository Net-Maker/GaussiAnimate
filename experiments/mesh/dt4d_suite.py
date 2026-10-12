"""Adapter for the DeformingThings4D animation benchmark."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


FIELDS = {
    "robust_lbs": "lbs",
    "mlp": "mlp_local",
    "gru": "gru_local",
    "fullmm": "mm_full_local",
    "partmm": "mm_part_local",
}


def evaluate_dt4d_method(
    method,
    output,
    *,
    experiment_root,
    python,
    summary,
    target_frames=200,
    training_steps=10000,
    expected_instances=None,
    expected_summary_sha256=None,
):
    """Prepare/train the shared suite when needed and select one method."""

    if method not in FIELDS:
        raise ValueError(f"unknown DT4D method: {method}")
    experiment_root = Path(experiment_root)
    summary = Path(summary)
    if not summary.is_file():
        subprocess.run(
            [
                str(python),
                "run_all.py",
                "--target_frames",
                str(target_frames),
                "--steps",
                str(training_steps),
            ],
            cwd=str(experiment_root),
            check=True,
        )
        subprocess.run(
            [str(python), "benchmark_all.py", "--animals", "all", "--save"],
            cwd=str(experiment_root),
            check=True,
        )
    summary_bytes = summary.read_bytes()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    if (
        expected_summary_sha256 is not None
        and summary_sha256 != expected_summary_sha256
    ):
        raise ValueError(
            "DT4D benchmark summary differs from the frozen protocol: "
            f"expected {expected_summary_sha256}, got {summary_sha256}"
        )
    data = json.loads(summary_bytes)
    rows = data.get("results", [])
    if expected_instances is not None and len(rows) != expected_instances:
        raise ValueError(
            "DT4D benchmark instance count differs from the frozen protocol: "
            f"expected {expected_instances}, got {len(rows)}"
        )
    field = FIELDS[method]
    values = [float(row[field]) * 1000.0 for row in rows if field in row]
    if not values:
        raise ValueError(f"no {field} results found in {summary}")
    result = {
        "kind": "geometry",
        "dataset": "DeformingThings4D",
        "method": method,
        "source": str(summary.resolve()),
        "protocol": {
            "target_frames": int(target_frames),
            "training_steps": int(training_steps),
            "summary_sha256": summary_sha256,
            "instance_ids": [row["animal"] for row in rows],
            "split_strategy": (
                "motion-prefix leave-one-out with similarity and temporal fallbacks"
            ),
        },
        "metrics": {
            "rmse_mm": float(np.mean(values)),
            "evaluated_instances": len(values),
        },
        "per_instance": [
            {
                "animal": row["animal"],
                "train_frames": int(row["n_train"]),
                "test_frames": int(row["n_test"]),
                "split": row["split_desc"],
                "rmse_mm": float(row[field]) * 1000.0,
            }
            for row in rows
            if field in row
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run/select a DT4D animation method")
    parser.add_argument("--method", choices=sorted(FIELDS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--target-frames", type=int, default=200)
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--expected-instances", type=int)
    parser.add_argument("--expected-summary-sha256")
    args = parser.parse_args()
    evaluate_dt4d_method(
        args.method,
        args.output,
        experiment_root=args.experiment_root,
        python=args.python,
        summary=args.summary,
        target_frames=args.target_frames,
        training_steps=args.training_steps,
        expected_instances=args.expected_instances,
        expected_summary_sha256=args.expected_summary_sha256,
    )
    print(args.output)


if __name__ == "__main__":
    main()
