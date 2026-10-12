"""Adapter for the VTO neural and motion-matching experiment suite."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


FIELDS = {
    "mlp": "MLP_rmse",
    "gru": "GRU_rmse",
    "fullmm": "FullMM_rmse",
    "partmm": "PartMM_rmse",
}


def evaluate_vto_method(
    garment,
    method,
    output,
    *,
    summary,
    hood_root,
    python,
    n_train=5,
    exclude="tshirt_shape00_104_17",
    expected_cases=None,
):
    """Run the shared prediction suite when needed and select one method."""

    if method not in FIELDS:
        raise ValueError(f"unknown VTO suite method: {method}")
    summary = Path(summary)
    if not summary.is_file():
        summary.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            "precompute_hood_data.py",
            "--garment",
            garment,
            "--n_train",
            str(n_train),
            "--exclude",
            exclude,
            "--outdir",
            str(summary.parent),
        ]
        subprocess.run(command, cwd=str(hood_root), check=True)
    rows = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"VTO suite produced no per-sequence metrics: {summary}")
    actual_cases = [str(row["seq_name"]) for row in rows]
    if expected_cases is not None:
        expected_cases = list(expected_cases)
        if actual_cases != expected_cases:
            raise ValueError(
                "VTO evaluation cases differ from the frozen protocol: "
                f"expected {expected_cases}, got {actual_cases}"
            )
    field = FIELDS[method]
    values = [float(row[field]) for row in rows]
    frames = [int(row["frames"]) for row in rows]
    result = {
        "kind": "geometry",
        "garment": garment,
        "method": method,
        "source": str(summary.resolve()),
        "protocol": {
            "n_train": int(n_train),
            "excluded_sequence": exclude or None,
            "evaluation_cases": actual_cases,
        },
        "metrics": {
            "rmse_mm": float(np.mean(values)),
            "frame_weighted_rmse_mm": float(np.average(values, weights=frames)),
        },
        "per_sequence": [
            {
                "sequence": row["seq_name"],
                "frames": int(row["frames"]),
                "rmse_mm": float(row[field]),
            }
            for row in rows
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run/select a VTO animation method")
    parser.add_argument("--garment", choices=["tshirt", "dress"], required=True)
    parser.add_argument("--method", choices=sorted(FIELDS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--hood-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--n-train", type=int, default=5)
    parser.add_argument("--exclude", default="tshirt_shape00_104_17")
    parser.add_argument(
        "--expected-cases",
        help="Comma-separated frozen evaluation sequence list",
    )
    args = parser.parse_args()
    evaluate_vto_method(
        args.garment,
        args.method,
        args.output,
        summary=args.summary,
        hood_root=args.hood_root,
        python=args.python,
        n_train=args.n_train,
        exclude=args.exclude,
        expected_cases=(
            args.expected_cases.split(",") if args.expected_cases else None
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
