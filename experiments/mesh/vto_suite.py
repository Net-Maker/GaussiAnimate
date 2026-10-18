"""Adapter for the VTO neural and motion-matching experiment suite."""

import argparse
import json
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
    hood_root=None,
    python=None,
    vto_root=None,
    smpl_model_path=None,
    device="cuda",
    steps=3000,
    seed=42,
    n_train=5,
    exclude="tshirt_shape00_104_17",
    expected_cases=None,
):
    """Run the shared prediction suite when needed and select one method."""

    if method not in FIELDS:
        raise ValueError(f"unknown VTO suite method: {method}")
    summary = Path(summary)
    if not summary.is_file():
        if not vto_root or not smpl_model_path or not expected_cases:
            raise ValueError("Fresh VTO runs require --vto-root, --smpl-model-path, and --expected-cases")
        from .vto_training import run_suite
        rows = run_suite(summary.parent, garment=garment, vto_root=vto_root,
                         smpl_model_path=smpl_model_path, expected_cases=expected_cases,
                         n_train=n_train, device=device, steps=steps, seed=seed, exclude=exclude)
        summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    rows = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"VTO suite produced no per-sequence metrics: {summary}")
    if vto_root is not None and any(
        ("training_steps" in row and int(row["training_steps"]) != steps)
        or ("seed" in row and int(row["seed"]) != seed)
        for row in rows
    ):
        raise ValueError("Existing VTO summary uses different training settings; choose a new summary path")
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
    parser.add_argument("--hood-root", help=argparse.SUPPRESS)
    parser.add_argument("--python", help=argparse.SUPPRESS)
    parser.add_argument("--vto-root")
    parser.add_argument("--smpl-model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
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
        vto_root=args.vto_root,
        smpl_model_path=args.smpl_model_path,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        n_train=args.n_train,
        exclude=args.exclude,
        expected_cases=(
            args.expected_cases.split(",") if args.expected_cases else None
        ),
    )
    print(args.output)


if __name__ == "__main__":
    main()
