"""Run and evaluate the camera-ready PartMM patch-length ablation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_patch_size(row):
    value = row.get("patch_size")
    if value is None:
        value = row.get("config", {}).get("patch_size")
    return int(value) if value is not None else None


def run_patch_length(
    patch_size,
    output_dir,
    *,
    code_root=None,
    vto_root,
    smpl_model_path,
    smpl_fk_model_dir=None,
    ssdr_cache=None,
    expected_cases,
    device="cuda",
    n_train=5,
):
    """Run the bundled research matcher; generate its source fit locally."""
    from .vto_partmm import run_ablation

    if ssdr_cache and Path(ssdr_cache).suffix in {".npy", ".npz"}:
        raise ValueError("Use --fit-dir for a locally generated source-fit directory, not a legacy SSDR cache")
    return run_ablation(
        output_dir, patch_size=patch_size, vto_root=vto_root,
        smpl_model_path=smpl_model_path, expected_cases=expected_cases,
        n_train=n_train, device=device, fit_dir=ssdr_cache,
    )


def evaluate_patch_length(
    patch_size,
    source,
    output,
    *,
    expected_cases,
):
    """Validate per-case records and recompute the Table 5 aggregate."""

    source = Path(source).resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"patch-length source has no per-case rows: {source}")
    rows = [
        row
        for row in rows
        if row.get("variant", "default") == "default"
        and row.get("method", "partmm") == "partmm"
    ]
    actual_cases = [str(row["seq_name"]) for row in rows]
    expected_cases = list(expected_cases)
    if actual_cases != expected_cases:
        raise ValueError(
            "patch-length cases differ from the frozen protocol: "
            f"expected {expected_cases}, got {actual_cases}"
        )
    actual_patch_sizes = {_row_patch_size(row) for row in rows}
    if actual_patch_sizes != {int(patch_size)}:
        raise ValueError(
            f"expected patch size {patch_size}, got {sorted(actual_patch_sizes)}"
        )
    values = np.asarray([float(row["rmse_mm"]) for row in rows])
    frames = np.asarray([int(row["frames"]) for row in rows])
    if np.any(frames <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("patch-length rows contain invalid frames or RMSE")
    result = {
        "kind": "geometry_ablation",
        "table": "Table 5",
        "patch_size": int(patch_size),
        "protocol": {
            "training_sequences": 5,
            "evaluation_cases": actual_cases,
            "aggregation": "unweighted_case_mean",
        },
        "metrics": {
            "rmse_mm": float(values.mean()),
            "frame_weighted_rmse_mm": float(np.average(values, weights=frames)),
        },
        "per_sequence": [
            {
                "sequence": row["seq_name"],
                "frames": int(row["frames"]),
                "rmse_mm": float(row["rmse_mm"]),
            }
            for row in rows
        ],
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-cases", required=True)
    parser.add_argument("--run-if-missing", action="store_true")
    parser.add_argument("--runner-code-root")
    parser.add_argument("--vto-root")
    parser.add_argument("--smpl-model-path")
    parser.add_argument("--smpl-fk-model-dir")
    parser.add_argument("--fit-dir", "--ssdr-cache", dest="ssdr_cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source = Path(args.source)
    cases = [item for item in args.expected_cases.split(",") if item]
    if not source.is_file():
        if not args.run_if_missing:
            raise FileNotFoundError(source)
        required = {
            "vto_root": args.vto_root,
            "smpl_model_path": args.smpl_model_path,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ValueError(f"runner arguments are missing: {missing}")
        run_patch_length(
            args.patch_size,
            source.parent,
            code_root=args.runner_code_root,
            vto_root=args.vto_root,
            smpl_model_path=args.smpl_model_path,
            smpl_fk_model_dir=args.smpl_fk_model_dir,
            ssdr_cache=args.ssdr_cache,
            expected_cases=cases,
            device=args.device,
        )
    evaluate_patch_length(
        args.patch_size,
        source,
        args.output,
        expected_cases=cases,
    )
    print(args.output)


if __name__ == "__main__":
    main()
