"""Validate and package the recovered per-instance D4D table evidence."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = "2d6017baba956f3bcc191453409fa8bafdfa238fea7529082fbc8da040daafa5"
FIELDS = {
    "robust_lbs": "lbs",
    "mlp": "mlp_local",
    "gru": "gru_local",
    "fullmm": "mm_full_local",
    "partmm": "mm_part_local",
}
PAPER_RMSE_MM = {
    "robust_lbs": 65.3,
    "mlp": 68.8,
    "gru": 58.4,
    "fullmm": 53.3,
    "partmm": 54.6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dt4d_evidence(summary: Path):
    summary = Path(summary).resolve()
    digest = _sha256(summary)
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"D4D summary hash differs: expected {EXPECTED_SHA256}, got {digest}"
        )
    rows = json.loads(summary.read_text(encoding="utf-8")).get("results", [])
    if len(rows) != 26:
        raise ValueError(f"expected 26 D4D instances, found {len(rows)}")
    means = {}
    for method, field in FIELDS.items():
        values = np.asarray([row[field] for row in rows], dtype=np.float64) * 1000.0
        if not np.isfinite(values).all():
            raise ValueError(f"{field} contains a non-finite result")
        means[method] = float(values.mean())
    return {
        "target_id": "d4d_animal_animation",
        "target": "Table 3 / D4D animal animation",
        "metric": (
            "mean per-frame per-vertex Euclidean distance in millimeters "
            "(paper label: RMSE)"
        ),
        "protocol_classification": {
            "split": (
                "historical motion-prefix leave-one-out proxy with "
                "similarity and temporal fallbacks"
            ),
            "robust_lbs": (
                "inverse-distance-to-bone weighting, not the "
                "paper-stated robust BBW transfer"
            ),
            "partmm": (
                "per-frame nearest-neighbor retrieval on local rotations, "
                "not the paper-stated multiscale PartMM protocol"
            ),
        },
        "evidence_level": "aggregate_only_no_per_frame_prediction_geometry",
        "summary_sha256": digest,
        "instances": len(rows),
        "instance_ids": [row["animal"] for row in rows],
        "split_strategies": sorted({row["split_desc"] for row in rows}),
        "mean_rmse_mm": means,
        "paper_consistency": {
            method: {
                "paper_rmse_mm": PAPER_RMSE_MM[method],
                "measured_rmse_mm": value,
                "matches_paper_one_decimal": round(value, 1)
                == PAPER_RMSE_MM[method],
            }
            for method, value in means.items()
        },
        "all_paper_rows_reproduced": all(
            round(value, 1) == PAPER_RMSE_MM[method]
            for method, value in means.items()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=(
            "~/research/data/DeformingThings4D/skeleton2ssdr_experiment/"
            "benchmark_results/benchmark_results.json"
        ),
        type=Path,
    )
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = args.summary.expanduser()
    evidence = audit_dt4d_evidence(summary)
    if args.adopt_dir:
        args.adopt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary, args.adopt_dir / "benchmark_results.json")
        output = args.adopt_dir / "evidence.json"
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output
    else:
        print(json.dumps(evidence, indent=2))
        return
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
