"""Validate and package the historical VTO Table 3 result arrays."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


EXPECTED_CASES = {
    "tshirt": (
        "tshirt_shape00_91_33",
        "tshirt_shape00_91_36",
        "tshirt_shape00_91_38",
        "tshirt_shape00_91_61",
        "tshirt_shape00_91_62",
    ),
    "dress": (
        "dress_shape00_91_38",
        "dress_shape00_91_61",
        "dress_shape00_91_62",
    ),
}
METHOD_FIELDS = {
    "mlp": ("mlp_vertices", "mlp_rmse"),
    "gru": ("gru_vertices", "gru_rmse"),
    "fullmm": ("fullmm_vertices", "fullmm_rmse"),
    "partmm": ("partmm_vertices", "partmm_rmse"),
}
PAPER_RMSE_MM = {
    "tshirt": {
        "robust_lbs": 33.7,
        "mlp": 23.9,
        "gru": 22.9,
        "fullmm": 38.2,
        "partmm": 17.4,
    },
    "dress": {
        "robust_lbs": 52.2,
        "mlp": 36.8,
        "gru": 37.5,
        "fullmm": 37.0,
        "partmm": 36.7,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_record(path: Path, garment: str, expected_case: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "gt_vertices",
            "faces",
            "seq_name",
            "garment",
            *(item for pair in METHOD_FIELDS.values() for item in pair),
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        gt = data["gt_vertices"]
        if str(data["seq_name"].item()) != expected_case:
            raise ValueError(f"{path} has the wrong sequence identity")
        if str(data["garment"].item()) != garment:
            raise ValueError(f"{path} has the wrong garment identity")
        methods = {}
        for method, (vertices_key, scalar_key) in METHOD_FIELDS.items():
            prediction = data[vertices_key]
            if prediction.shape != gt.shape:
                raise ValueError(f"{path}:{vertices_key} shape differs from ground truth")
            recomputed = float(
                np.linalg.norm(prediction.astype(np.float64) - gt, axis=-1).mean()
                * 1000.0
            )
            stored = float(data[scalar_key])
            methods[method] = {
                "stored_rmse_mm": stored,
                "recomputed_rmse_mm": recomputed,
                "consistent": abs(stored - recomputed) < 1e-4,
            }
    return {
        "case": expected_case,
        "frames": int(gt.shape[0]),
        "vertices": int(gt.shape[1]),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "methods": methods,
        "ready": all(method["consistent"] for method in methods.values()),
    }


def _robust_lbs_record(
    root: Path,
    garment: str,
    expected_cases,
    ground_truth_root: Path,
) -> Dict[str, Any]:
    root = Path(root).resolve()
    metrics_path = root.parent / "geometry_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = {row["sequence"]: row for row in metrics["per_sequence"]}
    if list(rows) != list(expected_cases):
        raise ValueError(f"{garment} Robust-LBS cases differ from the frozen protocol")

    weights_path = root / "transferred_weights.npz"
    with np.load(weights_path, allow_pickle=False) as weights:
        cloth_weights = weights["cloth_weights"]
        matched = weights["matched"]
        valid_weights = (
            cloth_weights.ndim == 2
            and cloth_weights.shape[1] == 24
            and matched.shape == (cloth_weights.shape[0],)
            and np.isfinite(cloth_weights).all()
            and (cloth_weights >= -1e-6).all()
            and np.allclose(cloth_weights.sum(axis=1), 1.0, atol=1e-5)
            and str(weights["garment"].item()) == garment
        )

    cases = []
    for case in expected_cases:
        prediction_path = root / f"{case}.npz"
        gt_path = Path(ground_truth_root) / f"{case}.npz"
        with np.load(
            prediction_path,
            allow_pickle=False,
        ) as prediction_data:
            with np.load(gt_path, allow_pickle=False) as gt_data:
                prediction = prediction_data["prediction"]
                gt = gt_data["gt_vertices"]
                if str(prediction_data["sequence"].item()) != case:
                    raise ValueError(
                        f"{prediction_path} has the wrong sequence identity"
                    )
                if str(prediction_data["garment"].item()) != garment:
                    raise ValueError(
                        f"{prediction_path} has the wrong garment identity"
                    )
                if prediction.shape != gt.shape:
                    raise ValueError(
                        f"{prediction_path} shape differs from ground truth"
                    )
                recomputed = float(
                    np.linalg.norm(
                        prediction.astype(np.float64)
                        - gt.astype(np.float64),
                        axis=-1,
                    ).mean()
                    * 1000.0
                )
        stored = float(rows[case]["mean_error_mm"])
        cases.append(
            {
                "case": case,
                "frames": int(prediction.shape[0]),
                "prediction_sha256": _sha256(prediction_path),
                "recomputed_rmse_mm": recomputed,
                "stored_rmse_mm": stored,
                "consistent": abs(recomputed - stored) < 1e-4,
            }
        )
    measured = float(np.mean([case["recomputed_rmse_mm"] for case in cases]))
    return {
        "source_root": str(root),
        "metrics_sha256": _sha256(metrics_path),
        "weights_sha256": _sha256(weights_path),
        "matched_vertices": int(matched.sum()),
        "vertices": int(len(matched)),
        "cases": cases,
        "mean_rmse_mm": measured,
        "paper_consistency": {
            "paper_rmse_mm": PAPER_RMSE_MM[garment]["robust_lbs"],
            "measured_rmse_mm": measured,
            "matches_paper_one_decimal": round(measured, 1)
            == PAPER_RMSE_MM[garment]["robust_lbs"],
        },
        "ready": valid_weights and all(case["consistent"] for case in cases),
    }


def audit_vto_evidence(
    roots: Mapping[str, Path],
    robust_lbs_roots: Mapping[str, Path] = None,
) -> Dict[str, Any]:
    garments = {}
    for garment, expected_cases in EXPECTED_CASES.items():
        root = Path(roots[garment]).resolve()
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if [row["seq_name"] for row in summary] != list(expected_cases):
            raise ValueError(f"{garment} summary differs from the frozen case protocol")
        records = [
            _case_record(root / f"{case}.npz", garment, case)
            for case in expected_cases
        ]
        means = {
            method: float(
                np.mean(
                    [
                        record["methods"][method]["recomputed_rmse_mm"]
                        for record in records
                    ]
                )
            )
            for method in METHOD_FIELDS
        }
        paper_consistency = {
            method: {
                "paper_rmse_mm": PAPER_RMSE_MM[garment][method],
                "measured_rmse_mm": value,
                "matches_paper_one_decimal": round(value, 1)
                == PAPER_RMSE_MM[garment][method],
            }
            for method, value in means.items()
        }
        robust_lbs = None
        if robust_lbs_roots is not None:
            robust_lbs = _robust_lbs_record(
                robust_lbs_roots[garment],
                garment,
                expected_cases,
                root,
            )
            paper_consistency["robust_lbs"] = robust_lbs["paper_consistency"]
        garments[garment] = {
            "source_root": str(root),
            "summary_sha256": _sha256(summary_path),
            "cases": records,
            "mean_rmse_mm": means,
            "paper_consistency": paper_consistency,
            "robust_lbs": robust_lbs,
            "ready": all(record["ready"] for record in records),
        }
    return {
        "target_id": "vto_garment_animation",
        "target": "Table 3 / VTO garment animation",
        "metric": "mean per-frame per-vertex Euclidean distance in millimeters",
        "garments": garments,
        "all_arrays_valid": all(
            item["ready"]
            and (item["robust_lbs"] is None or item["robust_lbs"]["ready"])
            for item in garments.values()
        ),
        "all_paper_rows_reproduced": all(
            comparison["matches_paper_one_decimal"]
            for item in garments.values()
            for comparison in item["paper_consistency"].values()
        ),
    }


def adopt_vto_evidence(
    evidence: Mapping[str, Any],
    roots: Mapping[str, Path],
    output_dir: Path,
    robust_lbs_roots: Mapping[str, Path] = None,
) -> Path:
    output_dir = Path(output_dir)
    for garment, expected_cases in EXPECTED_CASES.items():
        source = Path(roots[garment]).resolve()
        destination = output_dir / garment
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "summary.json", destination / "summary.json")
        for case in expected_cases:
            shutil.copy2(source / f"{case}.npz", destination / f"{case}.npz")
        if robust_lbs_roots is not None:
            robust_source = Path(robust_lbs_roots[garment]).resolve()
            robust_destination = destination / "robust_lbs"
            robust_destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                robust_source.parent / "geometry_metrics.json",
                robust_destination / "geometry_metrics.json",
            )
            for path in robust_source.glob("*.npz"):
                shutil.copy2(path, robust_destination / path.name)
    manifest = output_dir / "evidence.json"
    manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tshirt-root",
        default="~/research/GA/hood/output/precomputed_tshirt",
        type=Path,
    )
    parser.add_argument(
        "--dress-root",
        default="~/research/GA/hood/output/precomputed_dress",
        type=Path,
    )
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--robust-tshirt-root", type=Path)
    parser.add_argument("--robust-dress-root", type=Path)
    args = parser.parse_args()
    roots = {
        "tshirt": args.tshirt_root.expanduser(),
        "dress": args.dress_root.expanduser(),
    }
    robust_roots = None
    if args.robust_tshirt_root or args.robust_dress_root:
        if not (args.robust_tshirt_root and args.robust_dress_root):
            parser.error("both Robust-LBS roots are required")
        robust_roots = {
            "tshirt": args.robust_tshirt_root.expanduser(),
            "dress": args.robust_dress_root.expanduser(),
        }
    evidence = audit_vto_evidence(roots, robust_roots)
    if args.adopt_dir:
        output = adopt_vto_evidence(
            evidence, roots, args.adopt_dir, robust_roots
        )
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        output = args.output
    else:
        print(json.dumps(evidence, indent=2))
        return
    print(output)


if __name__ == "__main__":
    main()
