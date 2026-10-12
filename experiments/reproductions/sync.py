"""Normalize local reproduction results into small, portable JSON snapshots.

The source workspaces contain large datasets, checkpoints, renders, and absolute
paths. This module copies only numerical evidence and provenance hashes into the
public repository. Running it again is the audit path from a local result back
to the checked-in snapshot.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Optional

from .dressrecon_bob.availability import ALL_CASES, audit


REPOSITORIES = {
    "dressrecon": {
        "url": "https://github.com/jefftan969/dressrecon",
        "revision": "91b3b79",
    },
    "dynamic3dgaussians": {
        "url": "https://github.com/JonathonLuiten/Dynamic3DGaussians",
        "revision": "7dbbd4d",
    },
    "deform2dgs": {
        "url": "https://github.com/Net-Maker/Deform-2DGS",
        "revision": "16a7093",
    },
}
METHOD_REPOSITORIES = {
    "dgmesh": {
        "url": "https://github.com/Isabella98Liu/DG-Mesh",
        "revision": "754f42c",
    },
    "gaussiananimator": {
        "url": "https://github.com/Net-Maker/GaussianAnimator",
        "revision": "57b791a",
    },
    "deform2dgs": REPOSITORIES["deform2dgs"],
    "deformable3dgs": {
        "url": "https://github.com/ingra14m/Deformable-3D-Gaussians",
        "revision": "ac42372",
    },
    "dynamic2dgs": {
        "url": "https://github.com/hustvl/Dynamic-2DGS",
        "revision": "4355abb",
    },
    "scgs": {
        "url": "https://github.com/yihua7/SC-GS",
        "revision": "26cd57d",
    },
    "riggs_stage1": {
        "url": "https://github.com/yaoyx689/RigGS",
        "revision": "2ce695b",
    },
    "riggs_final_plyseq": {
        "url": "https://github.com/yaoyx689/RigGS",
        "revision": "2ce695b",
    },
}

IMAGE_METRICS = ("psnr", "ssim", "lpips", "fg_psnr")
CROSS_METRICS = (
    "joints",
    "inner_edges",
    "outer_bones",
    "rmse_x1000",
    "joint_accel_x1000",
    "outer_accel_x1000",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path, repository: str) -> Dict[str, Any]:
    return {
        "repository": REPOSITORIES[repository],
        "result_file": path.name,
        "sha256": _sha256(path),
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _numeric(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _averages(
    per_sequence: Mapping[str, Mapping[str, Any]],
    methods: Iterable[str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for method in methods:
        rows = [
            record["methods"][method]
            for record in per_sequence.values()
            if method in record["methods"]
        ]
        metrics = {}
        for metric in IMAGE_METRICS:
            values = [row.get(metric) for row in rows if row.get(metric) is not None]
            if values:
                metrics[metric] = mean(values)
        result[method] = {"sequence_count": len(rows), "metrics": metrics}
    return result


def sync_dressrecon_original(source: Path, output: Path) -> None:
    records = _read_json(source)
    per_sequence = {}
    for record in records:
        per_sequence[record["seqname"].replace("-", "_")] = {
            "source_sequence": record["seqname"],
            "num_test_frames": record["num_test_frames"],
            "num_evaluated": record["num_evaluated"],
            "methods": {
                "dressrecon": record["baseline"],
                "dressrecon_motion_matching": record["motion_matching"],
            },
        }
    snapshot = {
        "schema_version": 1,
        "reproduction_id": "dressrecon_original_dna",
        "description": "DressRecon baseline and local motion-matching extension.",
        "evidence": _source(source, "dressrecon"),
        "per_sequence": per_sequence,
        "averages": _averages(
            per_sequence, ("dressrecon", "dressrecon_motion_matching")
        ),
        "total_evaluated_frames": sum(
            record["num_evaluated"] for record in records
        ),
    }
    _write(output, snapshot)


def _dataset(sequence: str) -> str:
    if sequence.startswith("dna_"):
        return "dna_rendering"
    if sequence.startswith("actorshq_"):
        return "actorshq"
    raise ValueError(f"unrecognized human sequence: {sequence}")


def sync_human_novel_pose(
    bob_source: Path,
    lbs_dna_source: Path,
    lbs_actorshq_source: Path,
    ssdr_source: Path,
    output: Path,
) -> None:
    bob = _read_json(bob_source)
    lbs_dna = _read_json(lbs_dna_source)
    lbs_actors = _read_json(lbs_actorshq_source)
    ssdr = _read_json(ssdr_source)
    sequences = sorted(bob["per_sequence"])
    per_sequence: Dict[str, Any] = {}
    for sequence in sequences:
        bob_record = bob["per_sequence"][sequence]
        lbs_records = lbs_dna if sequence.startswith("dna_") else lbs_actors
        lbs_record = lbs_records["per_sequence"][sequence]
        ssdr_record = ssdr["per_sequence"][sequence]
        per_sequence[sequence] = {
            "dataset": _dataset(sequence),
            "num_test_frames": bob_record.get("num_test_frames"),
            "num_eval_cams": bob_record.get("num_eval_cams"),
            "methods": {
                "d3dgs_gt": bob_record["oracle"],
                "smpl_bob": bob_record["bob"],
                "smpl_lbs": {
                    **lbs_record["lbs"],
                    "vertex_rmse_mm": lbs_record.get("lbs_vertex_rmse_mm"),
                },
                "ssdr_reconstruction": {
                    **ssdr_record["ssdr"],
                    "vertex_rmse_mm": ssdr_record.get("ssdr_vertex_rmse_mm"),
                },
            },
        }

    by_dataset = {}
    for dataset in ("dna_rendering", "actorshq"):
        subset = {
            key: value
            for key, value in per_sequence.items()
            if value["dataset"] == dataset
        }
        by_dataset[dataset] = {
            "sequence_count": len(subset),
            "methods": _averages(
                subset, ("d3dgs_gt", "smpl_lbs", "smpl_bob", "ssdr_reconstruction")
            ),
        }

    paper_values = {
        "dna_rendering": {
            "smpl_lbs": {"psnr": 24.10, "ssim": 0.948, "lpips": 0.049},
            "smpl_bob": {"psnr": 23.03, "ssim": 0.935, "lpips": 0.057},
            "smpl_partmm": {"psnr": 28.28, "ssim": 0.964, "lpips": 0.040},
        },
        "actorshq": {
            "smpl_lbs": {"psnr": 17.54, "ssim": 0.738, "lpips": 0.315},
            "smpl_bob": {"psnr": 16.66, "ssim": 0.722, "lpips": 0.259},
            "smpl_partmm": {"psnr": 24.26, "ssim": 0.850, "lpips": 0.182},
        },
    }
    consistency = {}
    for dataset, methods in paper_values.items():
        consistency[dataset] = {}
        for method, reference in methods.items():
            measured_method = (
                "ssdr_reconstruction" if method == "smpl_partmm" else method
            )
            measured = by_dataset[dataset]["methods"][measured_method]["metrics"]
            delta = {
                metric: measured[metric] - value
                for metric, value in reference.items()
            }
            consistent = (
                abs(delta["psnr"]) <= 0.01
                and abs(delta["ssim"]) <= 0.001
                and abs(delta["lpips"]) <= 0.001
            )
            consistency[dataset][method] = {
                "status": (
                    "invalid_proxy_matches_number"
                    if method == "smpl_partmm" and consistent
                    else "invalid_proxy_mismatch"
                    if method == "smpl_partmm"
                    else "matches_rounded_paper"
                    if consistent
                    else "mismatch"
                ),
                "measured_method": measured_method,
                "paper": reference,
                "delta": delta,
            }

    snapshot = {
        "schema_version": 1,
        "reproduction_id": "human_novel_pose_local_reproductions",
        "description": (
            "D3DGS oracle, robust LBS, local DressRecon BoB, and an SSDR "
            "reconstruction diagnostic evaluated on the paper's human split."
        ),
        "evidence": {
            "bob_and_oracle": _source(bob_source, "dynamic3dgaussians"),
            "lbs_dna": _source(lbs_dna_source, "dynamic3dgaussians"),
            "lbs_actorshq": _source(
                lbs_actorshq_source, "dynamic3dgaussians"
            ),
            "ssdr_reconstruction": _source(ssdr_source, "dynamic3dgaussians"),
        },
        "per_sequence": per_sequence,
        "by_dataset": by_dataset,
        "paper_consistency": consistency,
        "notes": [
            "DNA robust-LBS evidence uses the corrected lbs_eval_v2 run.",
            "ActorsHQ robust-LBS evidence uses the server lbs_eval run.",
            (
                "The source previously labelled PartMM is eval_ssdr_metrics.py: "
                "it reconstructs an SSDR cache fitted on all frames and performs "
                "no PartMM retrieval. It is retained only as an invalid proxy."
            ),
        ],
    }
    _write(output, snapshot)


def sync_cross_method(source: Path, output: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        record: Dict[str, Any] = {
            "dataset": row["dataset"],
            "scene": row["scene"],
            "method": row["label"],
            "display_name": row["display_name"],
            "run_id": row["seq"],
            "status": row["status"],
            "input_source": row["source"],
            "timed_out": row["timed_out"].lower() == "true",
        }
        for field in CROSS_METRICS:
            value = _numeric(row[field])
            if value is not None:
                record[field] = int(value) if field in {
                    "joints", "inner_edges", "outer_bones"
                } else value
        returncode = _numeric(row["returncode"])
        if returncode is not None:
            record["returncode"] = int(returncode)
        elapsed = _numeric(row["elapsed_sec"])
        if elapsed is not None:
            record["elapsed_seconds"] = elapsed
        normalized.append(record)

    by_method = {}
    for method in sorted({row["method"] for row in normalized}):
        method_rows = [row for row in normalized if row["method"] == method]
        successful = [
            row for row in method_rows if row["status"].startswith("ok")
        ]
        metrics = {}
        for field in CROSS_METRICS:
            values = [row[field] for row in successful if field in row]
            if values:
                metrics[field] = mean(values)
        by_method[method] = {
            "display_name": method_rows[0]["display_name"],
            "total_runs": len(method_rows),
            "successful_runs": len(successful),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in method_rows).items())
            ),
            "metrics_on_successful_runs": metrics,
        }

    snapshot = {
        "schema_version": 1,
        "reproduction_id": "cross_method_skeletonization",
        "description": (
            "Skelebone extraction from point trajectories reproduced by "
            "multiple dynamic reconstruction papers."
        ),
        "evidence": _source(source, "dynamic3dgaussians"),
        "source_repositories": METHOD_REPOSITORIES,
        "total_runs": len(normalized),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in normalized).items())
        ),
        "per_run": normalized,
        "by_method": by_method,
        "aggregation_policy": (
            "Means use only status=ok or ok_precomputed; timeouts and partial "
            "runs remain explicit in per_run and status_counts."
        ),
    }
    _write(output, snapshot)


def sync_deform2dgs_splits(source_dir: Path, output: Path) -> None:
    runs = {}
    evidence = []
    for path in sorted(source_dir.glob("*/split_experiments_results.json")):
        values = _read_json(path)
        folder = path.parent.name
        if folder.startswith("exp-lbs-"):
            variant = "lbs"
            scene = folder[len("exp-lbs-") :]
        elif folder.startswith("exp-split-"):
            variant = "split"
            scene = folder[len("exp-split-") :]
        else:
            continue
        run_id = f"{variant}_{scene}"
        complete = all(isinstance(value, (int, float)) for value in values.values())
        runs[run_id] = {
            "variant": variant,
            "scene": scene,
            "status": "complete" if complete else "incomplete",
            "psnr_by_observation_fraction": values,
        }
        evidence.append(
            {
                "run_id": run_id,
                "result_file": f"{folder}/{path.name}",
                "sha256": _sha256(path),
            }
        )

    by_variant = {}
    for variant in ("lbs", "split"):
        complete = [
            row
            for row in runs.values()
            if row["variant"] == variant and row["status"] == "complete"
        ]
        metrics = {}
        for fraction in (f"{percent}%" for percent in range(10, 101, 10)):
            values = [
                row["psnr_by_observation_fraction"][fraction] for row in complete
            ]
            if values:
                metrics[f"psnr_{fraction[:-1]}"] = mean(values)
        by_variant[variant] = {
            "total_runs": sum(
                row["variant"] == variant for row in runs.values()
            ),
            "complete_runs": len(complete),
            "incomplete_runs": sum(
                row["variant"] == variant and row["status"] != "complete"
                for row in runs.values()
            ),
            "metrics_on_complete_runs": metrics,
        }

    snapshot = {
        "schema_version": 1,
        "reproduction_id": "deform2dgs_observation_split",
        "description": (
            "Deform-2DGS PSNR as the observed training-frame fraction grows "
            "from 10% to 100%."
        ),
        "repository": REPOSITORIES["deform2dgs"],
        "evidence": evidence,
        "runs": runs,
        "by_variant": by_variant,
    }
    _write(output, snapshot)


def sync_bob_availability(
    dna_results_dir: Path,
    actorshq_results_dir: Path,
    annotations_dir: Path,
    actorshq_smplx_dir: Path,
    data_dir: Path,
    output: Path,
) -> None:
    snapshot = audit(
        ALL_CASES,
        dna_results_dir,
        actorshq_results_dir,
        annotations_dir,
        actorshq_smplx_dir,
        data_dir,
        include_paths=False,
    )
    snapshot["scope"] = (
        "Current raw assets needed for recomputation; independent of the "
        "complete checked-in result evidence."
    )
    _write(output, snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local reproduction metrics into portable snapshots"
    )
    parser.add_argument(
        "--dressrecon-original",
        type=Path,
        default=Path("~/research/code/dressrecon/batch_evaluation_results.json"),
    )
    parser.add_argument(
        "--bob",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/server_output/"
            "bob_eval_v2/summary.json"
        ),
    )
    parser.add_argument(
        "--lbs-dna",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/output/lbs_eval_v2/summary.json"
        ),
    )
    parser.add_argument(
        "--lbs-actorshq",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/server_output/lbs_eval/summary.json"
        ),
    )
    parser.add_argument(
        "--ssdr-diagnostic",
        dest="ssdr_diagnostic",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/server_output/ssdr_eval/summary.json"
        ),
    )
    parser.add_argument(
        "--cross-method",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/output/"
            "cross_method_pointcloud_pipeline_full/cache_first_summary.csv"
        ),
    )
    parser.add_argument(
        "--deform2dgs-splits",
        type=Path,
        default=Path("~/research/code/Deform-2DGS/tempexp"),
    )
    parser.add_argument(
        "--actorshq-params",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/output/actorshq_exp_iso"
        ),
    )
    parser.add_argument(
        "--dna-annotations",
        type=Path,
        default=Path(
            "~/research/data/DNA-Rendering/"
            "data_used_in_4K4D/annotations"
        ),
    )
    parser.add_argument(
        "--actorshq-smplx",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/actorshq_smplx"
        ),
    )
    parser.add_argument(
        "--d3dgs-data",
        type=Path,
        default=Path("~/research/GA/Dynamic3DGaussians/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()
    values = vars(args)
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = value.expanduser().resolve()
    output_dir = args.output_dir
    sync_dressrecon_original(
        args.dressrecon_original,
        output_dir / "dressrecon_original.json",
    )
    sync_human_novel_pose(
        args.bob,
        args.lbs_dna,
        args.lbs_actorshq,
        args.ssdr_diagnostic,
        output_dir / "human_novel_pose.json",
    )
    sync_cross_method(
        args.cross_method,
        output_dir / "cross_method_skeletonization.json",
    )
    sync_deform2dgs_splits(
        args.deform2dgs_splits,
        output_dir / "deform2dgs_observation_split.json",
    )
    sync_bob_availability(
        args.bob.parents[1] / "dna_exp_1x",
        args.actorshq_params,
        args.dna_annotations,
        args.actorshq_smplx,
        args.d3dgs_data,
        output_dir / "bob_recompute_availability.json",
    )
    print(f"wrote five reproduction snapshots to {output_dir}")


if __name__ == "__main__":
    main()
