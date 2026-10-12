"""Preflight the local assets needed to recompute the BoB experiment."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


DNA_CASES = (
    "dna_0012_11",
    "dna_0013_01",
    "dna_0013_03",
    "dna_0013_09",
    "dna_0013_11",
    "dna_0019_08",
    "dna_0021_03",
    "dna_0023_06",
)
ACTORSHQ_CASES = (
    "actorshq_actor01_seq1",
    "actorshq_actor01_seq2",
    "actorshq_actor02_seq1",
    "actorshq_actor02_seq2",
    "actorshq_actor04_seq2",
    "actorshq_actor05_seq1",
    "actorshq_actor06_seq1",
    "actorshq_actor06_seq2",
    "actorshq_actor07_seq1",
    "actorshq_actor08_seq1",
)
ALL_CASES = DNA_CASES + ACTORSHQ_CASES


def inspect_case(
    case: str,
    dna_results_dir: Path,
    actorshq_results_dir: Path,
    annotations_dir: Path,
    actorshq_smplx_dir: Path,
    data_dir: Path,
    include_paths: bool = False,
) -> Dict[str, Any]:
    if case not in ALL_CASES:
        raise ValueError(f"unknown BoB case: {case}")
    if case.startswith("dna_"):
        suffix = case[len("dna_") :]
        assets = {
            "params": dna_results_dir / f"{case}_1x" / "params.npz",
            "pose_annotations": annotations_dir / f"{suffix}_annots.smc",
            "evaluation_metadata": data_dir / f"{case}_1x" / "train_meta.json",
        }
        dataset = "dna_rendering"
    else:
        assets = {
            "params": actorshq_results_dir / case / "params.npz",
            "pose_annotations": actorshq_smplx_dir / case,
            "evaluation_metadata": data_dir / case / "train_meta.json",
        }
        dataset = "actorshq"
    availability = {
        name: path.is_file() if path.suffix else path.is_dir()
        for name, path in assets.items()
    }
    record: Dict[str, Any] = {
        "dataset": dataset,
        "assets": availability,
        "fit_ready": availability["params"] and availability["pose_annotations"],
        "evaluation_ready": availability["params"]
        and availability["evaluation_metadata"],
    }
    if include_paths:
        record["paths"] = {name: str(path) for name, path in assets.items()}
    return record


def audit(
    cases: Iterable[str],
    dna_results_dir: Path,
    actorshq_results_dir: Path,
    annotations_dir: Path,
    actorshq_smplx_dir: Path,
    data_dir: Path,
    include_paths: bool = False,
) -> Dict[str, Any]:
    records = {
        case: inspect_case(
            case,
            dna_results_dir,
            actorshq_results_dir,
            annotations_dir,
            actorshq_smplx_dir,
            data_dir,
            include_paths=include_paths,
        )
        for case in cases
    }
    by_dataset = {}
    for dataset in ("dna_rendering", "actorshq"):
        subset = [row for row in records.values() if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "cases": len(subset),
            "fit_ready": sum(row["fit_ready"] for row in subset),
            "evaluation_ready": sum(row["evaluation_ready"] for row in subset),
        }
    return {
        "schema_version": 1,
        "reproduction_id": "dressrecon_bob_recompute_availability",
        "cases": records,
        "by_dataset": by_dataset,
        "all_fit_ready": all(row["fit_ready"] for row in records.values()),
    }


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BoB recomputation assets")
    parser.add_argument("--case", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dna-results-dir", type=Path, required=True)
    parser.add_argument("--actorshq-results-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--actorshq-smplx-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-paths", action="store_true")
    parser.add_argument("--require-fit-ready", action="store_true")
    args = parser.parse_args()
    cases = ALL_CASES if args.all else tuple(args.case or ())
    if not cases:
        parser.error("select --all or at least one --case")
    result = audit(
        cases,
        args.dna_results_dir.expanduser(),
        args.actorshq_results_dir.expanduser(),
        args.annotations_dir.expanduser(),
        args.actorshq_smplx_dir.expanduser(),
        args.data_dir.expanduser(),
        include_paths=args.include_paths,
    )
    _write(args.output, result)
    print(json.dumps(result["by_dataset"], sort_keys=True))
    if args.require_fit_ready and not result["all_fit_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

