"""Validate and package the recovered PartMM runtime evidence."""

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

from experiments.mesh.runtime import load_runtime_detail


EXPECTED_SHA256 = {
    "paper_runtime": "44185ee48dbd309624c08f0d7eb16817b595c54bacc8688aef8b64ccd378dbd0",
    "legacy_runtime": "a97619d4cd41710bf6e411f02498271aacd4ee1e2a07313129ea59b35f688436",
    "ssdr_timing": "c3d67328d75a83845dfd60c188859b6a526d79278f9f115b59beed8b611675a3",
    "per_case_summary": "1dc43d7b7a4ab209acfbe46ef51b696f554a707880589533c70493828c3f3b3a",
    "mean_summary": "697e6ba0094da543c7381cef419f1162f5139719fb76b81ec08117d4256214e3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_file(path: Path, role: str):
    path = Path(path).resolve()
    digest = _sha256(path)
    expected = EXPECTED_SHA256[role]
    if digest != expected:
        raise ValueError(f"{role} hash differs: expected {expected}, got {digest}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def audit_runtime_evidence(detail_path: Path, result_dir: Path):
    detail_path = Path(detail_path).resolve()
    detail_dir = detail_path.parent
    result_dir = Path(result_dir).resolve()
    files = {
        "paper_runtime": _frozen_file(detail_path, "paper_runtime"),
        "legacy_runtime": _frozen_file(
            detail_dir / "partmm_runtime_detail_tshirt_shape00_91_33.json",
            "legacy_runtime",
        ),
        "ssdr_timing": _frozen_file(
            detail_dir / "ssdr_timing.json", "ssdr_timing"
        ),
        "per_case_summary": _frozen_file(
            result_dir / "per_case_summary.json", "per_case_summary"
        ),
        "mean_summary": _frozen_file(
            result_dir / "mean_summary.json", "mean_summary"
        ),
    }
    paper = load_runtime_detail(detail_path)
    rows = json.loads(
        (result_dir / "per_case_summary.json").read_text(encoding="utf-8")
    )
    later = next(
        row
        for row in rows
        if row["variant"] == "default"
        and row["seq_name"] == "tshirt_shape00_91_33"
    )
    later_ms = float(later["predict_s"]) * 1000.0 / int(later["frames"])
    paper_ms = paper["metrics"]["online_per_frame_ms"]
    return {
        "target_id": "partmm_runtime",
        "target": "Fig. R.1 / PartMM runtime",
        "protocol": {
            "database_frames": 840,
            "query_frames": 203,
            "outer_bones": 20,
            "target_sequence": "tshirt_shape00_91_33",
        },
        "paper_protocol_measurement": {
            "online_per_frame_ms": paper_ms,
            "offline_build_s": paper["metrics"]["offline_build_s"],
            "online_total_s": paper["metrics"]["online_total_s"],
            "end_to_end_s": paper["metrics"]["core_end_to_end_s"],
            "matches_paper_one_decimal": round(paper_ms, 1) == 13.5,
        },
        "later_remeasurement": {
            "online_per_frame_ms": later_ms,
            "offline_build_s": float(later["build_s"]),
            "online_total_s": float(later["predict_s"]),
            "rmse_mm": float(later["rmse_mm"]),
            "note": "Kept as a separate later run; it does not replace the paper measurement.",
        },
        "files": files,
        "current_audit_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "note": "This is the audit environment, not a recovered manifest of the original timing run.",
        },
        "exact_paper_measurement_recovered": round(paper_ms, 1) == 13.5,
        "fully_reproducible": False,
        "blockers": [
            "The hardware and software manifest captured at the time of the original timing run is absent.",
            "The recovered experiment source tree is uncommitted and therefore has no immutable source revision.",
        ],
    }


def adopt_runtime_evidence(evidence, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, record in evidence["files"].items():
        source = Path(record["path"])
        shutil.copy2(source, output_dir / f"{role}{source.suffix}")
    output = output_dir / "evidence.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detail",
        type=Path,
        default=(
            Path("~/research/GA/hood/rebuttal_partmm_tshirt_shape00_91_33")
            / "data/runtime_partmm_detail"
            / "partmm_runtime_v2_tshirt_shape00_91_33.json"
        ),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=(
            Path("~/research/GA/hood/rebuttal_partmm_tshirt_shape00_91_33")
            / "results/ablation_tshirt_5cases_n5"
        ),
    )
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = audit_runtime_evidence(
        args.detail.expanduser(), args.result_dir.expanduser()
    )
    if args.adopt_dir:
        output = adopt_runtime_evidence(evidence, args.adopt_dir)
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
