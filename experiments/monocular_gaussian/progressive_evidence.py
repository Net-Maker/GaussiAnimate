"""Audit and adopt the historical evidence for Fig. 9 progressive skeletonization."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


PAPER_COLUMNS = {0: 1, 30: 60, 60: 120, 80: 160, 100: 200}
PORTABLE_ARTIFACTS = (
    "ablation_summary.json",
    "inner_skeleton/inner_skeleton.npz",
    "inner_skeleton/inner_skeleton.ply",
    "inner_skeleton_refined/inner_skeleton.npz",
    "inner_skeleton_refined/inner_skeleton.ply",
    "skeleton_vis/kinematic_tree.npz",
    "skeleton_vis/skeleton_frame0.ply",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_manifest(frame_dir: Path) -> Dict[str, Any]:
    frames = sorted(frame_dir.glob("frame_*.ply"))
    digest = hashlib.sha256()
    for path in frames:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
    return {
        "count": len(frames),
        "first": frames[0].name if frames else None,
        "last": frames[-1].name if frames else None,
        "name_size_sha256": digest.hexdigest(),
    }


def _column_record(results_root: Path, percentage: int, frames: int) -> Dict[str, Any]:
    if percentage == 0:
        return {
            "percentage": 0,
            "expected_frames": frames,
            "status": "missing_exact_historical_artifact",
            "protocol_match": False,
            "has_ssdr": False,
            "has_skeleton": False,
            "has_refined_skeleton": False,
            "artifacts": {},
        }

    directory = results_root / f"{percentage:03d}pct"
    summary_path = directory / "ablation_summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    protocol_match = bool(
        summary
        and summary.get("percentage") == percentage
        and summary.get("num_frames_used") == frames
        and summary.get("total_frames") == 200
        and summary.get("num_points") == 30000
        and summary.get("num_bones_requested") == 50
    )
    ssdr = directory / "ssdr_results_bones50.npz"
    skeleton = directory / "inner_skeleton/inner_skeleton.npz"
    refined = directory / "inner_skeleton_refined/inner_skeleton.npz"
    artifacts = {}
    for relative in PORTABLE_ARTIFACTS:
        path = directory / relative
        if path.is_file():
            artifacts[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    if ssdr.is_file():
        artifacts["ssdr_results_bones50.npz"] = {
            "bytes": ssdr.stat().st_size,
            "sha256": _sha256(ssdr),
            "adopted": False,
        }
    has_skeleton = skeleton.is_file()
    if protocol_match and has_skeleton:
        status = "recovered_exact_protocol_complete"
    elif protocol_match and ssdr.is_file():
        status = "recovered_exact_protocol_ssdr_only"
    else:
        status = "missing_or_protocol_mismatch"
    return {
        "percentage": percentage,
        "expected_frames": frames,
        "status": status,
        "protocol_match": protocol_match,
        "has_ssdr": ssdr.is_file(),
        "has_skeleton": has_skeleton,
        "has_refined_skeleton": refined.is_file(),
        "summary": summary,
        "artifacts": artifacts,
    }


def audit_progressive_evidence(
    historical_riggs: Path,
    paper_pdf: Optional[Path] = None,
    implementation: Optional[Path] = None,
) -> Dict[str, Any]:
    historical_riggs = Path(historical_riggs).resolve()
    case_root = (
        historical_riggs
        / "saved_final_results_for_skeleton"
        / "trex_512_final_node"
    )
    results_root = (
        historical_riggs
        / "saved_final_results_for_skeleton"
        / "ablation_frames"
        / "trex_512_final_node"
    )
    columns = [
        _column_record(results_root, percentage, frames)
        for percentage, frames in PAPER_COLUMNS.items()
    ]
    input_mesh = case_root / "mesh/frame_0000_post.ply"
    evidence: Dict[str, Any] = {
        "target_id": "progressive_skeletonization",
        "target": "Fig. 9 / progressive skeletonization",
        "case": "DG-Mesh T-Rex",
        "paper_columns": list(PAPER_COLUMNS),
        "historical_riggs": str(historical_riggs),
        "input_frames": _frame_manifest(case_root / "deformed_pointclouds"),
        "input_mesh": {
            "present": input_mesh.is_file(),
            "sha256": _sha256(input_mesh) if input_mesh.is_file() else None,
        },
        "columns": columns,
        "complete_columns": sum(
            column["status"] == "recovered_exact_protocol_complete"
            for column in columns
        ),
        "all_ready": all(
            column["status"] == "recovered_exact_protocol_complete"
            for column in columns
        ),
    }
    for key, path in (("paper_pdf", paper_pdf), ("historical_implementation", implementation)):
        resolved = Path(path).resolve() if path else None
        evidence[key] = {
            "path": str(resolved) if resolved else None,
            "present": bool(resolved and resolved.is_file()),
            "sha256": _sha256(resolved) if resolved and resolved.is_file() else None,
        }
    return evidence


def adopt_progressive_evidence(
    evidence: Mapping[str, Any], historical_riggs: Path, output_dir: Path
) -> Path:
    historical_riggs = Path(historical_riggs).resolve()
    source_root = (
        historical_riggs
        / "saved_final_results_for_skeleton"
        / "ablation_frames"
        / "trex_512_final_node"
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for column in evidence["columns"]:
        percentage = int(column["percentage"])
        if percentage == 0:
            continue
        for relative in PORTABLE_ARTIFACTS:
            source = source_root / f"{percentage:03d}pct" / relative
            if source.is_file():
                destination = output_dir / f"{percentage:03d}pct" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    manifest = output_dir / "evidence.json"
    manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-riggs",
        default="~/research/GA/RigGS",
        type=Path,
    )
    parser.add_argument(
        "--paper-pdf",
        default="~/research/code/GA_arxiv/PDFs/progressive.pdf",
        type=Path,
    )
    parser.add_argument(
        "--implementation",
        default="~/research/GA/RigGS/ablation_skinning_skeleton.py",
        type=Path,
    )
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = audit_progressive_evidence(
        args.historical_riggs.expanduser(),
        args.paper_pdf.expanduser(),
        args.implementation.expanduser(),
    )
    if args.adopt_dir:
        output = adopt_progressive_evidence(
            evidence, args.historical_riggs.expanduser(), args.adopt_dir
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
