"""Audit and adopt the historical evidence for the ARAP/SSDR ablation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


ARAP_SSDR_PORTABLE = (
    "cfg_args",
    "inner_skeleton/inner_skeleton.npz",
    "inner_skeleton/inner_skeleton.ply",
    "inner_skeleton_refined/inner_skeleton.npz",
    "inner_skeleton_refined/inner_skeleton.ply",
    "mesh_skinning/mesh_with_skinning_colors.ply",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> Dict[str, Any]:
    return {
        "present": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def audit_arap_evidence(
    historical_riggs: Path,
    paper_pdf: Optional[Path] = None,
    raw_node_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    historical_riggs = Path(historical_riggs).resolve()
    stage1 = historical_riggs / "outputs/stage1_res/girlwalk_512_node"
    final = (
        historical_riggs
        / "saved_final_results_for_skeleton"
        / "girlwalk_512_final_node"
    )
    raw_manifest = Path(raw_node_manifest).resolve() if raw_node_manifest else None
    arap_checkpoint = stage1 / "deform/iteration_40000/deform.pth"
    arap_point_cloud = stage1 / "point_cloud/iteration_40000/point_cloud.ply"
    ssdr = final / "ssdr_results_bones50.npz"
    skeleton = final / "inner_skeleton_refined/inner_skeleton.npz"
    paper_pdf = Path(paper_pdf).resolve() if paper_pdf else None
    result = {
        "target_id": "arap_ssdr_ablation",
        "target": "Fig. 8 / ARAP and SSDR",
        "case": "DG-Mesh girlwalk",
        "paper_pdf": _artifact(paper_pdf) if paper_pdf else {"present": False},
        "exact_paper_source_renders_recovered": False,
        "variants": {
            "arap_no_ssdr": {
                "status": (
                    "recomputed_raw_node_weights"
                    if raw_manifest and raw_manifest.is_file()
                    else "historical_inputs_only"
                ),
                "deform_checkpoint": _artifact(arap_checkpoint),
                "point_cloud": _artifact(arap_point_cloud),
                "raw_node_manifest": (
                    {
                        **_artifact(raw_manifest),
                        "summary": json.loads(
                            raw_manifest.read_text(encoding="utf-8")
                        ),
                    }
                    if raw_manifest and raw_manifest.is_file()
                    else {"present": False}
                ),
            },
            "no_arap_ssdr": {
                "status": "missing_historical_checkpoint_requires_retraining",
            },
            "arap_ssdr": {
                "status": (
                    "historical_result_complete"
                    if ssdr.is_file() and skeleton.is_file()
                    else "missing_or_incomplete"
                ),
                "ssdr": _artifact(ssdr),
                "refined_skeleton": _artifact(skeleton),
            },
        },
    }
    result["all_ready"] = (
        all(
            variant["status"]
            in {"recomputed_raw_node_weights", "historical_result_complete"}
            for variant in result["variants"].values()
        )
        and result["exact_paper_source_renders_recovered"]
    )
    return result


def adopt_arap_evidence(
    evidence: Dict[str, Any], historical_riggs: Path, output_dir: Path
) -> Path:
    source = (
        Path(historical_riggs).resolve()
        / "saved_final_results_for_skeleton"
        / "girlwalk_512_final_node"
    )
    output_dir = Path(output_dir)
    for relative in ARAP_SSDR_PORTABLE:
        source_path = source / relative
        if source_path.is_file():
            destination = output_dir / "arap_ssdr" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "evidence.json"
    manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-riggs", default="~/research/GA/RigGS", type=Path
    )
    parser.add_argument(
        "--paper-pdf",
        default="~/research/code/GA_arxiv/PDFs/ARAP.pdf",
        type=Path,
    )
    parser.add_argument("--raw-node-manifest", type=Path)
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = audit_arap_evidence(
        args.historical_riggs.expanduser(),
        args.paper_pdf.expanduser(),
        args.raw_node_manifest,
    )
    if args.adopt_dir:
        output = adopt_arap_evidence(
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
