"""Validate and package the recovered multiview table summaries."""

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np


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
EXPECTED_SHA256 = {
    "bob": "6c6c1d64003b1f43ad96f1c56def7316395bb7181de20370bb1f5262e7b0f785",
    "lbs_dna": "dd66dc57a18eb3c821ab827be49805bf3c5b696ed4875686ee27966982874d44",
    "lbs_actorshq": "34b4dad6802f9e68f22fc394fed10049aa5631a3676dacc0444a06340d1a5522",
    "ssdr_proxy": "a2d56efc6cda6a3b0cee47ed2d4d68a42248a020134a38e07cf0b48ba1cef16f",
}
PAPER = {
    "dna_rendering": {
        "d3dgs_gt": {"psnr": 32.16, "ssim": 0.978, "lpips": 0.033},
        "smpl_bob": {"psnr": 23.03, "ssim": 0.935, "lpips": 0.057},
        "smpl_lbs": {"psnr": 24.10, "ssim": 0.948, "lpips": 0.049},
        "smpl_partmm": {"psnr": 28.28, "ssim": 0.964, "lpips": 0.039},
    },
    "actorshq": {
        "d3dgs_gt": {"psnr": 30.56, "ssim": 0.938, "lpips": 0.136},
        "smpl_bob": {"psnr": 16.66, "ssim": 0.722, "lpips": 0.259},
        "smpl_lbs": {"psnr": 17.54, "ssim": 0.738, "lpips": 0.315},
        "smpl_partmm": {"psnr": 24.26, "ssim": 0.850, "lpips": 0.182},
    },
}


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen(path: Path, role: str):
    path = Path(path).resolve()
    digest = _sha256(path)
    if digest != EXPECTED_SHA256[role]:
        raise ValueError(
            f"{role} hash differs: expected {EXPECTED_SHA256[role]}, got {digest}"
        )
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _mean(rows, cases, method):
    return {
        metric: float(np.mean([rows[case][method][metric] for case in cases]))
        for metric in ("psnr", "ssim", "lpips")
    }


def _paper_check(measured, paper):
    tolerance = {"psnr": 0.01, "ssim": 0.001, "lpips": 0.001}
    return {
        "measured": measured,
        "paper": paper,
        "within_printed_precision": all(
            abs(measured[key] - paper[key]) <= tolerance[key] for key in tolerance
        ),
    }


def _actorshq_declared_subset_audit(bob_rows, lbs_rows):
    digits = {"psnr": 2, "ssim": 3, "lpips": 3}
    matching = []
    for cases in itertools.combinations(ACTORSHQ_CASES, 8):
        measured = {
            "d3dgs_gt": _mean(bob_rows, cases, "oracle"),
            "smpl_bob": _mean(bob_rows, cases, "bob"),
            "smpl_lbs": _mean(lbs_rows, cases, "lbs"),
        }
        if all(
            round(measured[method][metric], digits[metric])
            == PAPER["actorshq"][method][metric]
            for method in measured
            for metric in digits
        ):
            matching.append(list(cases))
    return {
        "paper_declared_subjects": 8,
        "recovered_sequences": len(ACTORSHQ_CASES),
        "tested_eight_sequence_subsets": len(
            list(itertools.combinations(ACTORSHQ_CASES, 8))
        ),
        "subsets_matching_all_three_printed_rows": matching,
        "ten_sequence_aggregate_matches": all(
            _paper_check(
                _mean(
                    bob_rows if method != "smpl_lbs" else lbs_rows,
                    ACTORSHQ_CASES,
                    {
                        "d3dgs_gt": "oracle",
                        "smpl_bob": "bob",
                        "smpl_lbs": "lbs",
                    }[method],
                ),
                PAPER["actorshq"][method],
            )["within_printed_precision"]
            for method in ("d3dgs_gt", "smpl_bob", "smpl_lbs")
        ),
    }


def audit_historical_evidence(paths, bob_model_roots=()):
    records = {}
    files = {}
    for role, path in paths.items():
        records[role], files[role] = _load_frozen(path, role)

    bob_rows = records["bob"]["per_sequence"]
    if tuple(bob_rows) != ACTORSHQ_CASES + DNA_CASES:
        raise ValueError("BoB summary cases differ from the paper protocol")
    lbs_dna_rows = records["lbs_dna"]["per_sequence"]
    if tuple(lbs_dna_rows) != DNA_CASES:
        raise ValueError("DNA LBS summary cases differ from the paper protocol")
    lbs_actor_rows = records["lbs_actorshq"]["per_sequence"]
    if not set(ACTORSHQ_CASES).issubset(lbs_actor_rows):
        raise ValueError("ActorsHQ LBS summary is missing paper cases")

    recovered_models = sorted(
        {
            path.parent.name
            for root in bob_model_roots
            for path in Path(root).expanduser().glob("*/bob_model.pt")
            if path.parent.name in set(DNA_CASES + ACTORSHQ_CASES)
        }
    )
    datasets = {}
    for dataset, cases, stored_average, lbs_rows in (
        ("dna_rendering", DNA_CASES, records["bob"]["dna_average"], lbs_dna_rows),
        (
            "actorshq",
            ACTORSHQ_CASES,
            records["bob"]["actorshq_average"],
            lbs_actor_rows,
        ),
    ):
        oracle = _mean(bob_rows, cases, "oracle")
        bob = _mean(bob_rows, cases, "bob")
        lbs = _mean(lbs_rows, cases, "lbs")
        if any(
            abs(bob[key] - float(stored_average["bob"][key])) > 1e-12
            for key in bob
        ):
            raise ValueError(f"{dataset} BoB stored average is inconsistent")
        datasets[dataset] = {
            "cases": list(cases),
            "d3dgs_gt": _paper_check(oracle, PAPER[dataset]["d3dgs_gt"]),
            "smpl_bob": _paper_check(bob, PAPER[dataset]["smpl_bob"]),
            "smpl_lbs": _paper_check(lbs, PAPER[dataset]["smpl_lbs"]),
        }
    return {
        "target_id": "human_novel_pose",
        "target": "Table 4 / unseen-pose clothed-human rendering",
        "artifact_level": "aggregate_metrics_only",
        "datasets": datasets,
        "files": files,
        "bob": {
            "paper_rows_recovered": all(
                item["smpl_bob"]["within_printed_precision"]
                for item in datasets.values()
            ),
            "raw_models_recovered": len(recovered_models),
            "expected_models": len(DNA_CASES) + len(ACTORSHQ_CASES),
            "model_cases": recovered_models,
            "fully_recomputable": len(recovered_models)
            == len(DNA_CASES) + len(ACTORSHQ_CASES),
        },
        "actorshq_case_count_audit": _actorshq_declared_subset_audit(
            bob_rows,
            lbs_actor_rows,
        ),
        "partmm": {
            "paper_rows_recovered": False,
            "historical_ssdr_summary_classification": (
                "invalid_all_frame_ssdr_proxy_not_partmm"
            ),
        },
        "all_paper_methods_recomputable": False,
    }


def adopt_historical_evidence(evidence, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, record in evidence["files"].items():
        shutil.copy2(record["path"], output_dir / f"{role}_summary.json")
    bob_dir = Path(evidence["files"]["bob"]["path"]).parent
    per_case = output_dir / "bob_per_case"
    per_case.mkdir(exist_ok=True)
    for path in bob_dir.glob("*.json"):
        if path.name != "summary.json":
            shutil.copy2(path, per_case / path.name)
    output = output_dir / "evidence.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bob",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/server_output/bob_eval_v2/summary.json"
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
        "--ssdr-proxy",
        type=Path,
        default=Path(
            "~/research/GA/Dynamic3DGaussians/server_output/ssdr_eval/summary.json"
        ),
    )
    parser.add_argument("--bob-model-root", type=Path, action="append", default=[])
    parser.add_argument("--adopt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {
        "bob": args.bob.expanduser(),
        "lbs_dna": args.lbs_dna.expanduser(),
        "lbs_actorshq": args.lbs_actorshq.expanduser(),
        "ssdr_proxy": args.ssdr_proxy.expanduser(),
    }
    evidence = audit_historical_evidence(paths, args.bob_model_root)
    if args.adopt_dir:
        output = adopt_historical_evidence(evidence, args.adopt_dir)
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
