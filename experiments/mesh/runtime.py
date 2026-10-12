"""Summarize the executable PartMM runtime benchmark."""

import argparse
import json
import subprocess
from pathlib import Path


EXPECTED_RUNTIME_CONFIG = {
    "garment": "tshirt",
    "n_train": 5,
    "n_train_frames": 840,
    "n_test_frames": 203,
    "target_seq": "tshirt_shape00_91_33",
    "num_bones": 20,
}


def load_runtime_detail(path, target_sequence="tshirt_shape00_91_33"):
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    config = record.get("config", {})
    expected = {**EXPECTED_RUNTIME_CONFIG, "target_seq": target_sequence}
    differences = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if differences:
        raise ValueError(f"runtime detail differs from the paper protocol: {differences}")

    timing = record.get("timing", {})
    required = {
        "build_part_binding_s",
        "build_part_databases_s",
        "smpl_fk_s",
        "retarget_motion_s",
        "reconstruct_s",
        "ssdr_decomposition_s",
        "offline_total_s",
        "online_total_s",
        "end_to_end_s",
        "online_per_frame_ms",
    }
    missing = required - set(timing)
    if missing:
        raise KeyError(f"runtime detail is missing {sorted(missing)}")

    offline = (
        float(timing["build_part_binding_s"])
        + float(timing["build_part_databases_s"])
        + float(timing["ssdr_decomposition_s"])
    )
    online = (
        float(timing["smpl_fk_s"])
        + float(timing["retarget_motion_s"])
        + float(timing["reconstruct_s"])
    )
    checks = {
        "offline_total_s": (offline, float(timing["offline_total_s"])),
        "online_total_s": (online, float(timing["online_total_s"])),
        "end_to_end_s": (offline + online, float(timing["end_to_end_s"])),
        "online_per_frame_ms": (
            online * 1000.0 / int(config["n_test_frames"]),
            float(timing["online_per_frame_ms"]),
        ),
    }
    inconsistent = {
        key: {"recomputed": recomputed, "stored": stored}
        for key, (recomputed, stored) in checks.items()
        if abs(recomputed - stored) > 1e-9
    }
    if inconsistent:
        raise ValueError(f"runtime detail has inconsistent totals: {inconsistent}")
    return {
        "config": config,
        "metrics": {
            "offline_build_s": float(timing["offline_total_s"]),
            "online_total_s": float(timing["online_total_s"]),
            "online_per_frame_ms": float(timing["online_per_frame_ms"]),
            "core_end_to_end_s": float(timing["end_to_end_s"]),
        },
        "breakdown": {key: float(value) for key, value in timing.items()},
        "source": str(path),
    }


def _load_later_remeasurement(per_case_path, target_sequence):
    rows = json.loads(Path(per_case_path).read_text(encoding="utf-8"))
    default = next(
        row
        for row in rows
        if row["variant"] == "default" and row["seq_name"] == target_sequence
    )
    frames = int(default["frames"])
    online = float(default["predict_s"])
    offline = float(default["build_s"])
    return {
        "config": {
            "database_frames": 840,
            "query_frames": frames,
            "outer_bones": 20,
            "target_sequence": target_sequence,
        },
        "metrics": {
            "offline_build_s": offline,
            "online_total_s": online,
            "online_per_frame_ms": online * 1000.0 / frames,
            "core_end_to_end_s": offline + online,
            "rmse_mm": float(default["rmse_mm"]),
        },
        "source": str(Path(per_case_path).resolve()),
    }


def benchmark_runtime(
    output,
    *,
    code_root,
    python,
    result_dir,
    vto_root,
    smpl_model_path,
    smpl_fk_model_dir,
    ssdr_cache,
    target_sequence="tshirt_shape00_91_33",
    runtime_detail=None,
):
    result_dir = Path(result_dir)
    mean_path = result_dir / "mean_summary.json"
    per_case_path = result_dir / "per_case_summary.json"
    if not per_case_path.is_file():
        subprocess.run(
            [
                str(python),
                "run_partmm_tshirt_ablation.py",
                "--target-seq",
                target_sequence,
                "--garment",
                "tshirt",
                "--n-train",
                "5",
                "--vto-data-root",
                str(vto_root),
                "--smpl-model-path",
                str(smpl_model_path),
                "--smpl-fk-model-dir",
                str(smpl_fk_model_dir),
                "--ssdr-cache",
                str(ssdr_cache),
                "--variants",
                "default",
                "--output-dir",
                str(result_dir),
                "--device",
                "cuda",
                "--no-save-vertices",
            ],
            cwd=str(code_root),
            check=True,
        )
    later = _load_later_remeasurement(per_case_path, target_sequence)
    result = {
        "kind": "runtime",
        "measurement_role": "later_remeasurement",
        **later,
    }
    if runtime_detail is not None:
        paper = load_runtime_detail(runtime_detail, target_sequence)
        result = {
            "kind": "runtime",
            "measurement_role": "paper_protocol_measurement",
            "config": {
                "database_frames": int(paper["config"]["n_train_frames"]),
                "query_frames": int(paper["config"]["n_test_frames"]),
                "outer_bones": int(paper["config"]["num_bones"]),
                "target_sequence": paper["config"]["target_seq"],
            },
            "metrics": paper["metrics"],
            "breakdown": paper["breakdown"],
            "source": paper["source"],
            "later_remeasurement": later,
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark VTO PartMM runtime")
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--vto-root", required=True)
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--smpl-fk-model-dir", required=True)
    parser.add_argument("--ssdr-cache", required=True)
    parser.add_argument("--target-sequence", default="tshirt_shape00_91_33")
    parser.add_argument("--runtime-detail")
    args = parser.parse_args()
    benchmark_runtime(
        args.output,
        code_root=args.code_root,
        python=args.python,
        result_dir=args.result_dir,
        vto_root=args.vto_root,
        smpl_model_path=args.smpl_model_path,
        smpl_fk_model_dir=args.smpl_fk_model_dir,
        ssdr_cache=args.ssdr_cache,
        target_sequence=args.target_sequence,
        runtime_detail=args.runtime_detail,
    )
    print(args.output)


if __name__ == "__main__":
    main()
