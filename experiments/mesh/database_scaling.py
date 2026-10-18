"""Validate PartMM database-scale accuracy, coverage, and profiling records."""

import argparse
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np

from .patch_length import run_patch_length


REQUIRED_PROFILE_FIELDS = {
    "sequences",
    "min_so3",
    "build_s",
    "predict_ms",
    "gpu_mb",
}


def _axis_angle_quaternion(values):
    values = np.asarray(values, dtype=np.float64)
    angles = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.empty_like(angles)
    small = angles < 1e-12
    scale[small] = 0.5
    scale[~small] = np.sin(angles[~small] * 0.5) / angles[~small]
    xyz = values * scale
    return np.concatenate([np.cos(angles * 0.5), xyz], axis=-1)


def minimum_pose_coverage(database_poses, query_poses, chunk_size=128):
    """Mean nearest rootless-pose distance using PartMM's quaternion geodesic."""

    database_poses = np.asarray(database_poses, dtype=np.float64)
    query_poses = np.asarray(query_poses, dtype=np.float64)
    if database_poses.ndim != 2 or query_poses.ndim != 2:
        raise ValueError("poses must have shape [frames, axis-angle dimensions]")
    if database_poses.shape[1] != query_poses.shape[1]:
        raise ValueError("database and query pose dimensions differ")
    if database_poses.shape[1] == 72:
        database_poses = database_poses.reshape(-1, 24, 3)[:, 1:]
        query_poses = query_poses.reshape(-1, 24, 3)[:, 1:]
    elif database_poses.shape[1] == 69:
        database_poses = database_poses.reshape(-1, 23, 3)
        query_poses = query_poses.reshape(-1, 23, 3)
    else:
        raise ValueError("expected 69 rootless or 72 full SMPL pose dimensions")
    if not len(database_poses) or not len(query_poses):
        raise ValueError("database and query poses must be nonempty")
    database_q = _axis_angle_quaternion(database_poses)
    query_q = _axis_angle_quaternion(query_poses)
    minima = []
    for start in range(0, len(query_q), int(chunk_size)):
        query_chunk = query_q[start : start + int(chunk_size)]
        dots = np.abs(
            np.einsum("qjc,djc->qdj", query_chunk, database_q)
        )
        angles = np.arccos(np.clip(dots, 0.0, 1.0))
        minima.extend(angles.mean(axis=-1).min(axis=-1).tolist())
    return float(np.mean(minima))


def _load_pose(path):
    with Path(path).open("rb") as handle:
        record = pickle.load(handle)
    pose = np.asarray(record["pose"], dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] not in {69, 72}:
        raise ValueError(f"invalid SMPL pose array in {path}: {pose.shape}")
    return pose


def _sequence_paths(vto_root, expected_cases):
    root = Path(vto_root) / "tshirt" / "simulations"
    excluded = "tshirt_shape00_104_17"
    paths = [
        path
        for path in sorted(root.glob("tshirt_shape00_*.pkl"))
        if path.stem != excluded and path.stem not in set(expected_cases)
    ]
    queries = [root / f"{case}.pkl" for case in expected_cases]
    missing = [str(path) for path in queries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing VTO query sequences: {missing}")
    return paths, queries


def profile_database_sizes(
    output,
    *,
    sizes,
    runner_code_root=None,
    raw_root,
    vto_root,
    smpl_model_path,
    smpl_fk_model_dir=None,
    ssdr_cache_template=None,
    expected_cases,
    device="cuda",
):
    """Run PartMM and write the raw measurements required by Table R.1."""

    import torch

    sizes = [int(size) for size in sizes]
    training_paths, query_paths = _sequence_paths(vto_root, expected_cases)
    query_poses = np.concatenate([_load_pose(path) for path in query_paths])
    measurements = []
    for size in sizes:
        if len(training_paths) < size:
            raise ValueError(
                f"requested {size} training sequences, found {len(training_paths)}"
            )
        database_poses = np.concatenate(
            [_load_pose(path) for path in training_paths[:size]]
        )
        run_dir = Path(raw_root) / f"n{size}"
        per_case = run_dir / "per_case_summary.json"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        run_patch_length(
            7,
            run_dir,
            code_root=runner_code_root,
            vto_root=vto_root,
            smpl_model_path=smpl_model_path,
            smpl_fk_model_dir=smpl_fk_model_dir,
            ssdr_cache=(str(ssdr_cache_template).format(sequences=size)
                        if ssdr_cache_template else str(Path(raw_root) / f"fit_n{size}")),
            expected_cases=expected_cases,
            device=device,
            n_train=size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024.0 ** 2)
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0
        rows = json.loads(per_case.read_text(encoding="utf-8"))
        rows = [row for row in rows if row.get("variant") == "default"]
        frames = sum(int(row["frames"]) for row in rows)
        measurements.append(
            {
                "sequences": size,
                "database_frames": int(len(database_poses)),
                "query_frames": int(frames),
                "min_so3": minimum_pose_coverage(database_poses, query_poses),
                "build_s": float(rows[0]["build_s"]),
                "predict_ms": float(
                    sum(float(row["predict_s"]) for row in rows) * 1000.0 / frames
                ),
                "gpu_mb": float(peak_allocated),
                "gpu_reserved_mb": float(peak_reserved),
                "per_case_source": _source(per_case),
                "rmse_mm": float(np.mean([float(row["rmse_mm"]) for row in rows])),
            }
        )
    result = {
        "schema_version": 1,
        "metric_definition": {
            "min_so3": "mean over query frames of the nearest database pose's mean rootless-joint quaternion distance acos(|q1 dot q2|) in radians",
            "predict_ms": "total prediction wall time divided by all evaluated frames",
            "gpu_mb": "torch CUDA peak allocated bytes converted to MiB",
        },
        "measurements": measurements,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    accuracy = {
        "train_sizes": sizes,
        "mm_rmse": [row["rmse_mm"] for row in measurements],
        "evaluation_cases": list(expected_cases),
        "source": str(output.resolve()),
    }
    (output.parent / "accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
    return result


def _source(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def evaluate_database_scaling(
    sequences,
    accuracy_source,
    profile_source,
    output,
    *,
    expected_sizes=(5, 20, 50),
):
    """Validate accuracy and profiler outputs for Table R.1."""

    expected_sizes = [int(value) for value in expected_sizes]
    sequences = int(sequences)
    if sequences not in expected_sizes:
        raise ValueError(f"unexpected database size: {sequences}")
    accuracy_path = Path(accuracy_source).resolve()
    profile_path = Path(profile_source).resolve()
    accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    sizes = [int(value) for value in accuracy.get("train_sizes", [])]
    values = accuracy.get("mm_rmse", [])
    if len(sizes) != len(values) or not set(expected_sizes).issubset(sizes):
        raise ValueError("accuracy source does not cover Table R.1 database sizes")
    rmse = float(values[sizes.index(sequences)])

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = profile.get("measurements", [])
    if not isinstance(rows, list):
        raise ValueError("profile measurements must be a list")
    by_size = {}
    for row in rows:
        missing = REQUIRED_PROFILE_FIELDS - set(row)
        if missing:
            raise KeyError(f"profile row is missing {sorted(missing)}")
        key = int(row["sequences"])
        if key in by_size:
            raise ValueError(f"duplicate profile row for {key} sequences")
        by_size[key] = row
    if set(by_size) != set(expected_sizes):
        raise ValueError(
            f"profile sizes differ: expected {expected_sizes}, got {sorted(by_size)}"
        )
    row = by_size[sequences]
    metrics = {
        "rmse_mm": rmse,
        "min_so3": float(row["min_so3"]),
        "build_s": float(row["build_s"]),
        "predict_ms": float(row["predict_ms"]),
        "gpu_mb": float(row["gpu_mb"]),
    }
    if any(value < 0 for value in metrics.values()):
        raise ValueError("database scaling metrics must be non-negative")
    result = {
        "kind": "database_scaling",
        "table": "Table R.1",
        "database_sequences": sequences,
        "metrics": metrics,
        "protocol": {
            "database_sizes": expected_sizes,
            "accuracy_source": "independent VTO scalability evaluation",
            "profile_source": "raw PartMM coverage/runtime/memory measurement",
        },
        "sources": [_source(accuracy_path), _source(profile_path)],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sequences", type=int)
    parser.add_argument("--accuracy-source")
    parser.add_argument("--profile-source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-sizes", default="5,20,50")
    parser.add_argument("--runner-code-root")
    parser.add_argument("--raw-root")
    parser.add_argument("--vto-root")
    parser.add_argument("--smpl-model-path")
    parser.add_argument("--smpl-fk-model-dir")
    parser.add_argument("--ssdr-cache-template")
    parser.add_argument("--expected-cases")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.profile:
        required = [
            "raw_root",
            "vto_root",
            "smpl_model_path",
            "expected_cases",
        ]
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(f"profiling arguments are missing: {missing}")
        profile_database_sizes(
            args.profile_source,
            sizes=[int(value) for value in args.expected_sizes.split(",")],
            runner_code_root=args.runner_code_root,
            raw_root=args.raw_root,
            vto_root=args.vto_root,
            smpl_model_path=args.smpl_model_path,
            smpl_fk_model_dir=args.smpl_fk_model_dir,
            ssdr_cache_template=args.ssdr_cache_template,
            expected_cases=args.expected_cases.split(","),
            device=args.device,
        )
        print(args.profile_source)
        return
    required = ["sequences", "accuracy_source", "output"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"evaluation arguments are missing: {missing}")
    evaluate_database_scaling(
        args.sequences,
        args.accuracy_source,
        args.profile_source,
        args.output,
        expected_sizes=[int(value) for value in args.expected_sizes.split(",")],
    )
    print(args.output)


if __name__ == "__main__":
    main()
