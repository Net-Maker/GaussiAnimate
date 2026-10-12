"""Common experiment entry point after a backend exports registered 4D points."""

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from partmm import PartMMConfig, animate
from skelebone import ExtractionConfig, extract_skelebones
from skelebone.io import load_point_sequence, load_pose
from skelebone.visualization import save_skelebone_figure

from .config import load_config, resolve_skelebone_config
from .visualization import (
    save_animation_figure,
    save_reconstruction_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured GaussiAnimate experiment")
    parser.add_argument("config")
    parser.add_argument("--novel-pose")
    parser.add_argument("--ground-truth-points", help="Optional [T,N,3] points for animation RMSE")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, require_environment=not args.validate_only)
    if args.validate_only:
        print(f"valid: {config['experiment']['name']}")
        return

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output / "config.yaml")
    (output / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    effective_skelebone = resolve_skelebone_config(config)
    manifest = {
        "experiment": config["experiment"]["name"],
        "dataset": config["dataset"]["name"],
        "dataset_version": config["dataset"]["version"],
        "seed": config["experiment"]["seed"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "checkpoint": config.get("reconstruction", {}).get("checkpoint"),
        "point_sequence": str(Path(config["dataset"]["point_sequence"]).resolve()),
        "canonical_mesh": (
            str(Path(config["dataset"]["canonical_mesh"]).resolve())
            if config["dataset"].get("canonical_mesh")
            else None
        ),
        "novel_pose": str(Path(args.novel_pose).resolve()) if args.novel_pose else None,
        "ground_truth_points": (
            str(Path(args.ground_truth_points).resolve())
            if args.ground_truth_points
            else None
        ),
        "effective_skelebone": effective_skelebone,
        "partmm": config["partmm"],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sequence = load_point_sequence(config["dataset"]["point_sequence"])
    extraction = extract_skelebones(
        sequence,
        output / "skelebone",
        mesh=config["dataset"].get("canonical_mesh"),
        config=ExtractionConfig(**effective_skelebone),
    )
    requested_metrics = set(config["evaluation"].get("metrics", []))
    supported_metrics = {"ssdr_rmse", "point_rmse", "point_mean_error", "normalized_rmse"}
    unsupported = requested_metrics - supported_metrics
    if unsupported:
        raise ValueError(f"unsupported evaluation metrics: {', '.join(sorted(unsupported))}")
    metrics = {}
    visualization_dir = output / "visualization"
    if config["evaluation"].get("save_visualization"):
        skinning_path = output / "skelebone" / extraction.bundle.artifacts[
            "skinning_curve_skeleton"
        ]
        skinning = np.load(skinning_path, allow_pickle=False)
        save_skelebone_figure(
            extraction.bundle.source_points,
            skinning["curve_points"],
            skinning["curve_edges"],
            extraction.bundle.skeleton.rest_joints,
            extraction.bundle.skeleton.parents,
            visualization_dir / "skelebone.png",
        )
    ssdr_relative = extraction.bundle.artifacts.get("ssdr")
    if ssdr_relative:
        ssdr_path = output / "skelebone" / ssdr_relative
        if ssdr_path.is_file():
            ssdr = np.load(ssdr_path, allow_pickle=False)
            if "rmse" in ssdr and "ssdr_rmse" in requested_metrics:
                metrics["ssdr_rmse"] = float(ssdr["rmse"])
            if config["evaluation"].get("save_visualization") and "reconstruction" in ssdr:
                save_reconstruction_figure(
                    sequence.points,
                    ssdr["reconstruction"],
                    visualization_dir / "reconstruction.png",
                )
    if args.novel_pose:
        result = animate(extraction.bundle, load_pose(args.novel_pose), PartMMConfig(**config["partmm"]))
        np.savez_compressed(
            output / "animation.npz",
            points=result.points,
            bone_transforms=result.bone_transforms,
            joint_positions=result.joint_positions,
            fps=np.asarray(result.fps),
        )
        if args.ground_truth_points:
            ground_truth = load_point_sequence(args.ground_truth_points).points
            if ground_truth.shape != result.points.shape:
                raise ValueError("ground-truth animation shape differs from PartMM output")
            distance = np.linalg.norm(result.points - ground_truth, axis=-1)
            point_rmse = float(np.sqrt(np.mean(distance * distance)))
            if "point_rmse" in requested_metrics:
                metrics["point_rmse"] = point_rmse
            if "point_mean_error" in requested_metrics:
                metrics["point_mean_error"] = float(distance.mean())
            if "normalized_rmse" in requested_metrics:
                extent = np.ptp(ground_truth.reshape(-1, 3), axis=0)
                diagonal = float(np.linalg.norm(extent))
                if diagonal <= 1e-12:
                    raise ValueError("cannot normalize RMSE for zero-extent ground truth")
                metrics["normalized_rmse"] = point_rmse / diagonal
        if config["evaluation"].get("save_visualization"):
            save_animation_figure(
                result.points,
                result.joint_positions,
                extraction.bundle.skeleton.parents,
                visualization_dir / "animation.png",
            )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest["status"] = "complete"
    manifest["skelebones"] = str(extraction.bundle_path)
    manifest["bvh"] = str(extraction.bvh_path)
    manifest["metrics"] = str(output / "metrics.json")
    manifest["requested_metrics"] = sorted(requested_metrics)
    manifest["unavailable_metrics"] = sorted(requested_metrics - metrics.keys())
    if visualization_dir.is_dir():
        manifest["visualization"] = str(visualization_dir)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
