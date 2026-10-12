"""Compare alignment repair and training-calibrated Robust LBS fallback."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import yaml

from partmm import PartMMConfig, PartMMInput, PartMMQuery, PartMotionMatcher
from partmm.fallback import PartMMFallback
from partmm.matching import frame_match_distances
from partmm.partition import partition_skeleton


def metrics(prediction, target):
    squared = np.sum((np.asarray(prediction, dtype=np.float64) - target) ** 2, axis=-1)
    return {
        "point_rmse_mm": float(np.sqrt(squared.mean()) * 1000),
        "point_mean_error_mm": float(np.sqrt(squared).mean() * 1000),
    }


def run_case(settings, case):
    root = Path(settings["release_root"])
    dataset, name = case["dataset"], case["id"]
    case_run = root / "runs/multiview_novel_pose" / dataset / name
    run = case_run / "smpl_partmm"
    config = PartMMConfig(**settings["partmm"])
    with np.load(run / "ssdr/train_only.npz", allow_pickle=False) as cache:
        rest = cache["rest_pose"]
        weights = cache["weights"].T
        transforms = cache["world_transformations"]
    with np.load(run / "motion/smpl_motion.npz", allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    train = len(transforms)
    frames = len(motion["local_rotations"])
    if train != int(frames * settings["train_ratio"]):
        raise ValueError("SSDR cache does not match the configured training split")
    validation_start = int(train * (1 - settings["validation_ratio"]))
    if min(validation_start, train - validation_start) < config.patch_size:
        raise ValueError("training calibration needs at least one patch on each side")
    params_path = root / "resources/multiview/d3dgs" / dataset / name / "params.npz"
    with np.load(params_path, allow_pickle=False) as params:
        observed = params["means3D"]
    if observed.shape[:2] != (frames, len(rest)):
        raise ValueError("Gaussian parameters and source cache have different shapes")
    baseline_path = case_run / "smpl_lbs/lbs/warped_sequence.npy"
    baseline = np.load(baseline_path, mmap_mode="r", allow_pickle=False)
    baseline_metadata = json.loads((baseline_path.parent / "metadata.json").read_text())
    if baseline_metadata["protocol"]["fit_frames"] != [0]:
        raise ValueError("baseline must use robust weights fitted on frame zero only")
    if baseline.shape != observed.shape:
        raise ValueError("baseline and Gaussian sequence shapes differ")
    with np.load(baseline_path.parent / "robust_weights.npz", allow_pickle=False) as archive:
        robust_weights = archive["gaussian_weights"]
    bundle = PartMMInput(
        rest_joints=motion["rest_joints"], parents=motion["parents"],
        joint_names=motion["names"].tolist(), source_points=rest, skinning_weights=weights,
        source_local_rotations=motion["local_rotations"][:train],
        source_root_translation=motion["root_translation"][:train],
        bone_transforms=transforms, fps=float(motion["fps"]),
    )
    calibration_bundle = replace(
        bundle, source_local_rotations=bundle.source_local_rotations[:validation_start],
        source_root_translation=bundle.source_root_translation[:validation_start],
        bone_transforms=transforms[:validation_start],
    )
    calibration_query = PartMMQuery(
        motion["local_rotations"][validation_start:train],
        motion["root_translation"][validation_start:train], bundle.fps,
    )
    calibration_matcher = PartMotionMatcher(calibration_bundle, config)
    validation_candidate = calibration_matcher.animate(calibration_query).points
    validation_distances = frame_match_distances(
        calibration_bundle.source_local_rotations, calibration_query.local_rotations,
        calibration_matcher.part_masks, config.patch_size,
    )
    _, components = partition_skeleton(bundle.parents, config.num_parts, bundle.joint_names, config.partition_mode)
    part_weights = np.stack([robust_weights[:, sorted(part)].sum(axis=1) for part in components], axis=1)
    guard = PartMMFallback.calibrate(
        validation_candidate, baseline[validation_start:train], observed[validation_start:train],
        part_weights, validation_distances, distance_quantile=settings["distance_quantile"],
        distance_policy=settings.get("distance_policy", "validation"),
    )
    query = PartMMQuery(motion["local_rotations"][train:], motion["root_translation"][train:], bundle.fps)
    matcher = PartMotionMatcher(bundle, config)
    candidate = matcher.animate(query)
    distances = frame_match_distances(bundle.source_local_rotations, query.local_rotations, matcher.part_masks, config.patch_size)
    guarded, diagnostics = guard.apply(candidate.points, baseline[train:], distances)
    # Test geometry is used only for reporting, after the policy has been frozen.
    target = observed[train:]
    old_path = run / "partmm" / name / "test_warped_sequence.npy"
    old = np.load(old_path, mmap_mode="r", allow_pickle=False)
    result = {
        "case": name, "dataset": dataset, "partmm": asdict(config),
        "training_frames": train, "validation_frame_range": [validation_start, train],
        "test_frame_range": [train, frames], "seed": settings["seed"],
        "test_frame_ids": list(range(train, frames)),
        "calibration_reference": "observed_training_gaussians",
        "ssdr_fit_scope": "all_training_frames_including_internal_validation",
        "retrieval_calibration_scope": "prefix_before_internal_validation",
        "resources": {"params": str(params_path), "ssdr": str(run / "ssdr/train_only.npz"),
                      "robust_baseline": str(baseline_path), "old_prediction": str(old_path)},
        "metrics": {"previous": metrics(old, target), "robust_skin_transfer": metrics(baseline[train:], target),
                    "per_frame_alignment": metrics(candidate.points, target), "guarded": metrics(guarded, target)},
        "fallback": diagnostics,
    }
    output = Path(settings["output_dir"]) / name
    output.mkdir(parents=True, exist_ok=True)
    if settings.get("save_predictions", False):
        np.save(output / "test_warped_sequence.npy", guarded.astype(np.float32))
        np.save(output / "frame_ids.npy", np.arange(train, frames))
        np.save(output / "point_indices.npy", np.arange(len(rest)))
        full = np.lib.format.open_memmap(output / "warped_sequence.npy", mode="w+", dtype=np.float32, shape=observed.shape)
        full[:train] = observed[:train]
        full[train:] = guarded
        full.flush()
        del full
        result["warped_sequence_training_payload"] = "ground_truth_context_not_predictions"
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"case": name, "metrics": result["metrics"], "fallback": diagnostics}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    cases = [case for case in settings["cases"] if not args.case or case["id"] in args.case]
    if not cases:
        raise ValueError("no configured cases selected")
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    results = []
    for case in cases:
        results.append(run_case(settings, case))
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
