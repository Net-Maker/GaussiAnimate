"""Run leak-free human PartMM from training-only SSDR and SMPL motion."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.methods import (
    HeldOutPointProtocol,
    SSDRCache,
    point_error_metrics,
    save_prediction_diagnostic,
    write_prediction_artifacts,
)
from partmm import PartMMConfig, animate
from skelebone import PoseSequence, SkeleboneBundle, Skeleton


def run_human_partmm(
    params_path: Path,
    cache_path: Path,
    motion_path: Path,
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    config: PartMMConfig = PartMMConfig(),
) -> dict:
    with np.load(params_path, allow_pickle=False) as params:
        ground_truth = np.asarray(params["means3D"], dtype=np.float32)
    protocol = HeldOutPointProtocol.from_shape(
        ground_truth.shape,
        train_ratio,
    )
    frames = protocol.frames
    points = protocol.points
    training_frames = protocol.training_frames
    cache = SSDRCache.load(cache_path)
    rest = cache.rest_pose
    weights = cache.weights
    transforms = cache.world_transformations
    fitted = cache.training_frames
    if fitted != training_frames or len(transforms) != training_frames:
        raise ValueError(
            "SSDR cache is not training-only: expected "
            f"{training_frames} fitted frames, found {fitted}/{len(transforms)}"
        )
    if rest.shape != (points, 3):
        raise ValueError("SSDR cache and D3DGS params use different point sets")
    with np.load(motion_path, allow_pickle=False) as motion:
        local_rotations = np.asarray(motion["local_rotations"], dtype=np.float64)
        root_translation = np.asarray(motion["root_translation"], dtype=np.float64)
        rest_joints = np.asarray(motion["rest_joints"], dtype=np.float64)
        parents = np.asarray(motion["parents"], dtype=np.int64)
        names = motion["names"].tolist() if "names" in motion else None
        fps = float(motion["fps"]) if "fps" in motion else 30.0
        case = str(motion["case"]) if "case" in motion else None
        dataset = str(motion["dataset"]) if "dataset" in motion else None
    if len(local_rotations) != frames or len(root_translation) != frames:
        raise ValueError("SMPL motion and D3DGS params must have the same frame count")
    skeleton = Skeleton(rest_joints, parents, names)
    source_motion = PoseSequence(
        local_rotations[:training_frames],
        root_translation[:training_frames],
        fps,
    )
    novel_motion = PoseSequence(
        local_rotations[training_frames:],
        root_translation[training_frames:],
        fps,
    )
    bundle = SkeleboneBundle(
        skeleton=skeleton,
        weights=weights,
        source_points=rest,
        motion=source_motion,
        bone_transforms=transforms,
    )
    result = animate(bundle, novel_motion, config)
    target = ground_truth[training_frames:]
    metrics = point_error_metrics(result.points, target)
    output_dir = Path(output_dir)
    artifacts = write_prediction_artifacts(
        output_dir,
        protocol,
        test_prediction=result.points,
        training_prefix=ground_truth[:training_frames],
        bone_transforms=result.bone_transforms,
    )
    save_prediction_diagnostic(
        output_dir / "diagnostic.png",
        ground_truth[0],
        result.points[0],
        target[0],
    )
    artifacts["diagnostic"] = "diagnostic.png"
    metadata = {
        "schema_version": 1,
        "case": case,
        "dataset": dataset,
        **protocol.metadata(),
        "warped_sequence_training_payload": (
            "ground_truth_context_not_partmm_predictions"
        ),
        "ssdr_fitted_frames": fitted,
        "partmm": {
            "num_parts": config.num_parts,
            "patch_size": config.patch_size,
            "num_neighbors": config.num_neighbors,
            "pyramid_levels": config.pyramid_levels,
            "pyramid_blend": config.pyramid_blend,
            "overlap_rings": config.overlap_rings,
            "align_parts": config.align_parts,
            "alignment_mode": config.alignment_mode,
            "partition_mode": config.partition_mode,
            "bone_overlap_ratio": config.bone_overlap_ratio,
            "alignment_frames": config.alignment_frames,
            "blend_space": config.blend_space,
            "blend_temperature": config.blend_temperature,
        },
        "fallback": {"enabled": False, "mode": "raw_retrieval"},
        "metrics": metrics,
        "artifacts": artifacts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training-only SMPL + PartMM")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--ssdr-cache", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-parts", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--pyramid-levels", type=int, default=5)
    parser.add_argument("--pyramid-blend", type=float, default=0.7)
    parser.add_argument(
        "--alignment-mode",
        choices=("part_rigid", "edge_wise"),
        default="edge_wise",
    )
    parser.add_argument(
        "--partition-mode",
        choices=("balanced_tree", "humanoid_five"),
        default="humanoid_five",
    )
    parser.add_argument("--bone-overlap-ratio", type=float, default=1.5)
    args = parser.parse_args()
    metadata = run_human_partmm(
        args.params,
        args.ssdr_cache,
        args.motion,
        args.output_dir,
        train_ratio=args.train_ratio,
        config=PartMMConfig(
            num_parts=args.num_parts,
            patch_size=args.patch_size,
            num_neighbors=args.neighbors,
            pyramid_levels=args.pyramid_levels,
            pyramid_blend=args.pyramid_blend,
            alignment_mode=args.alignment_mode,
            partition_mode=args.partition_mode,
            bone_overlap_ratio=args.bone_overlap_ratio,
        ),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
