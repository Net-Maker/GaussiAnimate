"""Command-line entry point for dynamic point-sequence skeletonization."""

import argparse
import json

import numpy as np

from .api import extract_skelebones
from .io import load_point_sequence, point_sequence_summary
from .pipeline import ExtractionConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m skelebone",
        description="Extract SSDR, curve skeleton, skinning curve skeleton, kinematic tree, and BVH.",
    )
    parser.add_argument("input", help="Registered [T,N,3] .npy/.npz point sequence")
    parser.add_argument("output", nargs="?", help="Output directory")
    parser.add_argument("--mesh", help="Optional canonical triangle mesh for curve skeletonization")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-bones", type=int, default=50)
    parser.add_argument("--rest-frame", type=int, default=0)
    parser.add_argument(
        "--mesh-frame",
        type=int,
        default=None,
        help="frame the canonical mesh corresponds to (default: the rest frame)",
    )
    parser.add_argument("--joint-threshold", type=float, default=0.3)
    parser.add_argument("--joint-merge-distance", type=float, default=0.08)
    parser.add_argument("--curve-skeleton-bin")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check the input contract and print its dimensions without extracting",
    )
    args = parser.parse_args()
    sequence = load_point_sequence(args.input, fps=args.fps)
    if args.validate_only:
        print(json.dumps(point_sequence_summary(sequence), indent=2))
        return
    if not args.output:
        parser.error("output is required unless --validate-only is used")
    config = ExtractionConfig(
        num_bones=args.num_bones,
        rest_frame=args.rest_frame,
        mesh_frame=args.mesh_frame,
        joint_score_threshold=args.joint_threshold,
        joint_merge_distance=args.joint_merge_distance,
        curve_skeleton_bin=args.curve_skeleton_bin,
    )
    result = extract_skelebones(
        sequence,
        args.output,
        mesh=args.mesh,
        config=config,
    )
    if args.visualize:
        from .visualization import save_skelebone_figure

        skinning_path = result.bundle_path.parent / result.bundle.artifacts[
            "skinning_curve_skeleton"
        ]
        skinning = np.load(skinning_path, allow_pickle=False)
        figure_path = save_skelebone_figure(
            result.bundle.source_points,
            skinning["curve_points"],
            skinning["curve_edges"],
            result.bundle.skeleton.rest_joints,
            result.bundle.skeleton.parents,
            result.bundle_path.parent / "skelebone.png",
            faces=sequence.faces,
        )
        print(figure_path)
    print(result.bundle_path)
    print(result.bvh_path)
    print(result.manifest_path)


if __name__ == "__main__":
    main()
