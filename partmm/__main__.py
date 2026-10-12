"""Command-line PartMM animation."""

import argparse
import json

from .animator import PartMMConfig, animate
from .io import load_input, load_query, save_output


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m partmm",
        description="Retrieve outer-bone motion for a novel inner-skeleton pose sequence.",
    )
    parser.add_argument("bundle", help="Source skelebones.npz")
    parser.add_argument("novel_pose", help="Novel pose .npz")
    parser.add_argument("output", nargs="?", help="Output animation .npz")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--num-parts", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--pyramid-levels", type=int, default=5)
    parser.add_argument("--pyramid-blend", type=float, default=0.7)
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--alignment-mode",
        choices=("part_rigid", "edge_wise"),
        default="part_rigid",
    )
    parser.add_argument(
        "--partition-mode",
        choices=("balanced_tree", "humanoid_five"),
        default="balanced_tree",
    )
    parser.add_argument("--bone-overlap-ratio", type=float, default=1.5)
    args = parser.parse_args()
    config = PartMMConfig(
        num_parts=args.num_parts,
        patch_size=args.patch_size,
        num_neighbors=args.neighbors,
        pyramid_levels=args.pyramid_levels,
        pyramid_blend=args.pyramid_blend,
        align_parts=not args.no_align,
        alignment_mode=args.alignment_mode,
        partition_mode=args.partition_mode,
        bone_overlap_ratio=args.bone_overlap_ratio,
    )
    database = load_input(args.bundle)
    query = load_query(args.novel_pose)
    if args.validate_only:
        print(json.dumps({
            "source_frames": database.motion.num_frames,
            "query_frames": query.num_frames,
            "joints": database.skeleton.num_joints,
            "bones": database.num_bones,
            "points": len(database.source_points),
        }, indent=2))
        return
    if args.output is None:
        parser.error("output is required unless --validate-only is used")
    result = animate(database, query, config)
    save_output(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
