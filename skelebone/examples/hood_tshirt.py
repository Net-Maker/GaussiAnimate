"""Extract and visualize Skelebones from the real HOOD T-shirt sequence."""

import argparse
import os
from pathlib import Path

import numpy as np

from skelebone.io import save_point_sequence
from skelebone.pipeline import ExtractionConfig, SkelebonePipeline
from skelebone.types import DynamicPointSequence
from skelebone.visualization import save_skelebone_figure


DEFAULT_CASE = "tshirt_shape00_91_33"


def load_hood_tshirt(hood_root, case=DEFAULT_CASE, fps=30.0):
    """Load a registered T-shirt GT sequence exported by the HOOD workflow."""
    path = (
        Path(hood_root).expanduser()
        / "output"
        / "precomputed_tshirt"
        / f"{case}.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing HOOD T-shirt result: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        missing = {"gt_vertices", "faces"}.difference(arrays.files)
        if missing:
            raise ValueError(
                "HOOD result is missing: " + ", ".join(sorted(missing))
            )
        points = np.asarray(arrays["gt_vertices"], dtype=np.float64)
        faces = np.asarray(arrays["faces"], dtype=np.int64)
    return DynamicPointSequence(points, faces=faces, fps=fps)


def run(
    output_directory,
    hood_root,
    *,
    case=DEFAULT_CASE,
    fps=30.0,
    num_bones=20,
    max_iterations=50,
    poisson_depth=8,
    target_triangles=12000,
    curve_skeleton_bin=None,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    sequence = load_hood_tshirt(hood_root, case=case, fps=fps)
    input_path = save_point_sequence(
        output_directory / "sequence.npz", sequence
    )
    result = SkelebonePipeline(
        ExtractionConfig(
            num_bones=num_bones,
            ssdr_max_iterations=max_iterations,
            mesh_poisson_depth=poisson_depth,
            mesh_target_triangles=target_triangles,
            curve_skeleton_bin=curve_skeleton_bin,
        )
    ).run(sequence, output_directory / "result")
    skinning_path = result.bundle_path.parent / result.bundle.artifacts[
        "skinning_curve_skeleton"
    ]
    with np.load(skinning_path, allow_pickle=False) as arrays:
        preview_path = save_skelebone_figure(
            result.bundle.source_points,
            arrays["curve_points"],
            arrays["curve_edges"],
            result.bundle.skeleton.rest_joints,
            result.bundle.skeleton.parents,
            output_directory / "hood_tshirt_skelebone.png",
            faces=sequence.faces,
        )
    return {
        "input": input_path,
        "bundle": result.bundle_path,
        "bvh": result.bvh_path,
        "manifest": result.manifest_path,
        "preview": preview_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Skelebones from the real HOOD/VTO T-shirt case "
            f"{DEFAULT_CASE}."
        )
    )
    parser.add_argument("output", nargs="?", default="outputs/hood_tshirt")
    parser.add_argument(
        "--hood-root",
        default=os.environ.get("GA_HOOD_ROOT", "~/research/GA/hood"),
    )
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-bones", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--target-triangles", type=int, default=12000)
    parser.add_argument("--curve-skeleton-bin")
    args = parser.parse_args()
    paths = run(
        args.output,
        args.hood_root,
        case=args.case,
        fps=args.fps,
        num_bones=args.num_bones,
        max_iterations=args.max_iterations,
        poisson_depth=args.poisson_depth,
        target_triangles=args.target_triangles,
        curve_skeleton_bin=args.curve_skeleton_bin,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
