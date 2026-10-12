"""Render progressive skelebone results with one shared scientific view."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from skelebone import load_bundle
from skelebone.visualization import sample_indices


def _orthographic_project(points, azimuth=-68.0, elevation=12.0):
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    view = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    right = np.cross(view, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    return np.stack((points @ right, points @ up), axis=-1)


def save_progressive_figure(inputs, output: Path) -> Path:
    import matplotlib.pyplot as plt

    records = [(label, load_bundle(path)) for label, path in inputs]
    if not records:
        raise ValueError("at least one LABEL=PATH input is required")
    all_joints = np.concatenate(
        [bundle.skeleton.rest_joints for _, bundle in records],
        axis=0,
    )
    joint_low = all_joints.min(axis=0)
    joint_high = all_joints.max(axis=0)
    margin = max(float((joint_high - joint_low).max()) * 0.08, 1e-3)
    crop_low = joint_low - margin
    crop_high = joint_high + margin
    reference_points = records[0][1].source_points
    reference_points = reference_points[
        np.all(
            (reference_points >= crop_low)
            & (reference_points <= crop_high),
            axis=1,
        )
    ]
    reference = _orthographic_project(
        np.concatenate((reference_points, all_joints), axis=0)
    )
    low = reference.min(axis=0)
    high = reference.max(axis=0)
    padding = max(float((high - low).max()) * 0.035, 1e-3)
    figure, axes = plt.subplots(
        1,
        len(records),
        figsize=(2.7 * len(records), 2.8),
        facecolor="white",
    )
    axes = np.atleast_1d(axes)
    for axis, (label, bundle) in zip(axes, records):
        points = bundle.source_points
        visible = points[
            np.all((points >= crop_low) & (points <= crop_high), axis=1)
        ]
        sampled = visible[sample_indices(visible, max_points=4000)]
        sampled = _orthographic_project(sampled)
        axis.scatter(
            sampled[:, 0],
            sampled[:, 1],
            s=0.75,
            c="#c7cdd1",
            alpha=0.24,
        )
        joints = bundle.skeleton.rest_joints
        projected_joints = _orthographic_project(joints)
        for child, parent in enumerate(bundle.skeleton.parents):
            if parent >= 0:
                segment = projected_joints[[int(parent), child]]
                axis.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color="#087e8b",
                    linewidth=1.15,
                )
        axis.scatter(
            projected_joints[:, 0],
            projected_joints[:, 1],
            s=6,
            c="#d45f3e",
            zorder=3,
        )
        axis.set_xlim(low[0] - padding, high[0] + padding)
        axis.set_ylim(low[1] - padding, high[1] + padding)
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(label, fontsize=11, pad=4)
    figure.tight_layout(pad=0.2, w_pad=0.45)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_progressive_summary(
    inputs,
    figure: Path,
    output: Path,
    tool_files=(),
) -> Path:
    columns = []
    for label, path in inputs:
        bundle = load_bundle(path)
        columns.append(
            {
                "label": label,
                "frames": bundle.motion.num_frames,
                "points": len(bundle.source_points),
                "outer_bones": bundle.num_bones,
                "inner_joints": bundle.skeleton.num_joints,
                "skelebones_sha256": _sha256(path),
                "artifacts": {
                    name: _sha256(path.parents[1] / name)
                    for name in (
                        "full_sequence.npz",
                        "observed_sequence.npz",
                        "extraction.json",
                    )
                    if (path.parents[1] / name).is_file()
                },
            }
        )
    result = {
        "schema_version": 1,
        "target_id": "progressive_skeletonization",
        "target": "Fig. 9 / progressive skeletonization",
        "protocol": {
            "sequence": "DG-Mesh T-Rex",
            "columns": ["0%", "30%", "60%", "80%", "100%"],
            "frame_counts": [1, 60, 120, 160, 200],
            "points_per_frame": 30000,
            "requested_outer_bones": 50,
            "joint_score_threshold": 0.3,
        },
        "columns": columns,
        "figure": {
            "file": Path(figure).name,
            "sha256": _sha256(figure),
        },
        "implementation": {
            "progressive_visualization.py": _sha256(__file__),
            "tools": {
                label: {
                    "file": path.name,
                    "sha256": _sha256(path),
                }
                for label, path in tool_files
            },
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _input(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("input must use LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_input, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--tool", action="append", type=_input, default=[])
    args = parser.parse_args()
    figure = save_progressive_figure(args.input, args.output)
    if args.summary:
        save_progressive_summary(
            args.input,
            figure,
            args.summary,
            tool_files=args.tool,
        )
    print(figure)


if __name__ == "__main__":
    main()
