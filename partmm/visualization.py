"""Minimal temporal visualization for PartMM outputs.

Matplotlib is imported lazily so the PartMM runtime itself remains NumPy-only.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .types import PartMMOutput


def outer_bone_positions(source_points, skinning_weights, bone_transforms):
    """Track SSDR influence centroids through a retrieved transform sequence."""
    source_points = np.asarray(source_points, dtype=np.float64)
    weights = np.asarray(skinning_weights, dtype=np.float64)
    transforms = np.asarray(bone_transforms, dtype=np.float64)
    if weights.shape[0] != len(source_points):
        raise ValueError("skinning_weights must have shape [points,bones]")
    if transforms.ndim != 4 or transforms.shape[1:] != (weights.shape[1], 4, 4):
        raise ValueError("bone_transforms must have shape [frames,bones,4,4]")
    weight_sums = weights.sum(axis=0)
    if np.any(weight_sums <= 1e-12):
        raise ValueError("every outer bone must influence at least one point")
    rest_centers = (weights.T @ source_points) / weight_sums[:, None]
    centers_h = np.concatenate(
        [rest_centers, np.ones((len(rest_centers), 1), dtype=np.float64)], axis=1
    )
    return np.einsum("tbij,bj->tbi", transforms, centers_h)[..., :3]


def save_animation_gif(path, frames, fps=12.0):
    """Encode RGB arrays as a looping GIF for the optional visualizer."""
    from PIL import Image

    if fps <= 0 or not frames:
        raise ValueError("fps must be positive and frames must be non-empty")
    images = [Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB") for frame in frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return path


def _preview_frames(frame_count: int, frames: Optional[Sequence[int]]) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("the PartMM output must contain at least one frame")
    selected = (
        np.linspace(0, frame_count - 1, min(3, frame_count), dtype=np.int64)
        if frames is None
        else np.asarray(frames, dtype=np.int64)
    )
    if selected.ndim != 1 or not len(selected):
        raise ValueError("frames must be a non-empty one-dimensional sequence")
    if np.any(selected < 0) or np.any(selected >= frame_count):
        raise ValueError("frames contains an out-of-range frame index")
    return np.asarray(list(dict.fromkeys(selected.tolist())), dtype=np.int64)


def plot_preview(
    output: PartMMOutput,
    parents,
    frames: Optional[Sequence[int]] = None,
    max_points: int = 20000,
):
    """Create a shared-scale temporal preview and return its Matplotlib figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "PartMM visualization requires matplotlib; install it with "
            "`python -m pip install matplotlib`."
        ) from error

    if max_points < 1:
        raise ValueError("max_points must be positive")
    parents = np.asarray(parents, dtype=np.int64)
    if parents.shape != (output.joint_positions.shape[1],):
        raise ValueError("parents must have shape [joints]")
    selected = _preview_frames(len(output.points), frames)

    visible = np.concatenate(
        [output.points[selected].reshape(-1, 3), output.joint_positions[selected].reshape(-1, 3)]
    )
    lower = visible.min(axis=0)
    upper = visible.max(axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-3)

    figure = plt.figure(figsize=(4.2 * len(selected), 4.0), facecolor="white")
    for column, frame in enumerate(selected, start=1):
        axis = figure.add_subplot(1, len(selected), column, projection="3d")
        points = output.points[frame]
        stride = max(1, int(np.ceil(len(points) / max_points)))
        points = points[::stride]
        joints = output.joint_positions[frame]

        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=2.0, c="#AAB3B8", alpha=0.42, linewidths=0,
            depthshade=False, rasterized=True,
        )
        for child, parent in enumerate(parents):
            if parent >= 0:
                segment = joints[[int(parent), child]]
                axis.plot(
                    segment[:, 0], segment[:, 1], segment[:, 2],
                    color="#263238", linewidth=2.0,
                )
        axis.scatter(
            joints[:, 0], joints[:, 1], joints[:, 2],
            s=22, c="#D75A3A", edgecolors="white", linewidths=0.6,
            depthshade=False,
        )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=15, azim=-70, vertical_axis="y")
        axis.set_axis_off()
        axis.set_title(
            f"Frame {int(frame)}  |  {float(frame) / output.fps:.2f} s",
            fontsize=10,
            color="#263238",
            pad=4,
        )

    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91, wspace=0.02)
    return figure


def save_preview(
    path,
    output: PartMMOutput,
    parents,
    frames: Optional[Sequence[int]] = None,
    max_points: int = 20000,
    dpi: int = 180,
) -> Path:
    """Save a temporal PNG preview for a PartMM output."""
    import matplotlib.pyplot as plt

    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("PartMM previews must use the .png extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_preview(output, parents, frames=frames, max_points=max_points)
    try:
        figure.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


__all__ = [
    "outer_bone_positions",
    "plot_preview",
    "save_animation_gif",
    "save_preview",
]
