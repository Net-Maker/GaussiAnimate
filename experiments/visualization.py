"""Minimal, semantically colored experiment figures."""

from pathlib import Path

import numpy as np

def sample_indices(points, max_points=20000):
    """Use deterministic display-only subsampling, preserving point correspondence."""
    stride = max(1, int(np.ceil(len(points) / max_points)))
    return np.arange(0, len(points), stride)


def equal_3d_axes(axis, points):
    """Frame all panels with the same orthographic scale."""
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_proj_type("ortho")
    axis.set_axis_off()


def save_reconstruction_figure(observation, prediction, output_path, frame=None) -> Path:
    """Save the standard Input | Prediction | Error comparison."""
    import matplotlib.pyplot as plt

    observation = np.asarray(observation)
    prediction = np.asarray(prediction)
    if observation.shape != prediction.shape or observation.ndim != 3:
        raise ValueError("observation and prediction must share shape [T,N,3]")
    errors = np.linalg.norm(observation - prediction, axis=-1)
    frame = int(np.argmax(errors.mean(axis=1))) if frame is None else int(frame)
    indices = sample_indices(observation[frame])
    obs, pred, error = observation[frame, indices], prediction[frame, indices], errors[frame, indices]
    combined = np.concatenate([obs, pred])
    figure = plt.figure(figsize=(12, 4), facecolor="white")
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    axes[0].scatter(*obs.T, s=1.5, c="#69747c", depthshade=False)
    axes[1].scatter(*pred.T, s=1.5, c="#087e8b", depthshade=False)
    colored = axes[2].scatter(*pred.T, s=2, c=error, cmap="inferno", depthshade=False)
    for axis, title in zip(axes, ["Input", "Prediction", "Error"]):
        axis.set_title(title, fontsize=11)
        equal_3d_axes(axis, combined)
        axis.view_init(elev=12, azim=-68)
    figure.colorbar(colored, ax=axes[2], fraction=0.04, pad=0.02, label="Euclidean error")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def save_animation_figure(points, joints, parents, output_path) -> Path:
    """Show temporal behavior at the first, middle, and final frame."""
    import matplotlib.pyplot as plt

    points, joints = np.asarray(points), np.asarray(joints)
    if len(points) != len(joints):
        raise ValueError("point and joint animations must have equal frame counts")
    frames = np.unique([0, len(points) // 2, len(points) - 1])
    figure = plt.figure(figsize=(4 * len(frames), 4), facecolor="white")
    all_points = points.reshape(-1, 3)
    for panel, frame in enumerate(frames):
        axis = figure.add_subplot(1, len(frames), panel + 1, projection="3d")
        indices = sample_indices(points[frame])
        axis.scatter(*points[frame, indices].T, s=1, c="#b7bec3", alpha=0.35, depthshade=False)
        for child, parent in enumerate(parents):
            if parent >= 0:
                segment = joints[frame, [int(parent), child]]
                axis.plot(*segment.T, color="#087e8b", linewidth=1.8)
        axis.scatter(*joints[frame].T, s=10, c="#d45f3e", depthshade=False)
        axis.set_title(f"Frame {frame}", fontsize=11)
        equal_3d_axes(axis, all_points)
        axis.view_init(elev=12, azim=-68)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
