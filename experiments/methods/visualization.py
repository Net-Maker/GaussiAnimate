"""Minimal scientific diagnostics shared by animation methods."""

from pathlib import Path

import numpy as np


def save_prediction_diagnostic(
    path,
    canonical,
    prediction,
    target,
    seed=42,
) -> None:
    """Save a shared-view Input | Prediction | Error diagnostic."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = np.asarray(canonical)
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if canonical.shape != prediction.shape or prediction.shape != target.shape:
        raise ValueError(
            "diagnostic arrays must have the same [N, 3] shape"
        )

    count = min(10000, len(canonical))
    indices = np.random.default_rng(seed).choice(
        len(canonical),
        count,
        replace=False,
    )
    full_error = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    canonical = canonical[indices]
    prediction = prediction[indices]
    target = target[indices]
    error = full_error[indices]
    combined = np.concatenate([canonical, prediction, target], axis=0)
    lower, upper = np.quantile(combined, [0.005, 0.995], axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.52 * float(
        np.ptp(np.stack([lower, upper]), axis=0).max()
    )
    error_limit = max(float(np.quantile(error, 0.95)), 1e-6)

    figure = plt.figure(figsize=(12, 4), facecolor="white")
    panels = (
        ("Input", canonical, None),
        ("Prediction", prediction, None),
        ("Error", prediction, error),
    )
    for panel, (title, points, values) in enumerate(
        panels,
        start=1,
    ):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        colors = "#5d6670" if values is None else values
        scatter_options = {}
        if values is not None:
            scatter_options = {
                "cmap": "YlOrRd",
                "vmin": 0.0,
                "vmax": error_limit,
            }
        axis.scatter(
            points[:, 0],
            points[:, 2],
            -points[:, 1],
            c=colors,
            s=0.6,
            alpha=0.9,
            linewidths=0,
            **scatter_options,
        )
        figure.text(
            (panel - 0.5) / 3.0,
            0.965,
            title,
            ha="center",
            va="top",
            fontsize=11,
            color="#171a1d",
        )
        axis.set_proj_type("ortho")
        axis.view_init(elev=4, azim=-90)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[2] - radius, center[2] + radius)
        axis.set_zlim(-center[1] - radius, -center[1] + radius)
        axis.set_axis_off()
        if values is not None:
            figure.text(
                (panel - 0.5) / 3.0,
                0.04,
                f"mean {full_error.mean():.1f} mm",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#8b1a1a",
            )
    figure.tight_layout(
        rect=(0.0, 0.06, 1.0, 0.94),
        pad=0.4,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
