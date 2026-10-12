"""Summarize a completed PartMM repair run without changing its metrics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    results = json.loads((args.run / "summary.json").read_text())
    methods = ["previous", "per_frame_alignment", "robust_skin_transfer", "guarded"]
    labels = ["Previous PartMM", "Geometry fixes", "Robust Skin Transfer", "With fallback"]
    colors = ["#90969c", "#317c9e", "#bc9660", "#348166"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), layout="constrained")
    lines = ["# PartMM Repair Check", "", "All position errors are Euclidean RMSE in millimeters on the same held-out frames and all Gaussians.", "",
             "| Case | Previous | Geometry fixes | Robust Skin Transfer | With fallback | Baseline share |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for ax, result in zip(axes.flat, results):
        values = [result["metrics"][method]["point_rmse_mm"] for method in methods]
        bars = ax.barh(labels, values, color=colors, height=.6)
        ax.invert_yaxis()
        ax.set_title(result["case"], fontsize=12, loc="left", pad=10)
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_xlabel("RMSE (mm)", fontsize=10)
        ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.grid(axis="x", alpha=.12)
        ax.set_axisbelow(True)
        fraction = max(0.0, min(1.0, result["fallback"]["baseline_fraction"]))
        lines.append(f"| {result['case']} | " + " | ".join(f"{value:.2f}" for value in values) + f" | {fraction:.1%} |")
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.savefig(args.run / "comparison.png", dpi=160)
    plt.close(fig)
    lines += ["", "Fallback is calibrated on the final 20% of the training prefix, with that segment excluded from the retrieval database. SSDR itself is fitted on the full training prefix. No test geometry is used for policy calibration or selection.", "",
              "The baseline uses cached frame-zero Robust Skin Transfer weights. A validation-based fallback does not guarantee a per-frame test error bound. The large-error sequences remain unresolved.", "",
              "See config.yaml for the exact policy and per-case metadata.json for validation errors, thresholds, masks, resources, and test-frame IDs.", ""]
    (args.run / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
