"""Built-in stage handlers for the declarative experiment engine."""

import os
import shlex
import subprocess
from pathlib import Path

import numpy as np

from skelebone import ExtractionConfig, extract_skelebones

from experiments.mesh.export_dt4d import export_dt4d_points
from experiments.mesh.export_vto import export_vto_points
from experiments.metrics import (
    collect_json_metrics,
    evaluate_geometry,
    evaluate_image_directories,
)
from experiments.monocular_gaussian.prepare import export_riggs_points
from experiments.multiview_gaussian.prepare import export_d3dgs_points
from experiments.protocols import export_dt4d_high_motion, slice_point_sequence

from .registry import STAGE_REGISTRY, StageContext
from .storage import write_json


def _shell_join(command):
    """Format a command for logs on Python versions predating shlex.join."""

    return " ".join(shlex.quote(part) for part in command)


@STAGE_REGISTRY.register("command")
def run_command(stage, context: StageContext) -> None:
    command = [str(item) for item in stage["command"]]
    job = context.job
    cwd = Path(stage.get("cwd", job.config_path.parent))
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in stage.get("env", {}).items()}
    )
    if context.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(context.gpu)

    log_path = job.run_dir / "logs" / f"{stage['id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + _shell_join(command) + "\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"command exited with status {process.returncode}; see {log_path}"
        )


@STAGE_REGISTRY.register("image_metrics")
def run_image_metrics(stage, context: StageContext) -> None:
    evaluate_image_directories(
        stage["prediction_dir"],
        stage["reference_dir"],
        stage["output"],
        metrics=stage.get("metrics", ["psnr", "ssim", "lpips"]),
        mask_dir=stage.get("mask_dir"),
        lpips_network=stage.get("lpips_network", "vgg"),
        device=stage.get("device"),
    )


@STAGE_REGISTRY.register("geometry_metrics")
def run_geometry_metrics(stage, context: StageContext) -> None:
    evaluate_geometry(
        stage["prediction"],
        stage["reference"],
        stage["output"],
        prediction_key=stage.get("prediction_key", "points"),
        reference_key=stage.get("reference_key", "points"),
        unit_scale=stage.get("unit_scale", 1.0),
    )


@STAGE_REGISTRY.register("collect_json")
def run_collect_json(stage, context: StageContext) -> None:
    collect_json_metrics(
        stage["source"],
        stage["output"],
        stage["selectors"],
    )


@STAGE_REGISTRY.register("collect_npz")
def run_collect_npz(stage, context: StageContext) -> None:
    with np.load(stage["source"], allow_pickle=False) as data:
        metrics = {}
        for name, key in stage["selectors"].items():
            if key not in data:
                raise KeyError(f"{stage['source']} has no NPZ field {key!r}")
            value = data[key]
            if value.ndim != 0:
                raise ValueError(f"NPZ metric field {key!r} must be scalar")
            metrics[name] = value.item()
    write_json(
        Path(stage["output"]),
        {
            "kind": "imported_npz",
            "source": str(Path(stage["source"]).resolve()),
            "metrics": metrics,
        },
    )


@STAGE_REGISTRY.register("export_riggs")
def run_export_riggs(stage, context: StageContext) -> None:
    export_riggs_points(
        stage["frame_dir"],
        stage["output"],
        pattern=stage.get("pattern", "frame_*.ply"),
        max_points=stage.get("max_points"),
        seed=stage.get("seed", context.job.experiment["seed"]),
        fps=stage.get("fps", 30.0),
    )


@STAGE_REGISTRY.register("export_d3dgs")
def run_export_d3dgs(stage, context: StageContext) -> None:
    export_d3dgs_points(
        stage["params"],
        stage["output"],
        foreground_only=stage.get("foreground_only", True),
        max_points=stage.get("max_points"),
        seed=stage.get("seed", context.job.experiment["seed"]),
        fps=stage.get("fps", 30.0),
        frame_start=stage.get("frame_start", 0),
        frame_count=stage.get("frame_count"),
        frame_stride=stage.get("frame_stride", 1),
        high_motion_window=stage.get("high_motion_window"),
    )


@STAGE_REGISTRY.register("export_dt4d")
def run_export_dt4d(stage, context: StageContext) -> None:
    export_dt4d_points(
        stage["anime"],
        stage["output"],
        frame_stride=stage.get("frame_stride", 1),
        fps=stage.get("fps", 30.0),
    )


@STAGE_REGISTRY.register("export_dt4d_high_motion")
def run_export_dt4d_high_motion(stage, context: StageContext) -> None:
    export_dt4d_high_motion(
        stage["anime"],
        stage["output"],
        target_frames=stage.get("target_frames", 200),
        fps=stage.get("fps", 30.0),
    )


@STAGE_REGISTRY.register("export_vto")
def run_export_vto(stage, context: StageContext) -> None:
    export_vto_points(
        stage["pickle"],
        stage["output"],
        frame_stride=stage.get("frame_stride", 1),
        fps=stage.get("fps", 30.0),
        pelvis_offset=stage.get("pelvis_offset"),
    )


@STAGE_REGISTRY.register("slice_sequence")
def run_slice_sequence(stage, context: StageContext) -> None:
    slice_point_sequence(
        stage["input"],
        stage["output"],
        fraction=stage.get("fraction"),
        frame_count=stage.get("frame_count"),
    )


@STAGE_REGISTRY.register("skelebone")
def run_skelebone(stage, context: StageContext) -> None:
    extraction = extract_skelebones(
        stage["sequence"],
        stage["output_dir"],
        mesh=stage.get("mesh"),
        config=ExtractionConfig(**stage.get("config", {})),
    )
    if stage.get("result"):
        write_json(
            Path(stage["result"]),
            {
                "skelebones": str(extraction.bundle_path),
                "bvh": str(extraction.bvh_path),
                "manifest": str(extraction.manifest_path),
            },
        )
