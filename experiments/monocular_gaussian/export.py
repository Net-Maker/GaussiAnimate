"""Export registered mean trajectories from a stage-one monocular checkpoint."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np

from .riggs_loader import bundled_engine_root
from .ssdr_evaluate import _load_models


def export_tracks(model_path, output_dir, source_path=None, iteration=40000, frames=200):
    """Keep every Gaussian in checkpoint row order at uniformly sampled times."""
    import torch
    from plyfile import PlyData, PlyElement

    if frames < 2:
        raise ValueError("at least two frames are required")
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    if any(output_dir.glob("pointcloud_t*.ply")):
        raise ValueError("output already contains tracks; choose a new output directory")
    with redirect_stdout(io.StringIO()):
        dataset, models = _load_models(
            bundled_engine_root(), model_path, iteration, source_path=source_path,
        )
    gaussians, deform = models.gaussians, models.deform
    xyz = gaussians.get_xyz.detach()
    output_dir.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0., 1., frames, dtype=np.float32)
    with torch.inference_mode():
        for index, time in enumerate(times):
            fid = torch.tensor([float(time)], device=xyz.device, dtype=xyz.dtype)
            time_input = fid.unsqueeze(0).expand(len(xyz), -1) if deform.name == "mlp" else deform.deform.expand_time(fid)
            values = deform.step(
                xyz, time_input, feature=gaussians.feature,
                motion_mask=gaussians.motion_mask, is_training=False,
            )
            points = (xyz + values["d_xyz"]).cpu().numpy()
            if not np.isfinite(points).all():
                raise ValueError(f"nonfinite Gaussian means at frame {index}")
            vertices = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            for axis, name in enumerate(("x", "y", "z")):
                vertices[name] = points[:, axis]
            PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
                str(output_dir / f"pointcloud_t{index:03d}.ply")
            )
    loaded_iteration = int(models.scene.loaded_iter)
    manifest = {
        "model": str(model_path), "source_path": str(dataset.source_path),
        "source_ply": str(model_path / "point_cloud" / f"iteration_{loaded_iteration}" / "point_cloud.ply"),
        "iteration": loaded_iteration, "frames": frames, "points": len(xyz),
        "kind": deform.name, "frame_fids": times.tolist(),
        "use_isotropic_gs": bool(gaussians.use_isotropic_gs),
    }
    (output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--iteration", type=int, default=40000)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    manifest = export_tracks(**vars(args))
    print(f"Exported {manifest['frames']} frames, {manifest['points']} points: {args.output_dir}")


if __name__ == "__main__":
    main()
