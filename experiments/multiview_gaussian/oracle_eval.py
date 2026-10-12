"""Evaluate D3DGS oracle renders without requiring a driven-method artifact."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.scheduling import configure_visible_gpu


DNA_IDS = {
    "dna_0012_11",
    "dna_0013_01",
    "dna_0013_03",
    "dna_0013_09",
    "dna_0013_11",
    "dna_0019_08",
    "dna_0021_03",
    "dna_0023_06",
}
ACTORSHQ_IDS = {
    "actorshq_actor01_seq1",
    "actorshq_actor01_seq2",
    "actorshq_actor02_seq1",
    "actorshq_actor02_seq2",
    "actorshq_actor04_seq2",
    "actorshq_actor05_seq1",
    "actorshq_actor06_seq1",
    "actorshq_actor06_seq2",
    "actorshq_actor07_seq1",
    "actorshq_actor08_seq1",
}


def sequence_spec(sequence):
    if sequence in DNA_IDS:
        return {
            "dataset": "dna_rendering",
            "params_subdir": sequence + "_1x",
            "data_subdir": sequence + "_1x",
        }
    if sequence in ACTORSHQ_IDS:
        return {
            "dataset": "actorshq",
            "params_subdir": sequence,
            "data_subdir": sequence,
        }
    raise ValueError(f"unknown multiview sequence: {sequence}")


def resolve_params(
    sequence,
    *,
    release_root,
    dna_results_dir,
    actorshq_results_dir,
):
    spec = sequence_spec(sequence)
    candidates = [
        Path(release_root)
        / spec["dataset"]
        / sequence
        / "params.npz",
        Path(
            dna_results_dir
            if spec["dataset"] == "dna_rendering"
            else actorshq_results_dir
        )
        / spec["params_subdir"]
        / "params.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "D3DGS params not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def evaluate_oracle(
    sequence,
    output,
    *,
    d3dgs_root,
    release_root,
    dna_results_dir,
    actorshq_results_dir,
    data_dir,
    train_ratio=0.8,
    num_cameras=4,
    gpu=0,
):
    configure_visible_gpu(gpu)
    sys.path.insert(0, str(Path(d3dgs_root).resolve()))
    from experiments.backends.d3dgs.eval_lbs_metrics import (
        compute_fg_psnr,
        compute_lpips,
        compute_psnr,
        compute_ssim,
        load_gt_image,
        load_seg_mask,
        render_gaussians,
        select_evenly_spaced_cameras,
    )
    from helpers import setup_camera  # pylint: disable=import-error,import-outside-toplevel
    import lpips  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    torch.cuda.set_device(0)
    spec = sequence_spec(sequence)
    params_path = resolve_params(
        sequence,
        release_root=release_root,
        dna_results_dir=dna_results_dir,
        actorshq_results_dir=actorshq_results_dir,
    )
    metadata_path = Path(data_dir) / spec["data_subdir"] / "train_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(params_path, allow_pickle=False) as params:
        means = params["means3D"]
        colors = params["rgb_colors"]
        rotations = params["unnorm_rotations"]
        scales = params["log_scales"]
        opacities = params["logit_opacities"]
        cam_m = params["cam_m"] if "cam_m" in params.files else None
        cam_c = params["cam_c"] if "cam_c" in params.files else None

        frames = len(means)
        first_test = int(frames * train_ratio)
        test_frames = range(first_test, frames)
        camera_ids = select_evenly_spaced_cameras(
            len(metadata["fn"][0]), num_cameras
        )
        metric_values = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "fg_psnr": [],
        }
        lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
        for frame in test_frames:
            for camera_id in camera_ids:
                camera = setup_camera(
                    metadata["w"],
                    metadata["h"],
                    metadata["k"][frame][camera_id],
                    metadata["w2c"][frame][camera_id],
                    near=1.0,
                    far=100,
                )
                gt = load_gt_image(
                    data_dir,
                    spec["data_subdir"],
                    metadata,
                    frame,
                    camera_id,
                )
                if gt is None:
                    continue
                mask = load_seg_mask(
                    data_dir,
                    spec["data_subdir"],
                    metadata,
                    frame,
                    camera_id,
                )
                target = gt * mask.unsqueeze(0) if mask is not None else gt
                with torch.no_grad():
                    image = render_gaussians(
                        means[frame],
                        colors[frame],
                        rotations[frame],
                        scales,
                        opacities,
                        camera,
                        cam_m,
                        cam_c,
                        camera_id,
                    )
                metric_values["psnr"].append(compute_psnr(image, target))
                metric_values["ssim"].append(compute_ssim(image, target))
                metric_values["lpips"].append(
                    compute_lpips(image, target, lpips_fn)
                )
                if mask is not None:
                    metric_values["fg_psnr"].append(
                        compute_fg_psnr(image, gt, mask)
                    )

    metrics = {
        name: float(np.mean(values)) if values else -1.0
        for name, values in metric_values.items()
    }
    result = {
        "oracle": metrics,
        "protocol": {
            "sequence": sequence,
            "params": str(params_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            "train_ratio": float(train_ratio),
            "test_frames": [first_test, frames - 1],
            "camera_ids": camera_ids,
            "render_count": len(metric_values["psnr"]),
            "lpips_backbone": "vgg",
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d3dgs-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--dna-results-dir", required=True)
    parser.add_argument("--actorshq-results-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_oracle(
        args.sequence,
        args.output,
        d3dgs_root=args.d3dgs_root,
        release_root=args.release_root,
        dna_results_dir=args.dna_results_dir,
        actorshq_results_dir=args.actorshq_results_dir,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        num_cameras=args.num_cameras,
        gpu=args.gpu,
    )
    print(json.dumps(result["oracle"], indent=2))


if __name__ == "__main__":
    main()
