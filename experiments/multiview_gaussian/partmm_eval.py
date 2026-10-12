"""Render and evaluate a full human-Gaussian prediction on the paper protocol."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from experiments.scheduling import configure_visible_gpu

from .oracle_eval import resolve_params, sequence_spec


def evaluate_partmm(
    sequence,
    prediction_dir,
    output,
    *,
    d3dgs_root,
    release_root,
    dna_results_dir,
    actorshq_results_dir,
    data_dir,
    num_cameras=4,
    gpu=0,
    method_key="partmm",
    attribute_source="target_frame",
):
    configure_visible_gpu(gpu)
    sys.path.insert(0, str(Path(d3dgs_root).resolve()))
    from eval_lbs_metrics import (  # pylint: disable=import-error,import-outside-toplevel
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
    prediction_dir = Path(prediction_dir)
    prediction_metadata = json.loads(
        (prediction_dir / "metadata.json").read_text(encoding="utf-8")
    )
    test_frame_ids = [int(value) for value in prediction_metadata["test_frame_ids"]]
    predicted = np.load(
        prediction_dir / "warped_sequence.npy",
        mmap_mode="r",
    )
    point_indices = np.load(prediction_dir / "point_indices.npy")
    metadata_path = Path(data_dir) / spec["data_subdir"] / "train_meta.json"
    camera_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metric_values = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "fg_psnr": [],
    }
    with np.load(params_path, allow_pickle=False) as params:
        means = params["means3D"]
        colors = params["rgb_colors"]
        rotations = params["unnorm_rotations"]
        scales = params["log_scales"]
        opacities = params["logit_opacities"]
        cam_m = params["cam_m"] if "cam_m" in params.files else None
        cam_c = params["cam_c"] if "cam_c" in params.files else None
        if predicted.shape != means.shape:
            raise ValueError(
                f"prediction has shape {predicted.shape}, expected {means.shape}"
            )
        if not np.array_equal(point_indices, np.arange(means.shape[1])):
            raise ValueError("evaluation requires all Gaussians")
        expected_test = list(
            range(int(prediction_metadata["training_frames"]), len(means))
        )
        if test_frame_ids != expected_test:
            raise ValueError("metadata does not declare the contiguous 80/20 split")

        camera_ids = select_evenly_spaced_cameras(
            len(camera_metadata["fn"][0]), num_cameras
        )
        lpips_fn = lpips.LPIPS(net="vgg").cuda().eval()
        for frame in test_frame_ids:
            for camera_id in camera_ids:
                camera = setup_camera(
                    camera_metadata["w"],
                    camera_metadata["h"],
                    camera_metadata["k"][frame][camera_id],
                    camera_metadata["w2c"][frame][camera_id],
                    near=1.0,
                    far=100,
                )
                ground_truth = load_gt_image(
                    data_dir,
                    spec["data_subdir"],
                    camera_metadata,
                    frame,
                    camera_id,
                )
                if ground_truth is None:
                    continue
                mask = load_seg_mask(
                    data_dir,
                    spec["data_subdir"],
                    camera_metadata,
                    frame,
                    camera_id,
                )
                target = (
                    ground_truth * mask.unsqueeze(0)
                    if mask is not None
                    else ground_truth
                )
                with torch.no_grad():
                    color_frame = frame if attribute_source == "target_frame" else 0
                    rotation_frame = (
                        frame if attribute_source == "target_frame" else 0
                    )
                    image = render_gaussians(
                        predicted[frame],
                        colors[color_frame],
                        rotations[rotation_frame],
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
                        compute_fg_psnr(image, ground_truth, mask)
                    )

        squared_sum = 0.0
        distance_sum = 0.0
        distance_count = 0
        for frame in test_frame_ids:
            distance = np.linalg.norm(predicted[frame] - means[frame], axis=-1)
            squared_sum += float(np.sum(distance * distance, dtype=np.float64))
            distance_sum += float(np.sum(distance, dtype=np.float64))
            distance_count += distance.size

    result = {
        method_key: {
            name: float(np.mean(values)) if values else -1.0
            for name, values in metric_values.items()
        },
        f"{method_key}_vertex_rmse_mm": float(
            np.sqrt(squared_sum / distance_count) * 1000.0
        ),
        f"{method_key}_vertex_mean_error_mm": float(
            distance_sum / distance_count * 1000.0
        ),
        "protocol": {
            "sequence": sequence,
            "params": str(params_path.resolve()),
            "prediction": str(prediction_dir.resolve()),
            "metadata": str(metadata_path.resolve()),
            "split": "contiguous_prefix_80_20",
            "test_frame_ids": test_frame_ids,
            "camera_ids": camera_ids,
            "render_count": len(metric_values["psnr"]),
            "lpips_backbone": "vgg",
            "method": method_key,
            "attribute_source": attribute_source,
            "alignment_mode": prediction_metadata.get("partmm", {}).get(
                "alignment_mode"
            ),
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--d3dgs-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--dna-results-dir", required=True)
    parser.add_argument("--actorshq-results-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--method-key", default="partmm")
    parser.add_argument(
        "--attribute-source",
        choices=("target_frame", "canonical_frame"),
        default="target_frame",
    )
    args = parser.parse_args()
    result = evaluate_partmm(
        args.sequence,
        args.prediction_dir,
        args.output,
        d3dgs_root=args.d3dgs_root,
        release_root=args.release_root,
        dna_results_dir=args.dna_results_dir,
        actorshq_results_dir=args.actorshq_results_dir,
        data_dir=args.data_dir,
        num_cameras=args.num_cameras,
        gpu=args.gpu,
        method_key=args.method_key,
        attribute_source=args.attribute_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
