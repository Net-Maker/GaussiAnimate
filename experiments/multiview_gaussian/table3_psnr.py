"""Report training and held-out rendering PSNR for Table 3 methods."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments.scheduling import configure_visible_gpu
from experiments.engine.source import local_source_signatures
from experiments.engine.storage import write_json

from .oracle_eval import ACTORSHQ_IDS, DNA_IDS, sequence_spec


def load_ssdr_reconstruction(cache_path, params_shape):
    with np.load(cache_path, allow_pickle=False) as cache:
        reconstruction = cache["reconstruction_original"]
        training_frames = int(cache["training_frames"])
        total_frames = int(cache["total_frames"])
    if total_frames != params_shape[0] or training_frames != int(total_frames * 0.8):
        raise ValueError("SSDR cache does not match the 80/20 training split")
    if reconstruction.shape != (training_frames, *params_shape[1:]):
        raise ValueError("SSDR reconstruction does not match the Gaussian point set")
    if not np.isfinite(reconstruction).all():
        raise ValueError("SSDR reconstruction contains non-finite points")
    return reconstruction


def load_driven_trainset(prediction_dir, params_shape):
    prediction_dir = Path(prediction_dir)
    metadata = json.loads(
        (prediction_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if "ground_truth" in (metadata.get("warped_sequence_training_payload") or ""):
        raise ValueError("training payload contains ground truth, not predictions")
    training_frames = int(metadata["training_frames"])
    if training_frames != int(params_shape[0] * 0.8):
        raise ValueError("prediction does not match the 80/20 training split")
    predicted = np.load(prediction_dir / "warped_sequence.npy", mmap_mode="r")
    if predicted.shape != params_shape:
        raise ValueError("prediction does not match the Gaussian point set")
    point_indices = np.load(prediction_dir / "point_indices.npy")
    if not np.array_equal(point_indices, np.arange(params_shape[1])):
        raise ValueError("evaluation requires all Gaussians")
    return predicted[:training_frames]


def read_testset_psnr(
    evaluation_path, case, training_frames, total_frames, camera_ids,
    method_key="partmm_repaired",
):
    result = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    protocol = result["protocol"]
    if protocol["sequence"] != case:
        raise ValueError("test evaluation refers to another sequence")
    if protocol["attribute_source"] != "target_frame":
        raise ValueError("test evaluation uses different Gaussian attributes")
    if protocol["camera_ids"] != camera_ids:
        raise ValueError("train and test evaluations use different cameras")
    if protocol["test_frame_ids"] != list(range(training_frames, total_frames)):
        raise ValueError("test evaluation does not follow the training split")
    if protocol["method"] != method_key:
        raise ValueError("test evaluation uses a different method")
    psnr = float(result[method_key]["psnr"])
    if not np.isfinite(psnr):
        raise ValueError("test evaluation has a non-finite PSNR")
    return psnr


def evaluation_paths(case, root, data_dir, method):
    spec = sequence_spec(case)
    root = Path(root)
    params_path = root / "resources/multiview/d3dgs" / spec["dataset"] / case / "params.npz"
    run_root = root / "runs/multiview_novel_pose" / spec["dataset"] / case
    if method == "partmm":
        prediction_path = run_root / "smpl_partmm/ssdr/train_only.npz"
        test_path = root / "runs/partmm_repaired_evaluation" / f"{case}.json"
        method_key = "partmm_repaired"
    elif method == "lbs":
        prediction_path = run_root / "smpl_lbs/lbs"
        test_path = run_root / "smpl_lbs/evaluation" / f"{case}.json"
        method_key = "lbs"
    elif method == "bob":
        prediction_path = run_root / "smpl_bob/bob" / case
        test_path = run_root / "smpl_bob/evaluation" / f"{case}.json"
        method_key = "bob"
    else:
        raise ValueError(f"unknown method: {method}")
    metadata_path = Path(data_dir) / spec["data_subdir"] / "train_meta.json"
    return params_path, prediction_path, test_path, metadata_path, method_key


def _file_signature(path):
    path = Path(path).resolve()
    signature = {'path': str(path), 'exists': path.is_file()}
    if path.is_file():
        stat = path.stat()
        signature.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if path.suffix in {'.py', '.json'}:
            signature['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return signature


def evaluation_signature(case, root, d3dgs_root, data_dir, num_cameras, method):
    params, prediction, test, metadata, _ = evaluation_paths(case, root, data_dir, method)
    paths = [params, test, metadata]
    paths.extend([prediction] if method == 'partmm' else [
        prediction / 'warped_sequence.npy', prediction / 'point_indices.npy',
        prediction / 'metadata.json',
    ])
    paths.extend([Path(d3dgs_root) / 'eval_lbs_metrics.py', Path(d3dgs_root) / 'helpers.py'])
    paths.append(Path(__file__))
    return {
        'schema_version': 1, 'case': case, 'method': method,
        'num_cameras': num_cameras, 'data_dir': str(Path(data_dir).resolve()),
        'inputs': [_file_signature(path) for path in paths],
        'implementation': local_source_signatures(__file__),
    }


def _cached_row(path, case, method, signature):
    row = json.loads(path.read_text(encoding='utf-8'))
    if row.get('case') != case or row.get('method') != method:
        raise ValueError(f'{path.name}: case or method does not match')
    if not all(np.isfinite(row.get(key, np.nan)) for key in ('trainset_psnr', 'testset_psnr')):
        raise ValueError(f'{path.name}: non-finite PSNR')
    if row.get('cache_signature') != signature:
        raise ValueError(f'{path.name}: stale metrics; rerun without --summary-only or use --refresh')
    return row


def evaluate_trainset(case, root, d3dgs_root, data_dir, gpu, num_cameras,
                      method="partmm"):
    configure_visible_gpu(gpu)
    sys.path.insert(0, str(Path(d3dgs_root).resolve()))
    from eval_lbs_metrics import (  # pylint: disable=import-error,import-outside-toplevel
        compute_psnr, load_gt_image, load_seg_mask, render_gaussians,
        select_evenly_spaced_cameras,
    )
    from helpers import setup_camera  # pylint: disable=import-error,import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    torch.cuda.set_device(0)
    spec = sequence_spec(case)
    params_path, prediction_path, test_path, metadata_path, method_key = evaluation_paths(
        case, root, data_dir, method,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(params_path, allow_pickle=False) as params:
        means = params["means3D"]
        reconstruction = (
            load_ssdr_reconstruction(prediction_path, means.shape)
            if method == "partmm" else
            load_driven_trainset(prediction_path, means.shape)
        )
        camera_ids = select_evenly_spaced_cameras(
            len(metadata["fn"][0]), num_cameras
        )
        testset_psnr = read_testset_psnr(
            test_path, case, len(reconstruction), len(means), camera_ids,
            method_key=method_key,
        )
        colors = params["rgb_colors"]
        rotations = params["unnorm_rotations"]
        scales = params["log_scales"]
        opacities = params["logit_opacities"]
        cam_m = params["cam_m"] if "cam_m" in params.files else None
        cam_c = params["cam_c"] if "cam_c" in params.files else None
        values = []
        for frame, positions in enumerate(reconstruction):
            positions = np.array(positions, copy=True)
            for camera_id in camera_ids:
                ground_truth = load_gt_image(
                    data_dir, spec["data_subdir"], metadata, frame, camera_id
                )
                if ground_truth is None:
                    continue
                mask = load_seg_mask(
                    data_dir, spec["data_subdir"], metadata, frame, camera_id
                )
                target = (
                    ground_truth * mask.unsqueeze(0)
                    if mask is not None else ground_truth
                )
                camera = setup_camera(
                    metadata["w"], metadata["h"],
                    metadata["k"][frame][camera_id],
                    metadata["w2c"][frame][camera_id],
                    near=1.0, far=100,
                )
                with torch.no_grad():
                    image = render_gaussians(
                        positions, colors[frame], rotations[frame], scales,
                        opacities, camera, cam_m, cam_c, camera_id,
                    )
                values.append(compute_psnr(image, target))

    if not values:
        raise ValueError(f"no training images available for {case}")
    return {
        "case": case,
        "method": method,
        "trainset_psnr": float(np.mean(values)),
        "testset_psnr": testset_psnr,
        "protocol": {
            "train_prediction": str(prediction_path.resolve()),
            "test_prediction": method_key,
            "attribute_source": "target_frame",
            "camera_ids": camera_ids,
            "train_frames": len(reconstruction),
            "train_render_count": len(values),
            "train_mask": "ground truth segmentation",
            "test_evaluation": str(test_path.resolve()),
        },
    }


def summary_lines(rows, include_means=True):
    lines = [f"{'Case':<24} {'TrainSet PSNR':>12} {'Test Set PSNR':>14}"]
    for row in rows:
        lines.append(
            f"{row['case']:<24} {row['trainset_psnr']:>12.2f} "
            f"{row['testset_psnr']:>14.2f}"
        )
    if include_means and len(rows) > 1:
        for label, prefix in (("ActorsHQ mean", "actorshq_"), ("DNA-Rendering mean", "dna_")):
            subset = [row for row in rows if row["case"].startswith(prefix)]
            if subset:
                lines.append(
                    f"{label:<24} {np.mean([r['trainset_psnr'] for r in subset]):>12.2f} "
                    f"{np.mean([r['testset_psnr'] for r in subset]):>14.2f}"
                )
    return lines


def write_summary(output_dir, rows, method="partmm", expected_cases=None):
    summary = {"method": method, "cases": rows}
    if expected_cases is not None:
        ids = [row['case'] for row in rows]
        summary['complete'] = len(ids) == len(set(ids)) and set(ids) == set(expected_cases)
        if not summary['complete']:
            write_json(output_dir / 'summary.json', summary)
            return summary
    for label, prefix in (("actorshq", "actorshq_"), ("dna_rendering", "dna_")):
        subset = [row for row in rows if row["case"].startswith(prefix)]
        if not subset:
            continue
        summary[label] = {
            "trainset_psnr": float(np.mean([row["trainset_psnr"] for row in subset])),
            "testset_psnr": float(np.mean([row["testset_psnr"] for row in subset])),
        }
    write_json(output_dir / 'summary.json', summary)
    return summary


def combined_summary_lines(summaries):
    lines = [
        f"{'Method':<12} {'Dataset':<14} {'TrainSet PSNR':>12} {'Test Set PSNR':>14}"
    ]
    for method, label in (("lbs", "Robust LBS"), ("bob", "BoB"), ("partmm", "PartMM")):
        for dataset, dataset_label in (("actorshq", "ActorsHQ"),
                                       ("dna_rendering", "DNA-Rendering")):
            metrics = summaries[method][dataset]
            lines.append(
                f"{label:<12} {dataset_label:<14} "
                f"{metrics['trainset_psnr']:>12.2f} "
                f"{metrics['testset_psnr']:>14.2f}"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--method", choices=("partmm", "lbs", "bob", "all"), default="partmm")
    parser.add_argument("--d3dgs-root", type=Path, default=Path(os.environ.get(
        "GA_D3DGS_ROOT", "~/research/GA/Dynamic3DGaussians"
    )))
    parser.add_argument("--data-dir", type=Path, default=os.environ.get('GA_D3DGS_DATA'))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.num_cameras < 1:
        parser.error('--num-cameras must be positive')
    args.root = args.root.expanduser().resolve()
    args.d3dgs_root = args.d3dgs_root.expanduser().resolve()
    args.data_dir = (args.data_dir or args.d3dgs_root / 'data').expanduser().resolve()

    all_cases = sorted(ACTORSHQ_IDS) + sorted(DNA_IDS)
    cases = args.case or all_cases
    if len(cases) != len(set(cases)):
        parser.error('duplicate --case values')
    output_dir = args.root / "runs/table3_psnr"
    if args.method == "all":
        if args.case:
            parser.error("--method all reports dataset means; omit --case")
        if args.refresh:
            parser.error('--refresh requires a specific method')
        summaries = {}
        for method in ("lbs", "bob", "partmm"):
            directory = output_dir / (method if method != 'partmm' else '')
            rows = [_cached_row(
                directory / f'{case}.json', case, method,
                evaluation_signature(case, args.root, args.d3dgs_root, args.data_dir,
                                     args.num_cameras, method),
            ) for case in cases]
            summaries[method] = write_summary(directory, rows, method, cases)
        write_json(output_dir / 'all_methods_summary.json', summaries)
        print("\n".join(combined_summary_lines(summaries)))
        return
    if args.method != "partmm":
        output_dir /= args.method
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.summary_only and len(cases) > 1:
        print(summary_lines([])[0], flush=True)
    for case in cases:
        sequence_spec(case)
        output = output_dir / f"{case}.json"
        signature = evaluation_signature(case, args.root, args.d3dgs_root, args.data_dir,
                                         args.num_cameras, args.method)
        row = None
        if output.exists() and not args.refresh:
            try:
                row = _cached_row(output, case, args.method, signature)
            except ValueError:
                if args.summary_only:
                    raise
        if row is None:
            if args.summary_only:
                raise FileNotFoundError(output)
            row = evaluate_trainset(
                case, args.root, args.d3dgs_root, args.data_dir,
                args.gpu, args.num_cameras, args.method,
            )
            row['cache_signature'] = signature
            write_json(output, row)
        rows.append(row)
        if not args.summary_only and len(cases) > 1:
            print(summary_lines([row])[1], flush=True)

    write_summary(output_dir, rows, args.method, all_cases)
    if args.summary_only or len(cases) == 1:
        print("\n".join(summary_lines(rows, include_means=not bool(args.case))), flush=True)
    elif not args.case:
        print("\n".join(summary_lines(rows)[-2:]), flush=True)


if __name__ == "__main__":
    main()
