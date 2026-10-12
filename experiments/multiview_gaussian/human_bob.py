"""Fit the paper's SMPL-conditioned Bag of Bones baseline on all Gaussians."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.methods import (
    HeldOutPointProtocol,
    point_error_metrics,
    save_prediction_diagnostic,
    write_prediction_artifacts,
)
from experiments.reproductions.dressrecon_bob.pose_contract import (
    BOB_ADAPTER_ID,
    BOB_EVALUATION_ATTRIBUTE_SOURCE,
    BOB_ORACLE_EVALUATION_ATTRIBUTES,
    BOB_PREDICTED_GAUSSIAN_ATTRIBUTES,
    SMPL_JOINT_MAPPING_ID,
    SMPL_ROOT_HANDLING,
    SMPL_ROOT_JOINT_ID,
    SMPL_ROOT_TRANSLATION_HANDLING,
    SMPL_ROOTLESS_JOINTS,
    SMPL_ROOTLESS_POSE_DIM,
    split_smpl_root_and_links,
)


def _source_state(root):
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    dependency_digest = hashlib.sha256()
    for path in sorted((root / "lab4d").rglob("*.py")):
        dependency_digest.update(str(path.relative_to(root)).encode("utf-8"))
        dependency_digest.update(path.read_bytes())
    vendored_root = Path(__file__).parents[1] / "reproductions" / "dressrecon_bob"
    vendored_digest = hashlib.sha256()
    for path in sorted(vendored_root.glob("*.py")):
        vendored_digest.update(path.name.encode("utf-8"))
        vendored_digest.update(path.read_bytes())
    return {
        "adapter_id": BOB_ADAPTER_ID,
        "vendored_adapter": str(vendored_root.resolve()),
        "vendored_python_sha256": vendored_digest.hexdigest(),
        "upstream_dependency_root": str(root.resolve()),
        "upstream_revision": revision,
        "upstream_dirty": dirty,
        "upstream_lab4d_python_sha256": dependency_digest.hexdigest(),
        "license": "MIT",
    }


def run_human_bob(
    params,
    motion,
    output_dir,
    *,
    dressrecon_root,
    train_ratio=0.8,
    num_bones=25,
    num_iterations=2000,
    learning_rate=1e-3,
    batch_frames=8,
    seed=42,
    device="cuda",
):
    import torch

    dressrecon_root = Path(dressrecon_root)
    sys.path.insert(0, str(dressrecon_root.resolve()))
    from experiments.reproductions.dressrecon_bob.fit_bob import BoBFitter

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BoB requested CUDA but no CUDA device is available")

    with np.load(params, allow_pickle=False) as archive:
        target = np.asarray(archive["means3D"], dtype=np.float32)
    protocol = HeldOutPointProtocol.from_shape(
        target.shape,
        train_ratio,
    )
    frames = protocol.frames
    points = protocol.points
    with np.load(motion, allow_pickle=False) as archive:
        axis_angle = np.asarray(archive["axis_angle"], dtype=np.float32)
        case = str(archive["case"]) if "case" in archive.files else None
        dataset = (
            str(archive["dataset"]) if "dataset" in archive.files else None
        )
    root_axis_angle, poses = split_smpl_root_and_links(
        axis_angle,
        expected_frames=frames,
    )
    training_frames = protocol.training_frames

    fitter = BoBFitter(
        canonical_pts=target[0],
        target_pts=target,
        smpl_poses=poses,
        num_bones=num_bones,
        pose_dim=poses.shape[1],
        init_gauss_scale=0.1,
        delta_skin=True,
        device=device,
        train_ratio=train_ratio,
        skel_type="smpl",
        skel_init_gauss_scale=0.03,
    )
    started = time.perf_counter()
    history = fitter.fit(
        num_iters=num_iterations,
        lr=learning_rate,
        batch_frames=batch_frames,
        reg_delta_skin=0.01,
    )
    elapsed = time.perf_counter() - started
    metrics = fitter.evaluate()
    prediction = fitter.get_warped_sequence().astype(np.float32)
    if prediction.shape != target.shape:
        raise ValueError(
            f"BoB returned {prediction.shape}, expected {target.shape}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_prediction_artifacts(
        output_dir,
        protocol,
        test_prediction=prediction[training_frames:],
        full_prediction=prediction,
    )
    fitter.save(output_dir / "bob_model.pt")
    np.save(output_dir / "loss_history.npy", np.asarray(history))
    np.save(
        output_dir / "skinning_weights.npy",
        fitter.get_skinning_weights(),
    )
    np.save(
        output_dir / "skeleton_skinning_weights.npy",
        fitter.get_skel_skinning_weights(),
    )
    save_prediction_diagnostic(
        output_dir / "diagnostic.png",
        target[0],
        prediction[training_frames],
        target[training_frames],
        seed=seed,
    )

    point_metrics = point_error_metrics(
        prediction[training_frames:],
        target[training_frames:],
    )
    artifacts.update(
        {
            "model": "bob_model.pt",
            "loss": "loss_history.npy",
            "skinning_weights": "skinning_weights.npy",
            "skeleton_skinning_weights": (
                "skeleton_skinning_weights.npy"
            ),
            "diagnostic": "diagnostic.png",
        }
    )
    metadata = {
        "schema_version": 1,
        "case": case,
        "dataset": dataset,
        "method": "smpl_conditioned_bag_of_bones",
        **protocol.metadata(),
        "metrics": {
            **point_metrics,
            "fitter_test_mean_error_mm": float(
                metrics["test_mean"] * 1000.0
            ),
        },
        "protocol": {
            "split": "contiguous_prefix_80_20",
            "driven_points": "all_gaussians",
            "opacity_filter": None,
            "point_subsampling": None,
            "adapter_id": BOB_ADAPTER_ID,
            "joint_mapping_id": SMPL_JOINT_MAPPING_ID,
            "pose_joint_range": [1, 24],
            "pose_dimensions": SMPL_ROOTLESS_POSE_DIM,
            "skeleton_links": SMPL_ROOTLESS_JOINTS,
            "root_joint_id": SMPL_ROOT_JOINT_ID,
            "root_pose_dimensions": int(root_axis_angle.shape[1]),
            "root_handling": SMPL_ROOT_HANDLING,
            "root_translation_handling": (
                SMPL_ROOT_TRANSLATION_HANDLING
            ),
            "predicted_gaussian_attributes": list(
                BOB_PREDICTED_GAUSSIAN_ATTRIBUTES
            ),
            "evaluation_attribute_source": (
                BOB_EVALUATION_ATTRIBUTE_SOURCE
            ),
            "oracle_evaluation_attributes": list(
                BOB_ORACLE_EVALUATION_ATTRIBUTES
            ),
            "num_bones": int(num_bones),
            "num_iterations": int(num_iterations),
            "learning_rate": float(learning_rate),
            "batch_frames": int(batch_frames),
            "delta_skin_regularization": 0.01,
            "seed": int(seed),
            "device": device,
            "elapsed_seconds": elapsed,
        },
        "implementation": _source_state(dressrecon_root),
        "artifacts": artifacts,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dressrecon-root", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-bones", type=int, default=25)
    parser.add_argument("--num-iterations", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_human_bob(
        args.params,
        args.motion,
        args.output_dir,
        dressrecon_root=args.dressrecon_root,
        train_ratio=args.train_ratio,
        num_bones=args.num_bones,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        batch_frames=args.batch_frames,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
