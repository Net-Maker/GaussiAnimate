"""Robust SMPL-to-cloth LBS baseline used by the VTO paper experiment."""

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

from experiments.mesh.export_vto import canonicalize_root_motion
from experiments.mesh.robust_weights import transfer_skin_weights


def _load_vto_sequence(path, pelvis_offset):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    source_poses = np.asarray(data["pose"], dtype=np.float32)
    vertices = canonicalize_root_motion(
        np.asarray(data["vertices"], dtype=np.float64),
        source_poses[:, :3],
        np.asarray(data["translation"], dtype=np.float64),
        pelvis_offset,
    ).astype(np.float32)
    poses = source_poses.copy()
    poses[:, :3] = 0.0
    return {
        "poses": poses,
        "betas": np.asarray(data["shape"][0], dtype=np.float32),
        "vertices": vertices,
        "faces": np.asarray(data["faces"], dtype=np.int32),
    }


def _compute_body_vertices_and_transforms(model, poses, betas):
    import torch
    from smplx.lbs import (
        batch_rigid_transform,
        batch_rodrigues,
        blend_shapes,
        vertices2joints,
    )

    pose_tensor = torch.from_numpy(np.asarray(poses, dtype=np.float32))
    betas_tensor = torch.from_numpy(np.asarray(betas, dtype=np.float32)).unsqueeze(0)
    batch_size = pose_tensor.shape[0]
    shaped = model.v_template.unsqueeze(0) + blend_shapes(
        betas_tensor, model.shapedirs
    )
    joints = vertices2joints(model.J_regressor, shaped).expand(batch_size, -1, -1)
    rotations = batch_rodrigues(pose_tensor.reshape(-1, 3)).reshape(
        batch_size, -1, 3, 3
    )
    _, transforms = batch_rigid_transform(rotations, joints, model.parents)
    with torch.no_grad():
        body = model(
            betas=betas_tensor.expand(batch_size, -1),
            body_pose=pose_tensor[:, 3:],
            global_orient=pose_tensor[:, :3],
            transl=torch.zeros(batch_size, 3, dtype=torch.float32),
        )
    return (
        body.vertices.detach().cpu().numpy().astype(np.float32),
        transforms.detach().cpu().numpy().astype(np.float32),
        model.faces.astype(np.int32),
        model.lbs_weights.detach().cpu().numpy().astype(np.float32),
    )


def _deform_cloth_with_lbs(rest_vertices, weights, transforms):
    base_inverse = np.linalg.inv(transforms[0])
    relative = transforms @ base_inverse[None, ...]
    blended = np.einsum(
        "vj,tjk->tvk", weights, relative.reshape(len(relative), 24, 16)
    ).reshape(len(relative), len(rest_vertices), 4, 4)
    rest_h = np.concatenate(
        [
            np.asarray(rest_vertices, dtype=np.float32),
            np.ones((len(rest_vertices), 1), dtype=np.float32),
        ],
        axis=1,
    )
    return np.einsum("tvab,vb->tva", blended, rest_h)[:, :, :3].astype(np.float32)


def _pelvis_offset(model_path):
    import torch
    from experiments.mesh.smpl import load_smpl_model

    model = load_smpl_model(model_path, gender="female", batch_size=1)
    with torch.no_grad():
        output = model(
            body_pose=torch.zeros(1, 69),
            global_orient=torch.zeros(1, 3),
            transl=torch.zeros(1, 3),
        )
    return output.joints[0, 0].detach().cpu().numpy(), load_smpl_model


def select_vto_paths(paths, *, test_start, exclude, expected_cases=None):
    """Apply the historical exclusion and frozen evaluation-case protocol."""

    excluded = {item.strip() for item in exclude.split(",") if item.strip()}
    paths = [path for path in sorted(paths) if Path(path).stem not in excluded]
    if not paths:
        raise ValueError("no VTO sequences remain after exclusions")
    if expected_cases is None:
        test_paths = paths[int(test_start):]
    else:
        by_name = {Path(path).stem: path for path in paths}
        expected_cases = list(expected_cases)
        missing = [name for name in expected_cases if name not in by_name]
        if missing:
            raise ValueError(
                "frozen VTO evaluation cases are missing: "
                + ", ".join(missing)
            )
        test_paths = [by_name[name] for name in expected_cases]
    if not test_paths:
        raise ValueError("no held-out VTO sequences selected")
    return paths, test_paths


def evaluate_vto_lbs(
    garment,
    output,
    *,
    hood_root=None,
    vto_root,
    smpl_model_path,
    transfer_python=None,
    test_start=50,
    exclude="tshirt_shape00_104_17",
    expected_cases=None,
    result_dir=None,
):
    """Transfer SMPL weights and evaluate LBS on the held-out VTO motions."""

    pattern = Path(vto_root) / garment / "simulations" / f"{garment}_shape00_*.pkl"
    paths, test_paths = select_vto_paths(
        glob.glob(str(pattern)),
        test_start=test_start,
        exclude=exclude,
        expected_cases=expected_cases,
    )

    pelvis_offset, smpl_class = _pelvis_offset(smpl_model_path)
    bind = _load_vto_sequence(paths[0], pelvis_offset)
    bind_pose = bind["poses"][:1]
    bind_model = smpl_class(str(smpl_model_path), gender="female", batch_size=1)
    bind_body, bind_transforms, body_faces, body_weights = (
        _compute_body_vertices_and_transforms(
            bind_model,
            bind_pose,
            bind["betas"],
        )
    )
    rest_cloth = bind["vertices"][0]
    faces = bind["faces"]
    cloth_weights, matched, vids_to_smooth = transfer_skin_weights(
        bind_body[0],
        body_faces,
        body_weights,
        rest_cloth,
        faces,
        search_radius_scale=0.05,
        normal_threshold_degrees=30.0,
        smooth_steps=10,
        smooth_alpha=0.2,
    )
    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            result_dir / "transferred_weights.npz",
            cloth_weights=cloth_weights,
            matched=matched,
            vids_to_smooth=vids_to_smooth,
            binding_sequence=Path(paths[0]).stem,
            garment=garment,
        )

    rows = []
    for path in test_paths:
        query = _load_vto_sequence(path, pelvis_offset)
        poses = query["poses"]
        gt = query["vertices"]
        model = smpl_class(
            str(smpl_model_path),
            gender="female",
            batch_size=len(poses),
        )
        _, query_transforms, _, _ = (
            _compute_body_vertices_and_transforms(
                model,
                poses,
                query["betas"],
            )
        )
        transforms = np.concatenate([bind_transforms, query_transforms], axis=0)
        prediction = _deform_cloth_with_lbs(
            rest_cloth,
            cloth_weights,
            transforms,
        )
        prediction = prediction[1:]
        frames = min(len(prediction), len(gt))
        distance = np.linalg.norm(prediction[:frames] - gt[:frames], axis=-1)
        if result_dir is not None:
            np.savez_compressed(
                result_dir / f"{Path(path).stem}.npz",
                prediction=prediction[:frames],
                sequence=Path(path).stem,
                garment=garment,
            )
        rows.append(
            {
                "sequence": Path(path).stem,
                "frames": frames,
                "mean_error_mm": float(distance.mean() * 1000.0),
                "euclidean_rmse_mm": float(
                    np.sqrt(np.mean(np.square(distance))) * 1000.0
                ),
            }
        )

    mean_errors = [row["mean_error_mm"] for row in rows]
    euclidean_rmses = [row["euclidean_rmse_mm"] for row in rows]
    frame_counts = [row["frames"] for row in rows]
    result = {
        "kind": "geometry",
        "garment": garment,
        "binding_sequence": Path(paths[0]).stem,
        "test_start": int(test_start),
        "sequences": len(rows),
        "protocol": {
            "excluded_sequence": exclude or None,
            "evaluation_cases": [Path(path).stem for path in test_paths],
            "paper_metric": "mean_per_vertex_euclidean_error",
            "weight_transfer": "closest_surface_biharmonic_inpainting_smoothing",
            "implementation": "experiments.mesh.robust_weights",
            "reference_revision": "a60263c17f38625b98a62bea09c91f61f7b865f0",
        },
        "metrics": {
            "rmse_mm": float(np.mean(mean_errors)),
            "frame_weighted_rmse_mm": float(
                np.average(mean_errors, weights=frame_counts)
            ),
            "mean_error_mm": float(np.mean(mean_errors)),
            "frame_weighted_mean_error_mm": float(
                np.average(mean_errors, weights=frame_counts)
            ),
            "euclidean_rmse_mm": float(np.mean(euclidean_rmses)),
            "frame_weighted_euclidean_rmse_mm": float(
                np.average(euclidean_rmses, weights=frame_counts)
            ),
        },
        "per_sequence": rows,
    }
    if result_dir is not None:
        result["artifacts"] = {
            "directory": str(result_dir.resolve()),
            "weights": "transferred_weights.npz",
            "predictions": [f"{row['sequence']}.npz" for row in rows],
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate robust LBS on VTO")
    parser.add_argument("--garment", choices=["tshirt", "dress"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hood-root", help=argparse.SUPPRESS)
    parser.add_argument("--vto-root", required=True)
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--transfer-python", help=argparse.SUPPRESS)
    parser.add_argument("--test-start", type=int, default=50)
    parser.add_argument("--exclude", default="tshirt_shape00_104_17")
    parser.add_argument("--result-dir")
    parser.add_argument(
        "--expected-cases",
        help="Comma-separated frozen evaluation sequence list",
    )
    args = parser.parse_args()
    evaluate_vto_lbs(
        args.garment,
        args.output,
        hood_root=args.hood_root,
        vto_root=args.vto_root,
        smpl_model_path=args.smpl_model_path,
        transfer_python=args.transfer_python,
        test_start=args.test_start,
        exclude=args.exclude,
        expected_cases=(
            args.expected_cases.split(",") if args.expected_cases else None
        ),
        result_dir=args.result_dir,
    )
    print(args.output)


if __name__ == "__main__":
    main()
