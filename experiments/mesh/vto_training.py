"""Public VTO Table 4 runner, adapted from the author's precomputation script."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from .prepare_partmm import load_simulation
from .export_vto import canonicalize_root_motion
from .vto_partmm import run_ablation


def run_suite(output_dir, *, garment, vto_root, smpl_model_path, expected_cases,
              n_train=5, device="cuda", steps=3000, seed=42,
              exclude="tshirt_shape00_104_17"):
    import torch
    from partmm import load_input
    from experiments.backends.partmm_research.core import SMPLMotionData, SSDRMotionData
    from experiments.backends.partmm_research.core.matcher import SMPLToSSDRMatcher
    from experiments.backends.partmm_research.reconstruction import (
        reconstruct_vertices, result_to_lbs_transforms,
    )
    from .vto_neural import train_models, predict_transforms

    output_dir = Path(output_dir)
    rows = run_ablation(output_dir, patch_size=7, vto_root=vto_root,
                        smpl_model_path=smpl_model_path, expected_cases=expected_cases,
                        n_train=n_train, device=device, garment=garment, exclude=exclude)
    source_cases = rows[0]["source_cases"]
    database = load_input(output_dir / "source_fit" / "partmm_input.npz")
    root = Path(vto_root) / garment / "simulations"
    records = [load_simulation(root / (case + ".pkl")) for case in source_cases]
    frame_counts = [len(record["pose"]) for record in records]
    poses = np.concatenate([record["pose"] for record in records]).astype(np.float32)
    poses[:, :3] = 0
    device = torch.device(device)
    models, stats = train_models(poses, database.bone_transforms, frame_counts,
                                device=device, steps=steps, seed=seed)
    torch.save({"models": {name: model.state_dict() for name, model in models.items()},
                "normalization": stats, "steps": steps, "seed": seed,
                "source_cases": source_cases}, output_dir / "models.pt")
    reference = {"rest_pose": database.source_points, "weights": database.skinning_weights,
                 "faces": records[0]["faces"]}

    def motion(values):
        return SMPLMotionData({"poses": torch.from_numpy(values),
                               "trans": torch.zeros(len(values), 3)})

    with contextlib.redirect_stdout(io.StringIO()):
        source_smpl, source_ssdr = [], []
        start = 0
        for count in frame_counts:
            source_smpl.append(motion(poses[start:start + count]))
            transforms = database.bone_transforms[start:start + count].astype(np.float32)
            source_ssdr.append(SSDRMotionData({
                "transforms": torch.from_numpy(transforms.reshape(count, -1)),
                "trans": torch.zeros(count, 3),
            }))
            start += count
        matcher = SMPLToSSDRMatcher(patch_size=7, alpha=0.85, device=device)
        matcher.ssdr_rest_pose_vertices = database.source_points
        matcher.ssdr_skinning_weights = database.skinning_weights.T
        matcher.build_skeletal_binding(source_smpl, source_ssdr,
                                       database.source_points, database.skinning_weights.T)
        # FullMM does not use the legacy adaptive mask; these are the same patches.
        matcher.build_patch_database(source_smpl, source_ssdr)
    summary = []
    for row in rows:
        case = row["seq_name"]
        record = load_simulation(root / (case + ".pkl"))
        query_poses = record["pose"].astype(np.float32).copy()
        query_poses[:, :3] = 0
        gt = canonicalize_root_motion(record["vertices"], record["pose"][:, :3],
                                      record["translation"], database.rest_joints[0])
        result = {"seq_name": case, "frames": len(gt), "PartMM_rmse": row["rmse_mm"],
                  "source_cases": source_cases, "fit_scope": "source", "seed": seed,
                  "training_steps": steps}
        with contextlib.redirect_stdout(io.StringIO()):
            full, _, _ = matcher.motion_blending(
                motion(query_poses).sample().to(device), num_steps=2,
                matching_alpha=0.3, verbose=False, init_mode="skeleton", use_smpl_mask=True)
            predictions = {"fullmm": reconstruct_vertices(result_to_lbs_transforms(full, reference))}
            for method, model in models.items():
                transforms = predict_transforms(model, query_poses, stats, method, device)
                predictions[method] = reconstruct_vertices({**reference, "transformations": transforms})
        for method, vertices in predictions.items():
            if vertices.shape != gt.shape or not np.isfinite(vertices).all():
                raise ValueError(f"{case}/{method}: invalid prediction")
            key = {"fullmm": "FullMM_rmse", "mlp": "MLP_rmse", "gru": "GRU_rmse"}[method]
            result[key] = float(np.linalg.norm(vertices - gt, axis=-1).mean() * 1000)
            np.savez_compressed(output_dir / f"{case}_{method}.npz",
                                pred_vertices=vertices.astype(np.float32),
                                gt_vertices=gt.astype(np.float32), faces=reference["faces"])
        summary.append(result)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
