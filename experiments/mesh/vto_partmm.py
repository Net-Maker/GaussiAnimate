"""Run the research PartMM ablations from official VTO simulations and SMPL.

Adapted from the author's VTO rebuttal runner. The research matcher is bundled
in experiments.backends.partmm_research; data and SMPL models stay local.
"""

import contextlib
import io
import json
from pathlib import Path
import time

import numpy as np

from .prepare_partmm import load_simulation, prepare_bundle, _smpl_geometry
from .export_vto import canonicalize_root_motion


def run_ablation(output_dir, *, patch_size, vto_root, smpl_model_path,
                 expected_cases, n_train=5, device="cuda", fit_dir=None, garment="tshirt",
                 exclude="tshirt_shape00_104_17"):
    """Fit a source-only database and evaluate the fixed held-out queries."""
    import torch
    from partmm import load_input
    from experiments.backends.partmm_research import set_smpl_model_dir
    from experiments.backends.partmm_research.core import SMPLMotionData, SSDRMotionData
    from experiments.backends.partmm_research.core.part_matcher import PartBasedSMPLToSSDRMatcher
    from experiments.backends.partmm_research.reconstruction import (
        reconstruct_vertices, result_to_lbs_transforms,
    )

    cases = list(expected_cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("expected_cases must be nonempty and unique")
    if int(n_train) < 1 or int(patch_size) < 1:
        raise ValueError("n_train and patch_size must be positive")
    root = Path(vto_root) / garment / "simulations"
    excluded = set(cases) | {name for name in exclude.split(",") if name}
    training = [path.stem for path in sorted(root.glob(f"{garment}_shape00_*.pkl"))
                if path.stem not in excluded][:int(n_train)]
    if len(training) != n_train:
        raise ValueError(f"Need {n_train} source sequences, found {len(training)} in {root}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = Path(fit_dir) if fit_dir else output_dir / "source_fit"
    metadata_path = fit_dir / "preparation.json"
    if not metadata_path.is_file():
        prepare_bundle(fit_dir, vto_root=vto_root, smpl_model_path=smpl_model_path,
                       database_cases=training, query_case=cases[0], garment=garment)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source_cases") != training or metadata.get("fit_scope") != "source":
        raise ValueError(f"Source fit has a different training split: {fit_dir}")
    database = load_input(fit_dir / "partmm_input.npz")
    source_records = [load_simulation(root / (name + ".pkl")) for name in training]
    poses = np.concatenate([record["pose"] for record in source_records]).astype(np.float32)
    poses[:, :3] = 0
    if len(poses) != len(database.bone_transforms):
        raise ValueError("Source fit and official source frame counts differ")
    _, db_joints = _smpl_geometry(smpl_model_path, poses)
    model_path = Path(smpl_model_path).expanduser()
    set_smpl_model_dir(model_path.parent if model_path.is_file() else model_path)
    device = torch.device(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def motion(values):
        return SMPLMotionData({"poses": torch.from_numpy(values),
                               "trans": torch.zeros(len(values), 3)})

    reference = {"rest_pose": database.source_points,
                 "weights": database.skinning_weights,
                 "faces": source_records[0]["faces"]}
    sync()
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        source = motion(poses)
        transforms = database.bone_transforms.astype(np.float32)
        outer = SSDRMotionData({
            "transforms": torch.from_numpy(transforms.reshape(len(poses), -1)),
            "trans": torch.from_numpy(transforms[:, 0, :3, 3].copy()),
        })
        matcher = PartBasedSMPLToSSDRMatcher(
            patch_size=int(patch_size), alpha=0.85, matching_mode="full",
            device=device, skeleton_guided=True, skeleton_transform_weight=1.0,
            skeleton_mode="edge_wise", gender="female", is_cloth3d=False,
            use_motion_binding=True, motion_weight=0.5, distance_weight=0.5,
        )
        # The research binder uses bone-major weights; the portable bundle is point-major.
        matcher.build_part_binding(source, outer, database.source_points, database.skinning_weights.T)
        matcher.build_part_databases(source, outer)
    sync()
    build_s = time.perf_counter() - started
    rows = []
    for case in cases:
        record = load_simulation(root / (case + ".pkl"))
        if not np.array_equal(record["faces"], reference["faces"]):
            raise ValueError(f"{case}: garment topology differs from source")
        query_poses = record["pose"].astype(np.float32).copy()
        query_poses[:, :3] = 0
        _, joints = _smpl_geometry(smpl_model_path, query_poses)
        gt = canonicalize_root_motion(record["vertices"], record["pose"][:, :3],
                                      record["translation"], database.rest_joints[0])
        sync()
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = matcher.retarget_motion(
                motion(query_poses), num_steps=2, matching_alpha=0.3,
                coarse_ratio=1.0, pyr_factor=0.75,
                novel_joint_positions=torch.from_numpy(joints),
                db_joint_positions=torch.from_numpy(db_joints),
                novel_poses=query_poses, db_poses=poses,
            )
            lbs = result_to_lbs_transforms(result, reference)
            vertices = reconstruct_vertices(lbs)
        sync()
        predict_s = time.perf_counter() - started
        if vertices.shape != gt.shape or not np.isfinite(vertices).all():
            raise ValueError(f"{case}: invalid prediction shape or nonfinite values")
        rows.append({"seq_name": case, "variant": "default", "method": "partmm",
                     "patch_size": int(patch_size), "frames": len(gt),
                     "rmse_mm": float(np.linalg.norm(vertices - gt, axis=-1).mean() * 1000),
                     "build_s": build_s, "predict_s": predict_s,
                     "source_cases": training, "fit_scope": "source"})
        np.savez_compressed(output_dir / (case + ".npz"),
                            pred_vertices=vertices.astype(np.float32),
                            gt_vertices=gt.astype(np.float32), faces=reference["faces"])
    path = output_dir / "per_case_summary.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows
