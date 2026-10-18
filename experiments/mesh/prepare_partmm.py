"""Prepare the standalone T-shirt example from trusted official VTO/SMPL files.

VTO: https://github.com/isantesteban/vto-dataset
SMPL: https://smpl.is.tue.mpg.de/
The coordinate convention follows the author's VTO experiments; neither
simulation data nor body-model parameters are distributed with this module.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np

from partmm import PartMMInput, PartMMQuery, save_input, save_query
from partmm.examples.quickstart import (
    DEFAULT_DATABASE_CASES, DEFAULT_QUERY_CASE, SMPL_JOINT_NAMES,
    SMPL_PARENTS, axis_angle_to_quaternion,
)
from .export_vto import canonicalize_root_motion


def load_simulation(path):
    """Read a trusted VTO pickle and validate its registered mesh sequence."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing VTO simulation: {path}. Obtain it from "
            "https://github.com/isantesteban/vto-dataset"
        )
    with path.open("rb") as handle:
        record = pickle.load(handle, encoding="latin1")
    required = {"vertices", "faces", "pose", "translation", "shape"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError(f"{path.name}: expected VTO fields {sorted(required)}")
    data = {key: np.asarray(record[key]) for key in required}
    points = data["vertices"]
    if points.ndim != 3 or points.shape[2] != 3 or min(points.shape[:2]) < 1:
        raise ValueError(f"{path.name}: vertices must be [frames, points, 3]")
    frames, vertices = points.shape[:2]
    for key, shape in (("pose", (frames, 72)), ("translation", (frames, 3)),
                       ("shape", (frames, 10))):
        if data[key].shape != shape:
            raise ValueError(f"{path.name}: {key} must have shape {shape}")
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError(f"{path.name}: arrays must contain finite numbers")
    faces = data["faces"]
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or not np.equal(faces, np.round(faces)).all()
            or faces.min() < 0 or faces.max() >= vertices):
        raise ValueError(f"{path.name}: invalid garment faces")
    if not np.allclose(data["shape"], 0, atol=1e-7):
        raise ValueError(f"{path.name}: this shape00 example requires zero SMPL betas")
    data["faces"] = faces.astype(np.int64)
    return data


def _smpl_geometry(model_path, query_poses):
    import torch
    from .smpl import load_smpl_model

    model = load_smpl_model(model_path, gender="female", batch_size=1)
    with torch.no_grad():
        rest = model().joints[0, :24].cpu().numpy()
        joints = []
        for start in range(0, len(query_poses), 64):
            pose = torch.as_tensor(query_poses[start:start + 64, 3:], dtype=torch.float32)
            count = len(pose)
            output = model(body_pose=pose, global_orient=torch.zeros(count, 3),
                           transl=torch.zeros(count, 3), betas=torch.zeros(count, 10))
            joints.append(output.joints[:, :24].cpu().numpy())
    return rest, np.concatenate(joints)


def _fit_ssdr(points, faces, fps, config):
    from skelebone.types import DynamicPointSequence
    from skelebone.extraction.ssdr import decompose_ssdr

    return decompose_ssdr(DynamicPointSequence(points, faces=faces, fps=fps), config)


def prepare_bundle(output, *, vto_root, smpl_model_path,
                   database_cases=DEFAULT_DATABASE_CASES, query_case=DEFAULT_QUERY_CASE,
                   num_bones=20, iterations=100, seed=42, fps=30.0, garment="tshirt"):
    """Fit source-only SSDR and export self-contained animation/evaluation inputs."""
    cases = tuple(database_cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("database cases must be nonempty and unique")
    if query_case in cases:
        raise ValueError("The query must not be included in the source SSDR fit")
    if garment not in {"tshirt", "dress"}:
        raise ValueError("garment must be tshirt or dress")
    if not all(name.startswith(f"{garment}_shape00_") and Path(name).name == name
               for name in cases + (query_case,)):
        raise ValueError(f"Use {garment}_shape00 sequence identifiers, without paths")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    from skelebone.extraction.ssdr import SSDRConfig

    config = SSDRConfig(num_bones=num_bones, max_iterations=iterations,
                        random_seed=seed, initialization="motion")
    root = Path(vto_root).expanduser() / garment / "simulations"
    sources = [load_simulation(root / (name + ".pkl")) for name in cases]
    query_data = load_simulation(root / (query_case + ".pkl"))
    faces = sources[0]["faces"]
    count = sources[0]["vertices"].shape[1]
    for name, data in zip(cases + (query_case,), sources + [query_data]):
        if data["vertices"].shape[1] != count or not np.array_equal(data["faces"], faces):
            raise ValueError(f"{name}: garment topology differs from the source")
    rest_joints, query_joints = _smpl_geometry(smpl_model_path, query_data["pose"])
    pelvis = rest_joints[0]

    def canonical(data):
        return canonicalize_root_motion(data["vertices"], data["pose"][:, :3],
                                        data["translation"], pelvis)

    points = np.concatenate([canonical(data) for data in sources])
    fit = _fit_ssdr(points, faces, fps, config)
    database = PartMMInput(
        rest_joints=rest_joints, parents=SMPL_PARENTS, joint_names=SMPL_JOINT_NAMES,
        source_local_rotations=axis_angle_to_quaternion(
            np.concatenate([data["pose"] for data in sources])),
        source_root_translation=np.zeros((len(points), 3)),
        bone_transforms=fit.transformations, source_points=fit.rest_pose,
        skinning_weights=fit.weights, fps=fps,
    )
    query = PartMMQuery(axis_angle_to_quaternion(query_data["pose"]),
                        np.zeros((len(query_data["pose"]), 3)), fps)
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "input": save_input(output / "partmm_input.npz", database),
        "query": save_query(output / "partmm_query.npz", query),
        "reference": output / "partmm_reference.npz",
        "preparation": output / "preparation.json",
    }
    np.savez_compressed(paths["reference"], faces=faces,
                        gt_vertices=canonical(query_data).astype(np.float32),
                        skeleton_joints=query_joints.astype(np.float32),
                        case=np.asarray(query_case), fps=np.asarray(fps))
    offsets = np.cumsum([0] + [len(data["pose"]) for data in sources])
    metadata = {
        "schema_version": 1, "fit_scope": "source", "source_cases": list(cases),
        "source_frame_ranges": [
            {"case": name, "raw_frames": [0, len(data["pose"])],
             "database_frames": [int(offsets[i]), int(offsets[i + 1])],
             "url": "https://github.com/isantesteban/vto-dataset/blob/main/"
                    f"{garment}/simulations/{name}.pkl"}
            for i, (name, data) in enumerate(zip(cases, sources))],
        "frame_range_convention": "zero-based, end exclusive, no subsampling",
        "query_case": query_case, "query_frames": len(query_data["pose"]),
        "query_url": "https://github.com/isantesteban/vto-dataset/blob/main/"
                     f"{garment}/simulations/{query_case}.pkl",
        "units": "meters", "fps": fps,
        "body_model": {"url": "https://smpl.is.tue.mpg.de/", "gender": "female",
                       "betas": "zero", "rest_joints": "zero pose",
                       "pelvis_offset": pelvis.tolist()},
        "ssdr": {"config": asdict(config), "actual_bones": database.num_bones,
                 "rmse_m": float(fit.rmse)},
    }
    paths["preparation"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--vto-root", required=True)
    parser.add_argument("--smpl-model-path", required=True)
    parser.add_argument("--database-cases", nargs="+", default=DEFAULT_DATABASE_CASES)
    parser.add_argument("--query-case", default=DEFAULT_QUERY_CASE)
    parser.add_argument("--num-bones", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    prepare_bundle(**vars(args))
    print(args.output)


if __name__ == "__main__":
    main()
