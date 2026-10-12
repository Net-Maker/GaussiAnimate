"""Export the paper's SMPL conditioning motion to the portable pose schema."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml


SMPL_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.empty_like(angle)
    small = angle < 1e-8
    scale[small] = 0.5
    scale[~small] = np.sin(half[~small]) / angle[~small]
    quaternion = np.concatenate([np.cos(half), axis_angle * scale], axis=-1)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion


def load_smpl_skeleton(model_path: Path, betas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    template = np.asarray(model["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)[..., :10]
    shaped = template + shapedirs @ np.asarray(betas, dtype=np.float64)[:10]
    joints = np.asarray(model["J_regressor"] @ shaped)
    kintree = np.asarray(model["kintree_table"], dtype=np.int64)
    ids = {int(joint_id): index for index, joint_id in enumerate(kintree[1])}
    parents = np.full(24, -1, dtype=np.int64)
    for joint in range(1, 24):
        parents[joint] = ids[int(kintree[0, joint])]
    return joints, parents


def _case_record(resources: Path, case_id: str) -> Tuple[str, Dict]:
    config = yaml.safe_load(Path(resources).read_text(encoding="utf-8"))
    for dataset in config["datasets"]:
        for case in dataset["cases"]:
            if case["id"] == case_id:
                return dataset["id"], case
    raise KeyError(f"case is not declared in {resources}: {case_id}")


def load_dna_motion(
    case: Dict,
    annotation_root: Path,
    reader_root: Path,
    frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sys.path.insert(0, str(Path(reader_root)))
    from SMCReader import SMCReader

    reader = SMCReader(str(Path(annotation_root) / case["annotation"]))
    poses = []
    translations = []
    betas = None
    scale = 1.0
    for frame in range(frames):
        try:
            record = reader.get_SMPLx(Frame_id=frame)
        except (AssertionError, IndexError, KeyError, ValueError) as error:
            raise ValueError(
                f"{case['id']} contains fewer than the requested {frames} frames"
            ) from error
        if record is None:
            raise ValueError(
                f"{case['id']} contains fewer than the requested {frames} frames"
            )
        body = np.asarray(record["fullpose"], dtype=np.float64)[:22]
        poses.append(np.concatenate([body, np.zeros((2, 3))], axis=0))
        translations.append(np.asarray(record["transl"], dtype=np.float64))
        if betas is None:
            betas = np.asarray(record["betas"], dtype=np.float64)
            scale = float(record.get("scale", 1.0))
    if not poses:
        raise ValueError(f"no SMPL-X frames found for {case['id']}")
    return np.stack(poses), np.stack(translations), betas, scale


def load_actorshq_motion(
    case: Dict,
    smplx_root: Path,
    frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    directory = Path(smplx_root) / case["id"]
    poses = []
    translations = []
    betas = None
    for frame in range(case["start_frame"], case["start_frame"] + frames):
        path = directory / f"{frame:06d}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))[0]
        full_pose = np.asarray(record["poses"], dtype=np.float64).reshape(-1)
        root = np.asarray(record["Rh"], dtype=np.float64).reshape(3)
        body = full_pose[3:66].reshape(21, 3)
        poses.append(
            np.concatenate([root[None], body, np.zeros((2, 3))], axis=0)
        )
        translations.append(
            np.asarray(record["Th"], dtype=np.float64).reshape(3)
        )
        if betas is None:
            betas = np.asarray(record["shapes"], dtype=np.float64).reshape(-1)
    return np.stack(poses), np.stack(translations), betas, 1.0


def export_human_pose(
    case_id: str,
    output: Path,
    *,
    resources: Path,
    smpl_model: Path,
    dna_annotations: Path,
    dna_reader_root: Path,
    actorshq_smplx: Path,
    frames: int,
    fps: float = 30.0,
) -> Path:
    dataset, case = _case_record(resources, case_id)
    if dataset == "dna_rendering":
        poses, translations, betas, scale = load_dna_motion(
            case, dna_annotations, dna_reader_root, frames
        )
    elif dataset == "actorshq":
        poses, translations, betas, scale = load_actorshq_motion(
            case, actorshq_smplx, frames
        )
    else:
        raise ValueError(f"unsupported human dataset: {dataset}")
    if len(poses) != frames or len(translations) != frames:
        raise ValueError(
            f"{case_id}: expected {frames} pose frames, found {len(poses)}"
        )
    rest_joints, parents = load_smpl_skeleton(smpl_model, betas)
    rest_joints *= scale
    local_rotations = axis_angle_to_quaternion(poses)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rest_joints=rest_joints,
        parents=parents,
        names=np.asarray(SMPL_NAMES),
        betas=np.asarray(betas, dtype=np.float64)[:10],
        body_scale=np.asarray(scale, dtype=np.float64),
        axis_angle=poses,
        local_rotations=local_rotations,
        root_translation=translations,
        fps=np.asarray(fps),
        case=np.asarray(case_id),
        dataset=np.asarray(dataset),
        source_frames=np.asarray(len(poses)),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SMPL conditioning motion")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--smpl-model", type=Path, required=True)
    parser.add_argument("--dna-annotations", type=Path, required=True)
    parser.add_argument("--dna-reader-root", type=Path, required=True)
    parser.add_argument("--actorshq-smplx", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    print(
        export_human_pose(
            args.case,
            args.output,
            resources=args.resources,
            smpl_model=args.smpl_model,
            dna_annotations=args.dna_annotations,
            dna_reader_root=args.dna_reader_root,
            actorshq_smplx=args.actorshq_smplx,
            frames=args.frames,
            fps=args.fps,
        )
    )


if __name__ == "__main__":
    main()
