"""Portable, pickle-free NPZ contracts for standalone PartMM."""

from pathlib import Path

import numpy as np

from .types import PartMMInput, PartMMOutput, PartMMQuery


PARTMM_INPUT_SCHEMA_VERSION = 1
PARTMM_QUERY_SCHEMA_VERSION = 1
PARTMM_OUTPUT_SCHEMA_VERSION = 1


def _npz_path(path) -> Path:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("PartMM files must use the .npz extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_input(path, data: PartMMInput) -> Path:
    data = PartMMInput.from_bundle(data)
    path = _npz_path(path)
    np.savez_compressed(
        path,
        partmm_input_schema_version=np.asarray(PARTMM_INPUT_SCHEMA_VERSION),
        rest_joints=data.rest_joints,
        parents=data.parents,
        names=np.asarray(data.joint_names),
        local_rotations=data.source_local_rotations,
        root_translation=data.source_root_translation,
        bone_transforms=data.bone_transforms,
        source_points=data.source_points,
        weights=data.skinning_weights,
        fps=np.asarray(data.fps),
    )
    return path


def load_input(path) -> PartMMInput:
    """Load either a standalone PartMM input or `skelebones.npz`."""
    with np.load(path, allow_pickle=False) as arrays:
        return PartMMInput(
            rest_joints=arrays["rest_joints"],
            parents=arrays["parents"],
            joint_names=(arrays["names"].tolist() if "names" in arrays else None),
            source_local_rotations=arrays["local_rotations"],
            source_root_translation=arrays["root_translation"],
            bone_transforms=arrays["bone_transforms"],
            source_points=arrays["source_points"],
            skinning_weights=arrays["weights"],
            fps=float(arrays["fps"]) if "fps" in arrays else 30.0,
        )


def save_query(path, query: PartMMQuery) -> Path:
    query = PartMMQuery.from_pose(query)
    path = _npz_path(path)
    np.savez_compressed(
        path,
        partmm_query_schema_version=np.asarray(PARTMM_QUERY_SCHEMA_VERSION),
        local_rotations=query.local_rotations,
        root_translation=query.root_translation,
        fps=np.asarray(query.fps),
    )
    return path


def load_query(path) -> PartMMQuery:
    with np.load(path, allow_pickle=False) as arrays:
        return PartMMQuery(
            arrays["local_rotations"],
            arrays["root_translation"],
            float(arrays["fps"]) if "fps" in arrays else 30.0,
        )


def _pack_matches(matches):
    part_ids = np.asarray(sorted(matches), dtype=np.int64)
    if not len(part_ids):
        return part_ids, np.empty((0, 0, 0), dtype=np.int64)
    values = [np.asarray(matches[int(part)], dtype=np.int64) for part in part_ids]
    if len({value.shape for value in values}) != 1:
        raise ValueError("matched index arrays must share one shape")
    return part_ids, np.stack(values)


def save_output(path, output: PartMMOutput) -> Path:
    path = _npz_path(path)
    part_ids, matches = _pack_matches(output.matched_indices)
    np.savez_compressed(
        path,
        partmm_output_schema_version=np.asarray(PARTMM_OUTPUT_SCHEMA_VERSION),
        points=output.points,
        bone_transforms=output.bone_transforms,
        joint_positions=output.joint_positions,
        matched_part_ids=part_ids,
        matched_source_frames=matches,
        fps=np.asarray(output.fps),
    )
    return path


def load_output(path) -> PartMMOutput:
    with np.load(path, allow_pickle=False) as arrays:
        part_ids = arrays.get("matched_part_ids", np.empty(0, dtype=np.int64))
        packed = arrays.get(
            "matched_source_frames", np.empty((0, 0, 0), dtype=np.int64)
        )
        matches = {
            int(part): packed[index]
            for index, part in enumerate(part_ids)
        }
        return PartMMOutput(
            points=arrays["points"],
            bone_transforms=arrays["bone_transforms"],
            joint_positions=arrays["joint_positions"],
            matched_indices=matches,
            fps=float(arrays["fps"]),
        )


__all__ = [
    "load_input",
    "load_output",
    "load_query",
    "save_input",
    "save_output",
    "save_query",
]
