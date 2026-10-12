"""End-to-end extraction of the GaussiAnimate skelebone representation."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Optional

import numpy as np

from .bvh import export_bvh
from .extraction import (
    CurveExtractionConfig,
    JointConfig,
    MeshConfig,
    SSDRConfig,
    TrackingConfig,
    build_kinematic_tree,
    decompose_ssdr,
    detect_joints,
    extract_curve_skeleton,
    prepare_mesh,
    project_skinning_to_curve,
    track_skeleton,
    transfer_skinning_to_mesh,
)
from .io import save_bundle
from .kinematics import forward_kinematics
from .types import DynamicPointSequence, SkeleboneBundle


@dataclass(frozen=True)
class ExtractionConfig:
    """All numerical controls used by the self-contained extraction pipeline."""

    num_bones: int = 50
    rest_frame: int = 0
    ssdr_tolerance: float = 1e-3
    ssdr_patience: int = 3
    ssdr_max_iterations: int = 100
    ssdr_max_influences: int = 4
    ssdr_weight_smoothness: float = 0.0
    ssdr_bind_update: int = 2
    ssdr_protocol: str = "motion_pointcloud"
    ssdr_require_num_bones: bool = True
    mesh_frame: Optional[int] = None
    mesh_poisson_depth: int = 9
    mesh_normal_radius: Optional[float] = None
    mesh_normal_neighbors: int = 30
    mesh_normal_consistency_neighbors: int = 15
    mesh_density_quantile: float = 0.01
    mesh_minimum_component_triangles: int = 10
    mesh_target_triangles: int = 20000
    curve_skeleton_bin: Optional[str] = None
    curve_quality: float = 0.1
    curve_medial: float = 0.25
    curve_timeout_seconds: float = 300.0
    curve_correspondence_neighbors: int = 3
    skinning_transfer_neighbors: int = 1
    skinning_align_bbox: bool = False
    joint_score_threshold: float = 0.3
    joint_merge_distance: float = 0.08
    joint_peak_window_hops: int = 8
    joint_max_count: Optional[int] = None
    joint_track_neighbors: int = 20

    def __post_init__(self) -> None:
        if self.mesh_frame is not None and self.mesh_frame != self.rest_frame:
            raise ValueError(
                f"mesh_frame {self.mesh_frame} does not match rest_frame "
                f"{self.rest_frame}: the canonical mesh must correspond to "
                "the rest frame used for skinning and tracking"
            )


@dataclass
class ExtractionResult:
    bundle: SkeleboneBundle
    bundle_path: Path
    bvh_path: Path
    manifest_path: Path


class SkelebonePipeline:
    """Map a registered point trajectory to outer and inner skeletal motion."""

    def __init__(self, config: ExtractionConfig = ExtractionConfig()):
        self.config = config

    def _component_configs(self):
        config = self.config
        ssdr = SSDRConfig(
            num_bones=config.num_bones,
            rest_frame=config.rest_frame,
            tolerance=config.ssdr_tolerance,
            patience=config.ssdr_patience,
            max_iterations=config.ssdr_max_iterations,
            max_influences=config.ssdr_max_influences,
            weight_smoothness=config.ssdr_weight_smoothness,
            bind_update=config.ssdr_bind_update,
            protocol=config.ssdr_protocol,
            require_num_bones=config.ssdr_require_num_bones,
        )
        mesh_frame = config.rest_frame if config.mesh_frame is None else config.mesh_frame
        mesh = MeshConfig(
            frame=mesh_frame,
            poisson_depth=config.mesh_poisson_depth,
            normal_radius=config.mesh_normal_radius,
            normal_neighbors=config.mesh_normal_neighbors,
            normal_consistency_neighbors=config.mesh_normal_consistency_neighbors,
            density_quantile=config.mesh_density_quantile,
            minimum_component_triangles=config.mesh_minimum_component_triangles,
            target_triangles=config.mesh_target_triangles,
        )
        curve_skeleton_bin = config.curve_skeleton_bin
        if not curve_skeleton_bin:
            repository_native = (
                Path(__file__).resolve().parents[1]
                / ".native"
                / "bin"
                / "curve_skeleton"
            )
            curve_skeleton_bin = (
                os.environ.get("GAUSSIANIMATE_CURVE_SKELETON_BIN")
                or os.environ.get("GA_CURVE_SKELETON_BIN")
                or shutil.which("curve_skeleton")
                or (str(repository_native) if repository_native.is_file() else None)
            )
        if not curve_skeleton_bin:
            raise ValueError(
                "curve_skeleton_bin is required; pass --curve-skeleton-bin, set "
                "GAUSSIANIMATE_CURVE_SKELETON_BIN, or add curve_skeleton to PATH"
            )
        curve = CurveExtractionConfig(
            executable=curve_skeleton_bin,
            quality=config.curve_quality,
            medial=config.curve_medial,
            timeout_seconds=config.curve_timeout_seconds,
            correspondence_neighbors=config.curve_correspondence_neighbors,
        )
        joints = JointConfig(
            score_threshold=config.joint_score_threshold,
            minimum_distance=config.joint_merge_distance,
            peak_window_hops=config.joint_peak_window_hops,
            max_joints=config.joint_max_count,
        )
        tracking = TrackingConfig(
            rest_frame=config.rest_frame,
            neighbors=config.joint_track_neighbors,
        )
        if config.skinning_transfer_neighbors < 1:
            raise ValueError("skinning_transfer_neighbors must be positive")
        return ssdr, mesh, curve, joints, tracking

    def validate(self) -> None:
        """Validate every component without executing geometry or optimization."""
        self._component_configs()

    def run(
        self,
        sequence: DynamicPointSequence,
        output_dir,
        mesh_for_skeleton=None,
        ssdr_backend=None,
    ) -> ExtractionResult:
        if not isinstance(sequence, DynamicPointSequence):
            sequence = DynamicPointSequence(sequence)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ssdr_config, mesh_config, curve_config, joint_config, tracking_config = (
            self._component_configs()
        )

        ssdr = decompose_ssdr(sequence, ssdr_config, backend=ssdr_backend)
        ssdr_dir = output_dir / "ssdr"
        ssdr_dir.mkdir(exist_ok=True)
        ssdr_path = ssdr_dir / "result.npz"
        np.savez_compressed(
            ssdr_path,
            weights=ssdr.weights,
            transformations=ssdr.transformations,
            bone_positions=ssdr.bone_positions,
            reconstruction=ssdr.reconstruction,
            rest_pose=ssdr.rest_pose,
            rmse=np.asarray(ssdr.rmse),
            protocol=np.asarray(ssdr.protocol),
            requested_bones=np.asarray(ssdr.requested_bones, dtype=np.int64),
            initialized_bones=np.asarray(ssdr.initialized_bones, dtype=np.int64),
            final_bones=np.asarray(ssdr.num_bones, dtype=np.int64),
            effective_bones=np.asarray(ssdr.effective_bones, dtype=np.int64),
            hard_bone_population=ssdr.hard_bone_population,
            soft_bone_mass=ssdr.soft_bone_mass,
            backend_name=np.asarray(ssdr.backend_name),
            backend_module=np.asarray(ssdr.backend_module),
            backend_path=np.asarray(ssdr.backend_path or ""),
            backend_sha256=np.asarray(ssdr.backend_sha256 or ""),
        )

        mesh = prepare_mesh(sequence, mesh_config, mesh_for_skeleton)
        if mesh_for_skeleton is not None:
            surface_source = "external_mesh"
        elif (
            sequence.faces is not None
            and len(mesh.vertices) == sequence.num_points
            and np.array_equal(mesh.faces, sequence.faces)
        ):
            surface_source = "input_faces"
        else:
            surface_source = "poisson"
        curve_dir = output_dir / "curve_skeleton"
        curve = extract_curve_skeleton(mesh, curve_dir, curve_config)
        mesh_weights = transfer_skinning_to_mesh(
            mesh.vertices,
            ssdr.rest_pose,
            ssdr.weights,
            k=self.config.skinning_transfer_neighbors,
            align_bbox=self.config.skinning_align_bbox,
        )
        curve_weights = project_skinning_to_curve(
            mesh_weights, curve.correspondence, curve.skeleton
        )
        skinning_dir = output_dir / "skinning_curve_skeleton"
        skinning_dir.mkdir(exist_ok=True)
        skinning_path = skinning_dir / "skinning.npz"
        np.savez_compressed(
            skinning_path,
            curve_points=curve.skeleton.points,
            curve_edges=curve.skeleton.edges,
            curve_weights=curve_weights,
            mesh_weights=mesh_weights,
        )

        detected = detect_joints(curve.skeleton, curve_weights, joint_config)
        tree = build_kinematic_tree(curve.skeleton, detected)
        tracked = track_skeleton(sequence, tree.skeleton, tracking_config)
        global_transforms, fk_positions = forward_kinematics(
            tree.skeleton, tracked.motion
        )
        tree_dir = output_dir / "kinematic_tree"
        tree_dir.mkdir(exist_ok=True)
        tree_path = tree_dir / "tree.npz"
        np.savez_compressed(
            tree_path,
            rest_joints=tree.skeleton.rest_joints,
            parents=tree.skeleton.parents,
            names=np.asarray(tree.skeleton.names),
            edges=tree.edges,
            node_types=np.asarray(tree.node_types),
            curve_indices=tree.curve_indices,
            joint_curve_indices=detected.curve_indices,
            joint_scores=detected.scores,
            all_joint_scores=detected.all_scores,
        )
        (tree_dir / "curve_paths.json").write_text(
            json.dumps([list(path) for path in tree.curve_paths], indent=2) + "\n",
            encoding="utf-8",
        )
        trajectory_path = tree_dir / "joint_trajectories.npz"
        np.savez_compressed(
            trajectory_path,
            tracked_joint_positions=tracked.trajectories.positions,
            fk_joint_positions=fk_positions,
            joint_global_transforms=global_transforms,
            local_rotations=tracked.motion.local_rotations,
            root_translation=tracked.motion.root_translation,
            rest_positions=tracked.trajectories.rest_positions,
            neighbor_indices=tracked.trajectories.neighbor_indices,
            fps=np.asarray(tracked.trajectories.fps),
        )

        relative = lambda path: str(Path(path).relative_to(output_dir))
        bundle = SkeleboneBundle(
            skeleton=tree.skeleton,
            weights=ssdr.weights,
            source_points=ssdr.rest_pose,
            motion=tracked.motion,
            bone_transforms=ssdr.transformations,
            artifacts={
                "ssdr": relative(ssdr_path),
                "mesh": relative(curve.mesh_path),
                "curve_skeleton": relative(curve_dir),
                "skinning_curve_skeleton": relative(skinning_path),
                "kinematic_tree": relative(tree_path),
                "joint_trajectories": relative(trajectory_path),
                "temporal_fk": relative(trajectory_path),
            },
        )
        bundle_path = save_bundle(output_dir / "skelebones.npz", bundle)
        bvh_path = export_bvh(output_dir / "motion.bvh", tree.skeleton, tracked.motion)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input": {
                        "frames": sequence.num_frames,
                        "points_per_frame": sequence.num_points,
                        "fps": float(sequence.fps),
                        "faces": 0 if sequence.faces is None else int(len(sequence.faces)),
                    },
                    "representation": {
                        "inner_joints": bundle.skeleton.num_joints,
                        "outer_bones": bundle.num_bones,
                    },
                    "ssdr": {
                        "protocol": ssdr.protocol,
                        "requested_bones": ssdr.requested_bones,
                        "initialized_bones": ssdr.initialized_bones,
                        "final_bones": ssdr.num_bones,
                        "effective_bones": ssdr.effective_bones,
                        "hard_bone_population": ssdr.hard_bone_population.tolist(),
                        "soft_bone_mass": ssdr.soft_bone_mass.tolist(),
                        "rmse": ssdr.rmse,
                        "backend": ssdr.diagnostics["backend"],
                    },
                    "surface": {
                        "source": surface_source,
                        "vertices": int(len(mesh.vertices)),
                        "triangles": int(len(mesh.faces)),
                    },
                    "main_outputs": {
                        "skelebones": relative(bundle_path),
                        "bvh": relative(bvh_path),
                    },
                    "artifacts": bundle.artifacts,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ExtractionResult(bundle, bundle_path, bvh_path, manifest_path)
