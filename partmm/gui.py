"""Minimal scientific skeleton-sequence editor built with Dear PyGui."""

import argparse
from pathlib import Path

import numpy as np

from .bvh import export_bvh
from .io import load_input, save_query
from .kinematics import forward_kinematics, matrix_to_quaternion, quaternion_to_matrix
from .types import PartMMInput, PartMMQuery


def _euler_xyz_to_matrix(degrees: np.ndarray) -> np.ndarray:
    x, y, z = np.deg2rad(degrees)
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    sy = np.hypot(matrix[0, 0], matrix[1, 0])
    if sy > 1e-8:
        angles = [np.arctan2(matrix[2, 1], matrix[2, 2]), np.arctan2(-matrix[2, 0], sy),
                  np.arctan2(matrix[1, 0], matrix[0, 0])]
    else:
        angles = [np.arctan2(-matrix[1, 2], matrix[1, 1]), np.arctan2(-matrix[2, 0], sy), 0]
    return np.rad2deg(angles)


class PoseEditorState:
    """Pure state model, separated from GUI callbacks for reproducible edits."""

    def __init__(self, bundle: PartMMInput):
        self.bundle = PartMMInput.from_bundle(bundle)
        source = self.bundle.motion
        self.pose = PartMMQuery(
            source.local_rotations.copy(),
            source.root_translation.copy(),
            source.fps,
        )
        self.frame = 0
        self.joint = 0

    def euler_degrees(self) -> np.ndarray:
        matrix = quaternion_to_matrix(self.pose.local_rotations[self.frame, self.joint])
        return _matrix_to_euler_xyz(matrix)

    def set_euler_degrees(self, degrees) -> None:
        matrix = _euler_xyz_to_matrix(np.asarray(degrees, dtype=np.float64))
        self.pose.local_rotations[self.frame, self.joint] = matrix_to_quaternion(matrix)

    def set_root_translation(self, translation) -> None:
        self.pose.root_translation[self.frame] = np.asarray(translation, dtype=np.float64)

    def joint_positions(self) -> np.ndarray:
        return forward_kinematics(self.bundle.skeleton, self.pose)[1][self.frame]

    def save(self, pose_path, bvh_path=None) -> None:
        save_query(pose_path, self.pose)
        if bvh_path is not None:
            export_bvh(bvh_path, self.bundle.skeleton, self.pose)


def build_editor_ui(dpg, state: PoseEditorState, output_pose):
    """Build all editor widgets and return the control synchronization callback."""

    output_pose = Path(output_pose)
    view_axes = {"Front": (0, 1), "Side": (2, 1), "Top": (0, 2)}

    def redraw() -> None:
        dpg.delete_item("skeleton_canvas", children_only=True)
        points = state.joint_positions()
        axes = view_axes[dpg.get_value("view")]
        projected = points[:, axes]
        extent = np.ptp(projected, axis=0)
        scale = 620.0 / max(float(extent.max()), 1e-6)
        center = (projected.min(0) + projected.max(0)) * 0.5
        screen = (projected - center) * scale + np.array([360, 360])
        screen[:, 1] = 720 - screen[:, 1]
        for child, parent in enumerate(state.bundle.skeleton.parents):
            if parent >= 0:
                start = tuple(float(value) for value in screen[int(parent)])
                end = tuple(float(value) for value in screen[child])
                dpg.draw_line(start, end, color=(72, 82, 91, 255), thickness=2,
                              parent="skeleton_canvas")
        for joint, point in enumerate(screen):
            color = (0, 122, 140, 255) if joint == state.joint else (210, 95, 62, 255)
            radius = 7 if joint == state.joint else 4
            center = tuple(float(value) for value in point)
            dpg.draw_circle(center, radius, color=color, fill=color, parent="skeleton_canvas")

    def sync_controls() -> None:
        dpg.set_value("joint", state.bundle.skeleton.names[state.joint])
        dpg.set_value("rotation", state.euler_degrees().tolist())
        dpg.set_value("translation", state.pose.root_translation[state.frame].tolist())
        redraw()

    def on_frame(_sender, value) -> None:
        state.frame = int(value)
        sync_controls()

    def on_joint(_sender, value) -> None:
        state.joint = state.bundle.skeleton.names.index(value)
        sync_controls()

    def on_rotation(_sender, value) -> None:
        state.set_euler_degrees(value[:3])
        redraw()

    def on_translation(_sender, value) -> None:
        state.set_root_translation(value[:3])
        redraw()

    def on_save(*_) -> None:
        state.save(output_pose, output_pose.with_suffix(".bvh"))
        dpg.set_value("status", f"Saved {output_pose}")

    with dpg.window(label="PartMM Pose Editor", tag="main"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=760, height=780, border=False):
                dpg.add_drawlist(width=720, height=720, tag="skeleton_canvas")
                dpg.add_radio_button(list(view_axes), default_value="Front", horizontal=True, tag="view",
                                     callback=lambda *_: redraw())
            with dpg.child_window(width=360, height=780, border=False):
                dpg.add_text("Pose")
                dpg.add_slider_int(label="Frame", min_value=0, max_value=state.pose.num_frames - 1,
                                   tag="frame", callback=on_frame)
                dpg.add_combo(state.bundle.skeleton.names, label="Joint", default_value=state.bundle.skeleton.names[0],
                              tag="joint", callback=on_joint)
                dpg.add_input_floatx(label="Rotation XYZ", size=3, tag="rotation", callback=on_rotation)
                dpg.add_input_floatx(label="Root translation", size=3, tag="translation", callback=on_translation)
                dpg.add_button(label="Save pose + BVH", callback=on_save)
                dpg.add_text("", tag="status", wrap=320)

    return sync_controls


def launch(bundle_path, output_pose="edited_pose.npz", headless_smoke_test=False) -> None:
    import dearpygui.dearpygui as dpg

    state = PoseEditorState(load_input(bundle_path))
    dpg.create_context()
    try:
        sync_controls = build_editor_ui(dpg, state, output_pose)
        sync_controls()
        if headless_smoke_test:
            required_items = ("main", "skeleton_canvas", "frame", "joint", "rotation", "translation")
            missing = [tag for tag in required_items if not dpg.does_item_exist(tag)]
            if missing:
                raise RuntimeError(f"pose editor failed to create widgets: {missing}")
            dpg.get_item_callback("rotation")("rotation", dpg.get_value("rotation"))
            dpg.get_item_callback("translation")("translation", dpg.get_value("translation"))
            return

        dpg.create_viewport(title="PartMM Pose Editor", width=1160, height=820)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        dpg.start_dearpygui()
    finally:
        dpg.destroy_context()


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit a skelebone pose sequence")
    parser.add_argument("bundle")
    parser.add_argument("--output", default="edited_pose.npz")
    parser.add_argument(
        "--headless-smoke-test",
        action="store_true",
        help="build and exercise all widgets without opening a viewport",
    )
    args = parser.parse_args()
    launch(args.bundle, args.output, args.headless_smoke_test)


if __name__ == "__main__":
    main()
