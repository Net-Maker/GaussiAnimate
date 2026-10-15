"""Static-asset pose editor with independent reference motion playback."""

import argparse
from collections import OrderedDict
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlparse

import numpy as np

from partmm import PartMMConfig, PartMMQuery, PartMotionMatcher
from partmm.types import PartMMOutput
from partmm.gui import _euler_xyz_to_matrix
from ..io import load_bundle, load_point_sequence
from ..kinematics import forward_kinematics, matrix_to_quaternion, quaternion_to_matrix
from .support import pose_support_angles


class AnimationScene:
    def __init__(self, name, bundle_path, sequence_path, max_match_angle=15.0):
        self.name = name
        if not np.isfinite(max_match_angle) or not 0 <= max_match_angle <= 180:
            raise ValueError("max_match_angle must be finite degrees in [0,180]")
        self.max_match_angle = float(max_match_angle)
        self.bundle = load_bundle(bundle_path)
        sequence = load_point_sequence(sequence_path)
        if self.bundle.motion is None:
            raise ValueError("the bundle has no source motion")
        if sequence.points.shape[:2] != (self.bundle.motion.num_frames, len(self.bundle.source_points)):
            raise ValueError("input sequence and bundle dimensions do not match")
        self.input = np.asarray(sequence.points, dtype=np.float32)
        self.source_joints = forward_kinematics(self.bundle.skeleton, self.bundle.motion)[1].astype(np.float32)
        self.attributes = dict(sequence.point_attributes)
        self.attributes.update(self.bundle.point_attributes)
        # A pose has no temporal neighborhood; keep the full source pose library.
        self.config = PartMMConfig(num_parts=min(5, self.bundle.skeleton.num_joints),
                                   patch_size=1, pyramid_levels=1)
        self.matcher = PartMotionMatcher(self.bundle, self.config)
        source = self.bundle.motion
        self.query = PartMMQuery(source.local_rotations[:1].copy(), source.root_translation[:1].copy(), source.fps)
        self.result = self.matcher.animate(self.query)
        self.base_points = np.asarray(self.result.points, dtype=np.float32).copy()
        self.preview_weights = np.zeros((len(self.bundle.source_points), self.bundle.skeleton.num_joints))
        for bone, joint in enumerate(self.matcher.bone_to_edge_parent):
            self.preview_weights[:, joint] += self.bundle.weights[:, bone]
        self.preview_weights /= self.preview_weights.sum(axis=1, keepdims=True)
        self.inverse_bind = np.linalg.inv(forward_kinematics(self.bundle.skeleton, self.query)[0])
        self.mode = "auto"
        self.driver = "partmm"
        self.drive_reason = "supported"
        self.match_angle_degrees = 0.0
        self.revision = 0
        self.joint_offsets = {}
        self.lock = threading.RLock()
        self._buffers = {}
        self._refresh_buffers()

    def _refresh_buffers(self):
        self._buffers = {
            "input": self.input,
            "driven": np.asarray(self.result.points, dtype=np.float32),
            "input_joints": self.source_joints,
            "driven_joints": np.asarray(self.result.joint_positions, dtype=np.float32),
            "base": self.base_points,
        }

    def buffer_urls(self):
        return {key: f"/api/scenes/{self.name}/buffers/{key}.bin?revision={self.revision}" for key in self._buffers}

    def metadata(self):
        with self.lock:
            parents = self.bundle.skeleton.parents
            editable = [joint for joint in range(len(parents)) if parents[joint] >= 0 and np.any(parents == joint)]
            colors = self.attributes.get("colors", np.tile([.22, .46, .49], (self.input.shape[1], 1)))
            palette = np.array([[.12, .48, .52], [.39, .58, .36], [.69, .38, .35], [.48, .45, .63], [.65, .59, .30], [.35, .51, .67]])
            skinning = self.bundle.weights @ palette[np.arange(self.bundle.num_bones) % len(palette)]
            arrays = (self.input, self._buffers["driven"], self.source_joints, self._buffers["driven_joints"])
            lower = np.min([array.min(axis=(0, 1)) for array in arrays], axis=0)
            upper = np.max([array.max(axis=(0, 1)) for array in arrays], axis=0)
            return {
                "id": self.name, "label": self.name, "frames": len(self.input),
                "points": self.input.shape[1], "joints": len(parents), "fps": self.bundle.motion.fps,
                "parents": parents.tolist(), "names": list(self.bundle.skeleton.names),
                "bounds": {"min": lower.tolist(), "max": upper.tolist()},
                "colors": np.clip(colors, 0., 1.).ravel().tolist(),
                "skinning_colors": skinning.ravel().tolist(),
                "default_joint": editable[0] if editable else int(self.bundle.skeleton.roots[0]),
                "buffers": self.buffer_urls(), "revision": self.revision,
                "joint_offsets": self.joint_offsets,
                **self.drive_metadata(),
                "rig": {
                    "rest_joints": self.bundle.skeleton.rest_joints.tolist(),
                    "local_rotations": self.bundle.motion.local_rotations[0].tolist(),
                    "root_translation": self.bundle.motion.root_translation[0].tolist(),
                    "preview_weights": self.preview_weights.ravel().tolist(),
                },
            }

    def drive_metadata(self):
        return {
            "mode": self.mode, "driver": self.driver, "drive_reason": self.drive_reason,
            "match_angle_degrees": self.match_angle_degrees,
            "max_match_angle_degrees": self.max_match_angle,
            "lbs_weight_source": "aggregated_ssdr_weights",
        }

    def lbs(self, query):
        """Apply the browser's exact inner-rig skinning to the fixed asset."""
        transforms, joints = forward_kinematics(self.bundle.skeleton, query)
        relative = transforms @ self.inverse_bind
        source = np.column_stack((self.base_points[0], np.ones(len(self.base_points[0]))))
        points = np.einsum("nb,tbij,nj->tni", self.preview_weights, relative, source,
                           optimize=True)[..., :3]
        return PartMMOutput(points, relative, joints, {}, query.fps)

    def _validate_rotation(self, joint, degrees):
        rotation = np.asarray(degrees, dtype=float)
        if rotation.shape != (3,) or not np.isfinite(rotation).all() or np.any(np.abs(rotation) > 180):
            raise ValueError("rotation must contain three finite angles in [-180,180]")
        if not isinstance(joint, int) or not 0 <= joint < self.bundle.skeleton.num_joints:
            raise ValueError("joint index is outside the skeleton")
        return rotation.tolist()

    def drive(self, joint, degrees, reset=False):
        rotation = self._validate_rotation(joint, degrees)
        with self.lock:
            offsets = {} if reset else dict(self.joint_offsets)
            if not reset:
                offsets[joint] = rotation
            return self.pose(offsets)

    def pose(self, offsets, mode=None):
        """Replace the complete static pose and return its result atomically."""
        if not isinstance(offsets, dict):
            raise ValueError("offsets must map joint indices to three angles")
        validated = {}
        for key, angles in offsets.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError("joint index must be an integer")
            joint = int(key)
            validated[joint] = self._validate_rotation(joint, angles)
        with self.lock:
            mode = self.mode if mode is None else mode
            if mode not in {"auto", "lbs"}:
                raise ValueError("mode must be auto or lbs")
            start = time.perf_counter()
            source = self.bundle.motion
            local = source.local_rotations[:1].copy()
            for index, angles in validated.items():
                local[:, index] = matrix_to_quaternion(
                    quaternion_to_matrix(local[:, index]) @ _euler_xyz_to_matrix(angles)
                )
            query = PartMMQuery(local, source.root_translation[:1].copy(), source.fps)
            angles = pose_support_angles(source.local_rotations, query.local_rotations,
                                         self.matcher.part_masks)
            match_angle = float(angles.max())
            driver, reason = "lbs", "manual"
            if mode == "auto" and match_angle <= self.max_match_angle + 1e-6:
                try:
                    result = self.matcher.animate(query)
                    if not np.all(np.isfinite(result.points)):
                        raise ValueError("matching produced nonfinite points")
                    driver, reason = "partmm", "supported"
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    reason = "matching_failed"
            elif mode == "auto":
                reason = "out_of_support"
            if driver == "lbs":
                result = self.lbs(query)
            self.query = query
            self.result = result
            self.mode, self.driver, self.drive_reason = mode, driver, reason
            self.match_angle_degrees = match_angle
            self.joint_offsets = validated
            self.revision += 1
            self._refresh_buffers()
            return {"buffers": self.buffer_urls(), "revision": self.revision,
                    "joint_offsets": validated, "points": result.points.ravel().tolist(),
                    "joint_positions": result.joint_positions.ravel().tolist(),
                    **self.drive_metadata(),
                    "elapsed_ms": (time.perf_counter() - start) * 1000}

    def export(self):
        with self.lock:
            data = io.BytesIO()
            arrays = {
                "points": self.result.points, "bone_transforms": self.result.bone_transforms,
                "joint_positions": self.result.joint_positions,
                "local_rotations": self.query.local_rotations,
                "root_translation": self.query.root_translation, "fps": np.asarray(self.query.fps),
                "source_points": self.base_points[0] if self.driver == "lbs" else self.bundle.source_points,
                "skinning_weights": self.preview_weights if self.driver == "lbs" else self.matcher.bundle.weights,
                "rig_type": np.asarray("inner_lbs" if self.driver == "lbs" else "outer_ssdr"),
                "driver": np.asarray(self.driver), "mode": np.asarray(self.mode),
                "drive_reason": np.asarray(self.drive_reason),
                "match_angle_degrees": np.asarray(self.match_angle_degrees),
                "max_match_angle_degrees": np.asarray(self.max_match_angle),
            }
            arrays.update({"point_attribute_" + key: value for key, value in self.attributes.items()})
            np.savez_compressed(data, **arrays)
            return data.getvalue()


class SceneStore:
    def __init__(self, root, sequence=None, up_axis="y", max_match_angle=15.0):
        root = Path(root).resolve()
        self.up_axis = up_axis
        self.max_match_angle = max_match_angle
        self.paths = {}
        bundle = root if root.is_file() else root / "skelebones.npz"
        if bundle.is_file():
            input_path = Path(sequence).resolve() if sequence else bundle.parent / "sequence.npz"
            if not input_path.is_file():
                raise ValueError("use --sequence to supply the registered input sequence")
            self.paths[bundle.parent.name] = (bundle, input_path)
        else:
            for path in sorted(root.glob("*/result/skelebones.npz")):
                input_path = path.parent.parent / "sequence.npz"
                if input_path.is_file():
                    self.paths[path.parent.parent.name] = (path, input_path)
        if not self.paths:
            raise ValueError("no complete Skelebone bundles and input sequences found")
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def scene(self, name):
        with self.lock:
            if name not in self.paths:
                raise KeyError("scene not found")
            if name not in self.cache:
                self.cache[name] = AnimationScene(name, *self.paths[name], max_match_angle=self.max_match_angle)
            self.cache.move_to_end(name)
            return self.cache[name]

    def catalogue(self):
        scenes = []
        for name, (bundle, sequence) in self.paths.items():
            with np.load(bundle, allow_pickle=False) as data:
                scenes.append({
                    "id": name, "label": name, "status": "ready",
                    "frames": int(len(data["local_rotations"])),
                    "points": int(len(data["source_points"])), "joints": int(len(data["parents"])),
                })
        return {"scenes": scenes, "default_scene": "trex" if "trex" in self.paths else scenes[0]["id"]}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, **kwargs):
        self.store = store
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if args and str(args[1] if len(args) > 1 else "").startswith("5"):
            super().log_message(format, *args)

    def send_bytes(self, data, content_type, status=200, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value, status=200):
        self.send_bytes(json.dumps(value, allow_nan=False).encode(), "application/json", status)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/scenes":
                self.send_json(self.store.catalogue())
                return
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "scenes"]:
                scene = self.store.scene(parts[2])
                if len(parts) == 3:
                    self.send_json(dict(scene.metadata(), up_axis=self.store.up_axis))
                elif len(parts) == 5 and parts[3] == "buffers" and parts[4].endswith(".bin"):
                    with scene.lock:
                        data = scene._buffers[parts[4][:-4]].astype("<f4", copy=False).tobytes()
                    self.send_bytes(data, "application/octet-stream")
                elif len(parts) == 4 and parts[3] == "export.npz":
                    self.send_bytes(scene.export(), "application/octet-stream", filename=f"{scene.name}-driven.npz")
                else:
                    raise KeyError("endpoint not found")
                return
            web = Path(__file__).parent / "web"
            name = "index.html" if path == "/" else path.lstrip("/")
            if name not in {"index.html", "app.js", "scene.js", "style.css"}:
                raise KeyError("file not found")
            self.send_bytes((web / name).read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")
        except (KeyError, FileNotFoundError) as error:
            self.send_json({"error": str(error)}, 404)
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def do_POST(self):
        parts = unquote(urlparse(self.path).path).strip("/").split("/")
        try:
            if len(parts) != 4 or parts[:2] != ["api", "scenes"] or parts[3] != "animate":
                raise KeyError("endpoint not found")
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 65536:
                raise ValueError("invalid animation request")
            value = json.loads(self.rfile.read(size))
            scene = self.store.scene(parts[2])
            if value.get("reset", False):
                result = scene.pose({}, mode=value.get("mode"))
            elif "offsets" in value:
                result = scene.pose(value["offsets"], mode=value.get("mode"))
            else:
                result = scene.drive(value["joint"], value["rotation"])
            self.send_json(result)
        except KeyError as error:
            self.send_json({"error": str(error)}, 404)
        except (ValueError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.send_json({"error": str(error)}, 500)


def main():
    parser = argparse.ArgumentParser(description="Pose a Skelebone asset with optional reference motion in a browser")
    parser.add_argument("root", type=Path, help="experiment output root or one skelebones.npz")
    parser.add_argument("--sequence", type=Path, help="registered input for a single bundle")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--up-axis", choices=("y", "z"), default="y")
    parser.add_argument("--max-match-angle", type=float, default=15.0,
                        help="maximum per-joint matching error in degrees before LBS fallback (default: 15)")
    args = parser.parse_args()
    store = SceneStore(args.root, args.sequence, args.up_axis, args.max_match_angle)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, store=store))
    print(f"Skelebone Studio: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
