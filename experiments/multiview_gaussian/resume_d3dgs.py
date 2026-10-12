"""Memory-safe launcher for the historical Dynamic3DGaussians trainer."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import numpy as np


DYNAMIC_KEYS = {"means3D", "rgb_colors", "unnorm_rotations"}

COLOR_BOUND_MODES = ("none", "sigmoid_rgb")
RGB_EPSILON = 1e-4


def decode_rgb_colors(raw_rgb):
    """Map unconstrained trainable logits to physical RGB values."""

    return raw_rgb.sigmoid()


def _decoded_params(params):
    decoded = dict(params)
    decoded["rgb_colors"] = decode_rgb_colors(params["rgb_colors"])
    return decoded


def _encode_initial_rgb(params):
    """Replace freshly initialized RGB values with finite trainable logits."""

    rgb = params["rgb_colors"]
    encoded = rgb.detach().clamp(RGB_EPSILON, 1.0 - RGB_EPSILON).logit()
    params["rgb_colors"] = type(rgb)(encoded.contiguous().requires_grad_(True))
    return params


def _install_color_bound_mode(trainer: ModuleType, color_bound_mode: str) -> None:
    """Install a differentiable RGB-logit view around the external trainer."""

    if color_bound_mode not in COLOR_BOUND_MODES:
        raise ValueError(
            f"unsupported D3DGS color bound mode: {color_bound_mode}"
        )
    if color_bound_mode == "none":
        return

    for initializer_name in (
        "initialize_params_from_pointcloud",
        "initialize_params_random",
    ):
        original_initializer = getattr(trainer, initializer_name)

        def initialize_with_logits(
            *args, _initializer=original_initializer, **kwargs
        ):
            params, variables = _initializer(*args, **kwargs)
            return _encode_initial_rgb(params), variables

        setattr(trainer, initializer_name, initialize_with_logits)

    original_get_loss = trainer.get_loss

    def get_loss_with_decoded_rgb(params, *args, **kwargs):
        return original_get_loss(_decoded_params(params), *args, **kwargs)

    trainer.get_loss = get_loss_with_decoded_rgb

    original_report_progress = trainer.report_progress

    def report_progress_with_decoded_rgb(params, *args, **kwargs):
        return original_report_progress(
            _decoded_params(params), *args, **kwargs
        )

    trainer.report_progress = report_progress_with_decoded_rgb

    original_initialize_per_timestep = trainer.initialize_per_timestep

    def initialize_per_timestep_with_decoded_reference(
        params, variables, optimizer
    ):
        params, variables = original_initialize_per_timestep(
            params, variables, optimizer
        )
        variables["prev_col"] = decode_rgb_colors(
            params["rgb_colors"]
        ).detach()
        return params, variables

    trainer.initialize_per_timestep = (
        initialize_per_timestep_with_decoded_reference
    )

    original_params2cpu = trainer.params2cpu

    def params2cpu_with_decoded_rgb(params, is_initial_timestep):
        return original_params2cpu(
            _decoded_params(params), is_initial_timestep
        )

    trainer.params2cpu = params2cpu_with_decoded_rgb

    original_save_training_state = trainer.save_training_state

    def save_training_state_with_marker(path, *args, **kwargs):
        result = original_save_training_state(path, *args, **kwargs)
        Path(f"{path}.color_bound_mode").write_text(
            f"{color_bound_mode}\n", encoding="utf-8"
        )
        return result

    trainer.save_training_state = save_training_state_with_marker

    original_load_training_state = trainer.load_training_state

    def load_training_state_with_marker(path):
        marker = Path(f"{path}.color_bound_mode")
        if not marker.is_file() or marker.read_text(
            encoding="utf-8"
        ).strip() != color_bound_mode:
            raise RuntimeError(
                "D3DGS training state has no matching color-bound marker; "
                "refusing to reinterpret saved RGB values as logits"
            )
        return original_load_training_state(path)

    trainer.load_training_state = load_training_state_with_marker


def load_output_history_once(path: Path) -> List[Dict[str, np.ndarray]]:
    """Load every NPZ member once and expose per-frame views to the trainer."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    frame_count = int(arrays["means3D"].shape[0])
    history = []
    for frame in range(frame_count):
        record = {}
        for key, value in arrays.items():
            if key in DYNAMIC_KEYS:
                record[key] = value[frame]
            elif frame == 0:
                record[key] = value[0] if frame_count == 1 else value
        history.append(record)
    return history


def _load_trainer(root: Path) -> ModuleType:
    script = root / "train_dna.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "_gaussianimate_external_train_dna", script
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load D3DGS trainer: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3dgs-root", type=Path, required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-timesteps", type=int, required=True)
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--isotropic", action="store_true")
    scale_group.add_argument("--anisotropic", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument(
        "--color-bound-mode",
        choices=COLOR_BOUND_MODES,
        default="none",
    )
    args = parser.parse_args()

    root = args.d3dgs_root.expanduser().resolve()
    os.chdir(root)
    trainer = _load_trainer(root)
    trainer.load_output_history = lambda path: load_output_history_once(
        Path(path)
    )
    _install_color_bound_mode(trainer, args.color_bound_mode)
    if not trainer.validate_data(args.seq, str(args.data_dir)):
        raise RuntimeError(f"D3DGS data validation failed: {args.seq}")
    trainer.train(
        args.seq,
        args.exp,
        str(args.data_dir),
        use_pointcloud=True,
        num_timesteps=args.num_timesteps,
        mask_background=not args.no_mask,
        num_iters=args.num_iters,
        output_base=str(args.output_dir),
        pure_fg=False,
        isotropic=args.isotropic and not args.anisotropic,
        use_floor_loss=not args.no_floor,
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )


if __name__ == "__main__":
    main()
