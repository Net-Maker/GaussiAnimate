"""Run D3DGS and adopt its checkpoint into the normalized release layout."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from .resume_d3dgs import COLOR_BOUND_MODES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "revision": None,
            "dirty": None,
            "git_error": repr(exc),
        }


def adopt_checkpoint(
    source: Path,
    target: Path,
    *,
    d3dgs_root: Path,
    command,
    frames: int,
    protocol=None,
    execution=None,
) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        temporary = target.with_name(target.name + ".adopting")
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        import shutil

        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    adapter_root = Path(__file__).resolve().parents[2]
    adapter_files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("resume_d3dgs.py").resolve(),
    )
    source_files = tuple(
        d3dgs_root / name
        for name in ("train_dna.py", "helpers.py", "external.py")
    )
    missing_source_files = [path for path in source_files if not path.is_file()]
    if missing_source_files:
        raise FileNotFoundError(missing_source_files[0])
    record = {
        "schema_version": 1,
        "method": "Dynamic3DGaussians",
        "expected_frames": int(frames),
        "params": {
            "path": str(target.resolve()),
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        },
        "source": {
            "path": str(source.resolve()),
            **_git_state(d3dgs_root),
            "files": {
                path.relative_to(d3dgs_root).as_posix(): _sha256(path)
                for path in source_files
            },
        },
        "adapter": {
            **_git_state(adapter_root),
            "files": {
                path.relative_to(adapter_root).as_posix(): _sha256(path)
                for path in adapter_files
            },
        },
        "command": command,
    }
    if protocol is not None:
        record["protocol"] = protocol
    if execution is not None:
        record["execution"] = execution
    provenance = target.parent / "training_provenance.json"
    temporary_provenance = provenance.with_name(provenance.name + ".writing")
    temporary_provenance.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_provenance, provenance)
    return record


def build_training_command(
    *,
    d3dgs_root: Path,
    data_dir: Path,
    output_dir: Path,
    train_id: str,
    frames: int,
    iterations: int,
    checkpoint_every: int,
    experiment: str,
    scale_mode: str = "isotropic",
    color_bound_mode: str = "none",
) -> List[str]:
    """Build the dataset-specific D3DGS command recorded in provenance."""

    if scale_mode not in ("isotropic", "anisotropic"):
        raise ValueError(f"unsupported D3DGS scale mode: {scale_mode}")
    if color_bound_mode not in COLOR_BOUND_MODES:
        raise ValueError(
            f"unsupported D3DGS color bound mode: {color_bound_mode}"
        )
    launcher = Path(__file__).with_name("resume_d3dgs.py").resolve()
    return [
        sys.executable,
        str(launcher),
        "--d3dgs-root",
        str(d3dgs_root),
        "--seq",
        train_id,
        "--exp",
        experiment,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--num-timesteps",
        str(frames),
        "--num-iters",
        str(iterations),
        f"--{scale_mode}",
        "--color-bound-mode",
        color_bound_mode,
        "--no-floor",
        "--checkpoint-every",
        str(checkpoint_every),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d3dgs-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-id", required=True)
    parser.add_argument("--resource-id", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--experiment", default="paper_dna")
    parser.add_argument(
        "--scale-mode",
        choices=("isotropic", "anisotropic"),
        default="isotropic",
    )
    parser.add_argument(
        "--color-bound-mode",
        choices=COLOR_BOUND_MODES,
        default="none",
    )
    parser.add_argument("--adopt-only", action="store_true")
    args = parser.parse_args()

    command = build_training_command(
        d3dgs_root=args.d3dgs_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_id=args.train_id,
        frames=args.frames,
        iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        experiment=args.experiment,
        scale_mode=args.scale_mode,
        color_bound_mode=args.color_bound_mode,
    )
    if not args.adopt_only:
        subprocess.run(command, cwd=args.d3dgs_root, check=True)
    source = args.output_dir / args.train_id / "params.npz"
    target = args.output_dir / args.resource_id / "params.npz"
    record = adopt_checkpoint(
        source,
        target,
        d3dgs_root=args.d3dgs_root,
        command=command,
        frames=args.frames,
        protocol={
            "scale_mode": args.scale_mode,
            "color_bound_mode": args.color_bound_mode,
            "frames": args.frames,
            "first_frame_iterations": args.iterations,
            "checkpoint_every_frames": args.checkpoint_every,
            "mask_background": True,
            "floor_loss": False,
        },
        execution={
            "mode": (
                "adopt_existing_checkpoint"
                if args.adopt_only
                else "trained_then_finalized"
            ),
            "training_command_executed": not args.adopt_only,
        },
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
