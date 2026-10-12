"""Standalone DeformingThings4D ``.anime`` to skelebone ``.npz`` preprocessing.

Parses the raw ``.anime`` mesh-animation format and writes the public
``DynamicPointSequence`` NPZ consumed by the skelebone pipeline. No legacy
workspace or RigGS imports are involved.

``.anime`` layout (little-endian): an ``int32`` header ``nf, nv, nt``;
``nv*3`` ``float32`` rest vertices; ``nt*3`` ``int32`` triangle indices; then
``(nf-1)*nv*3`` ``float32`` offsets. Offsets are absolute displacements from
the first (rest) frame, not per-frame deltas: ``points[0] = rest`` and
``points[i] = rest + offsets[i-1]``.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from skelebone.io import save_point_sequence
from skelebone.types import DynamicPointSequence

DEFAULT_FPS = 30.0

_HEADER = struct.Struct("<iii")
_VERTEX_DTYPE = np.dtype("<f4")
_INDEX_DTYPE = np.dtype("<i4")


def read_anime(path):
    """Parse a raw ``.anime`` file into ``(points[T, N, 3] float32, faces[F, 3] int32)``."""

    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError(f"{path}: too small for a .anime header ({file_size} bytes)")
        nf, nv, nt = _HEADER.unpack(header)
        if nf < 1 or nv < 1 or nt < 0:
            raise ValueError(f"{path}: invalid .anime dimensions nf={nf} nv={nv} nt={nt}")
        expected = (
            _HEADER.size
            + (nv * 3 + (nf - 1) * nv * 3) * _VERTEX_DTYPE.itemsize
            + nt * 3 * _INDEX_DTYPE.itemsize
        )
        if file_size != expected:
            state = "truncated" if file_size < expected else "malformed (trailing data)"
            raise ValueError(
                f"{path}: {state} .anime file: header nf={nf} nv={nv} nt={nt} "
                f"implies {expected} bytes, found {file_size}"
            )
        rest = np.fromfile(handle, dtype=_VERTEX_DTYPE, count=nv * 3).reshape(nv, 3)
        faces = np.fromfile(handle, dtype=_INDEX_DTYPE, count=nt * 3).reshape(nt, 3)
        offsets = np.fromfile(handle, dtype=_VERTEX_DTYPE, count=(nf - 1) * nv * 3).reshape(
            nf - 1, nv, 3
        )

    points = np.empty((nf, nv, 3), dtype=_VERTEX_DTYPE)
    points[0] = rest
    if nf > 1:
        points[1:] = rest[np.newaxis] + offsets
    return points, faces


def load_anime(path, fps: float = DEFAULT_FPS) -> DynamicPointSequence:
    """Load a raw DeformingThings4D ``.anime`` mesh animation."""

    points, faces = read_anime(path)
    return DynamicPointSequence(points, faces=faces, fps=fps)


def select_frame_range(num_frames: int, start_frame: int, frames):
    """Resolve a contiguous ``[start_frame, start_frame + frames)`` selection."""

    if start_frame < 0:
        raise ValueError(f"start frame must be non-negative, got {start_frame}")
    if frames is None:
        frames = num_frames - start_frame
    if frames < 1:
        raise ValueError(f"frame count must be at least 1, got {frames}")
    if start_frame + frames > num_frames:
        raise ValueError(
            f"requested frame range [{start_frame}, {start_frame + frames}) "
            f"exceeds the {num_frames} available frames"
        )
    return start_frame, start_frame + frames


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a raw DeformingThings4D .anime mesh animation into the "
            "public skelebone .npz point-sequence format."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="raw .anime file to read")
    parser.add_argument("--output", required=True, type=Path, help=".npz point sequence to write")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS, help="source animation frame rate (default: 30)"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="first frame to keep (default: 0)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="number of contiguous frames to keep (default: through the last frame)",
    )
    args = parser.parse_args(argv)

    try:
        points, faces = read_anime(args.input)
        start, end = select_frame_range(points.shape[0], args.start_frame, args.frames)
        sequence = DynamicPointSequence(points[start:end], faces=faces, fps=args.fps)
        saved = save_point_sequence(args.output, sequence)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
