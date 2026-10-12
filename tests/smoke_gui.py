"""Build and exercise the optional Dear PyGui editor without a display server."""

import tempfile
from pathlib import Path

import numpy as np

from partmm.gui import launch
from skelebone import PoseSequence, Skeleton, SkeleboneBundle
from skelebone.io import save_bundle


def main() -> None:
    skeleton = Skeleton(
        rest_joints=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        parents=np.array([-1, 0]),
        names=["root", "tip"],
    )
    pose = PoseSequence.identity(2, 2)
    bundle = SkeleboneBundle(
        skeleton=skeleton,
        weights=np.ones((2, 1)),
        source_points=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        motion=pose,
        bone_transforms=np.broadcast_to(np.eye(4), (2, 1, 4, 4)).copy(),
    )
    with tempfile.TemporaryDirectory() as directory:
        bundle_path = save_bundle(Path(directory) / "bundle.npz", bundle)
        launch(bundle_path, Path(directory) / "edited_pose.npz", headless_smoke_test=True)
    print("Dear PyGui pose editor smoke test passed")


if __name__ == "__main__":
    main()
