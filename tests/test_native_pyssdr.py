import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "native" / "pyssdr" / "build"


@pytest.mark.skipif(
    os.environ.get("GAUSSIANIMATE_TEST_NATIVE_PYSSDR") != "1",
    reason="set GAUSSIANIMATE_TEST_NATIVE_PYSSDR=1 after building native/pyssdr",
)
def test_native_backend_defaults_preserve_legacy_split_protocol():
    sys.path.insert(0, str(BUILD))
    try:
        import pyssdr
    finally:
        sys.path.pop(0)

    backend = pyssdr.MyDemBones()

    assert backend.clusterThreshold == 3
    assert backend.labelAssignmentMode == 1
    assert backend.pointCloudSmoothMode == 1


@pytest.mark.skipif(
    os.environ.get("GAUSSIANIMATE_TEST_NATIVE_PYSSDR") != "1",
    reason="set GAUSSIANIMATE_TEST_NATIVE_PYSSDR=1 after building native/pyssdr",
)
def test_native_backend_exposes_reproducible_initialization_controls():
    sys.path.insert(0, str(BUILD))
    try:
        import pyssdr
    finally:
        sys.path.pop(0)

    backend = pyssdr.MyDemBones()
    backend.clusterThreshold = 3
    backend.labelAssignmentMode = 1
    backend.splitSeedNeighbors = 0
    backend.pruneDelay = 0
    backend.zeroGrowthPatience = 0
    backend.pointCloudSmoothMode = 0
    backend.nIters = 2

    rest = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [10, 0, 0],
            [10, 1, 0],
            [11, 0, 0],
            [11, 1, 0],
        ],
        dtype=float,
    )
    posed = rest.copy()
    posed[4:, 1] += 3
    trajectory = np.concatenate([rest.T, posed.T], axis=0)
    backend.load_data(trajectory, [])
    weights, _, _ = backend.run_ssdr(2, "")

    assert backend.initializedBones == 2
    assert weights.shape == (2, 8)
    assert sorted(np.bincount(np.asarray(weights.argmax(axis=0)).reshape(-1))) == [4, 4]
    assert hasattr(backend, "labelToWeights")
    assert hasattr(backend, "computeTransFromLabel")
    assert hasattr(backend, "label")
