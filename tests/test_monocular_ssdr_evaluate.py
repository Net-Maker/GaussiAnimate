from types import SimpleNamespace

import numpy as np

from experiments.monocular_gaussian.ssdr_evaluate import (
    deformation_attributes_for_render,
    nearest_frame_index,
    ordered_unique_views,
    run_ssdr,
    select_source_indices,
    transfer_weights_nearest,
)


class _CollapsedBackend:
    clusterThreshold = -1
    labelAssignmentMode = 0
    splitSeedNeighbors = 0
    pruneDelay = 0
    zeroGrowthPatience = 0
    pointCloudSmoothMode = 1

    def load_data(self, trajectory, faces):
        self.trajectory = trajectory

    def run_ssdr(self, num_bones, output_path):
        frames = self.trajectory.shape[0] // 3
        points = self.trajectory.shape[1]
        weights = np.ones((1, points), dtype=float)
        transforms = np.broadcast_to(np.eye(4), (frames, 1, 4, 4)).copy()
        return weights, transforms, 0.0


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_):
        return self

    def numel(self):
        return 1

    def item(self):
        return self.value

    @property
    def shape(self):
        return (1,)


def test_ordered_unique_views_sorts_and_deduplicates_fids():
    views = [
        SimpleNamespace(fid=_FakeTensor(0.5), name="late"),
        SimpleNamespace(fid=_FakeTensor(0.0), name="first"),
        SimpleNamespace(fid=_FakeTensor(0.5), name="duplicate"),
    ]
    assert [view.name for view in ordered_unique_views(views)] == ["first", "late"]


def test_nearest_frame_index_uses_trajectory_fids():
    frame_fids = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    assert nearest_frame_index(frame_fids, 0.62) == 2
    assert nearest_frame_index(frame_fids, 0.9) == 4


def test_select_source_indices_keeps_highest_opacity_in_source_order():
    opacities = np.asarray([0.2, 0.9, 0.1, 0.8], dtype=np.float32)
    np.testing.assert_array_equal(
        select_source_indices(opacities, 2),
        np.asarray([1, 3]),
    )


def test_transfer_weights_nearest_covers_full_point_cloud():
    source = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    full = np.asarray([[0, 0, 0], [0.1, 0, 0], [0.9, 0, 0]], dtype=np.float32)
    weights = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    np.testing.assert_array_equal(
        transfer_weights_nearest(full, source, weights),
        np.asarray([[1, 1, 0], [0, 0, 1]], dtype=np.float32),
    )


def test_isotropic_render_preserves_dynamic_scaling():
    rotation = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    scaling = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32)

    render_rotation, render_scaling = deformation_attributes_for_render(
        rotation,
        scaling,
        use_isotropic_gaussians=True,
    )

    np.testing.assert_array_equal(render_rotation, np.zeros_like(rotation))
    assert render_scaling is scaling


def test_run_ssdr_propagates_strict_bone_count_without_name_error():
    trajectory = np.zeros((2, 6, 3), dtype=np.float32)

    try:
        run_ssdr(
            trajectory,
            num_bones=3,
            rest_frame=0,
            max_iterations=1,
            tolerance=1e-3,
            patience=1,
            max_influences=2,
            protocol="paper_legacy",
            initialization="backend",
            require_num_bones=True,
            backend=_CollapsedBackend(),
        )
    except RuntimeError as error:
        assert "returned 1 bones; requested 3" in str(error)
    else:
        raise AssertionError("strict Horse SSDR accepted a collapsed backend")
