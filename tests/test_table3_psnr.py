import json
import sys

import numpy as np
import pytest
from experiments.multiview_gaussian import table3_psnr

from experiments.multiview_gaussian.table3_psnr import (
    combined_summary_lines,
    load_driven_trainset,
    load_ssdr_reconstruction,
    read_testset_psnr,
    summary_lines,
    write_summary,
)


def test_trainset_uses_ssdr_reconstruction_and_rejects_wrong_split(tmp_path):
    cache = tmp_path / "train_only.npz"
    reconstructed = np.full((2, 3, 3), 0.25, dtype=np.float32)
    np.savez(
        cache,
        reconstruction_original=reconstructed,
        training_frames=np.int64(2),
        total_frames=np.int64(3),
    )

    np.testing.assert_array_equal(
        load_ssdr_reconstruction(cache, (3, 3, 3)), reconstructed
    )
    with pytest.raises(ValueError, match="training split"):
        load_ssdr_reconstruction(cache, (4, 3, 3))


def test_testset_psnr_requires_matching_case_and_protocol(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "partmm_repaired": {"psnr": 21.5},
                "protocol": {
                    "sequence": "dna_0012_11",
                    "method": "partmm_repaired",
                    "attribute_source": "target_frame",
                    "camera_ids": [0, 12, 24, 36],
                    "test_frame_ids": [2],
                },
            }
        ),
        encoding="utf-8",
    )

    assert read_testset_psnr(evaluation, "dna_0012_11", 2, 3, [0, 12, 24, 36]) == 21.5
    with pytest.raises(ValueError, match="camera"):
        read_testset_psnr(evaluation, "dna_0012_11", 2, 3, [0])
    with pytest.raises(ValueError, match="split"):
        read_testset_psnr(evaluation, "dna_0012_11", 1, 3, [0, 12, 24, 36])


def test_driven_trainset_uses_saved_prediction_not_ground_truth_context(tmp_path):
    prediction_dir = tmp_path / "lbs"
    prediction_dir.mkdir()
    predicted = np.full((3, 2, 3), 0.25, dtype=np.float32)
    np.save(prediction_dir / "warped_sequence.npy", predicted)
    np.save(prediction_dir / "point_indices.npy", np.arange(2))
    (prediction_dir / "metadata.json").write_text(
        json.dumps({"training_frames": 2}), encoding="utf-8"
    )
    train = load_driven_trainset(prediction_dir, (3, 2, 3))
    np.testing.assert_array_equal(train, predicted[:2])

    (prediction_dir / "metadata.json").write_text(
        json.dumps({
            "training_frames": 2,
            "warped_sequence_training_payload": "ground_truth_context_not_predictions",
        }), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ground truth"):
        load_driven_trainset(prediction_dir, (3, 2, 3))


def test_summary_prints_only_two_psnr_columns():
    rows = [
        {"case": "actorshq_actor01_seq1", "trainset_psnr": 30.0, "testset_psnr": 20.0},
        {"case": "dna_0012_11", "trainset_psnr": 32.0, "testset_psnr": 24.0},
    ]
    lines = summary_lines(rows)
    assert len(lines) == 5
    assert "TrainSet PSNR" in lines[0] and "Test Set PSNR" in lines[0]
    assert [line.split()[-2:] for line in lines[1:]] == [
        ["30.00", "20.00"],
        ["32.00", "24.00"],
        ["30.00", "20.00"],
        ["32.00", "24.00"],
    ]
    assert all("RMSE" not in line and "SSIM" not in line for line in lines)


def test_subset_summary_does_not_claim_dataset_mean():
    rows = [
        {"case": "dna_0012_11", "trainset_psnr": 30.0, "testset_psnr": 20.0},
        {"case": "dna_0013_01", "trainset_psnr": 31.0, "testset_psnr": 21.0},
    ]
    lines = summary_lines(rows, include_means=False)
    assert len(lines) == 3
    assert all("mean" not in line.lower() for line in lines)


def test_summary_file_keeps_dataset_means(tmp_path):
    rows = [
        {"case": "actorshq_actor01_seq1", "trainset_psnr": 30.0, "testset_psnr": 20.0},
        {"case": "dna_0012_11", "trainset_psnr": 32.0, "testset_psnr": 24.0},
    ]
    write_summary(tmp_path, rows)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["actorshq"] == {"trainset_psnr": 30.0, "testset_psnr": 20.0}
    assert summary["dna_rendering"] == {"trainset_psnr": 32.0, "testset_psnr": 24.0}


def test_combined_summary_contains_all_methods_and_only_psnr():
    summaries = {
        method: {
            "actorshq": {"trainset_psnr": 30.0, "testset_psnr": 20.0},
            "dna_rendering": {"trainset_psnr": 32.0, "testset_psnr": 24.0},
        }
        for method in ("partmm", "lbs", "bob")
    }
    lines = combined_summary_lines(summaries)
    assert len(lines) == 7
    assert "TrainSet PSNR" in lines[0] and "Test Set PSNR" in lines[0]
    assert all(line.split()[-2:] in (["30.00", "20.00"], ["32.00", "24.00"])
               for line in lines[1:])
    assert all("RMSE" not in line and "SSIM" not in line for line in lines)


@pytest.fixture
def report_cli(tmp_path, monkeypatch):
    calls = []

    def evaluate(case, root, renderer, data, gpu, cameras, method):
        calls.append((case, method, cameras))
        return {'case': case, 'method': method, 'trainset_psnr': float(len(calls)),
                'testset_psnr': 20.0, 'protocol': {'camera_ids': list(range(cameras))}}

    monkeypatch.setattr(table3_psnr, 'evaluate_trainset', evaluate)
    monkeypatch.setattr(table3_psnr, 'ACTORSHQ_IDS', ['actorshq_actor01_seq1'])
    monkeypatch.setattr(table3_psnr, 'DNA_IDS', ['dna_0012_11'])

    def run(*arguments):
        monkeypatch.setattr(sys, 'argv', ['table3_psnr', '--root', str(tmp_path),
                           '--d3dgs-root', str(tmp_path / 'renderer'),
                           '--data-dir', str(tmp_path / 'data'), *arguments])
        table3_psnr.main()

    return run, calls


def test_cli_cache_tracks_camera_count_and_upstream_params(report_cli, tmp_path):
    run, calls = report_cli
    run('--case', 'dna_0012_11')
    run('--case', 'dna_0012_11')
    assert len(calls) == 1
    run('--case', 'dna_0012_11', '--num-cameras', '1')
    assert len(calls) == 2
    params = tmp_path / 'resources/multiview/d3dgs/dna_rendering/dna_0012_11/params.npz'
    params.parent.mkdir(parents=True)
    params.write_bytes(b'new checkpoint')
    run('--case', 'dna_0012_11', '--num-cameras', '1')
    assert len(calls) == 3


def test_summary_only_rejects_stale_cache(report_cli):
    run, _ = report_cli
    run('--case', 'dna_0012_11')
    with pytest.raises(ValueError, match='stale'):
        run('--case', 'dna_0012_11', '--num-cameras', '1', '--summary-only')


def test_all_rebuilds_means_from_refreshed_case_rows(report_cli, tmp_path):
    run, calls = report_cli
    for method in ('lbs', 'bob', 'partmm'):
        run('--method', method)
    run('--method', 'partmm', '--case', 'dna_0012_11', '--refresh')
    latest = float(len(calls))
    run('--method', 'all')
    summary = json.loads((tmp_path / 'runs/table3_psnr/all_methods_summary.json').read_text())
    assert summary['partmm']['dna_rendering']['trainset_psnr'] == latest


def test_all_checks_actual_case_ids_not_just_count(report_cli, tmp_path):
    run, _ = report_cli
    for method in ('lbs', 'bob', 'partmm'):
        run('--method', method)
    path = tmp_path / 'runs/table3_psnr/lbs/dna_0012_11.json'
    row = json.loads(path.read_text())
    row['case'] = 'actorshq_actor01_seq1'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='case'):
        run('--method', 'all')
