"""Tests for the differentiable D3DGS RGB parameterization."""

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.multiview_gaussian.resume_d3dgs import (
    COLOR_BOUND_MODES,
    _install_color_bound_mode,
    decode_rgb_colors,
)
from experiments.multiview_gaussian.train_d3dgs import build_training_command
from experiments.paper import load_paper_config, render_template


def _training_command(**overrides):
    arguments = {
        "d3dgs_root": Path("/external/d3dgs"),
        "data_dir": Path("/prepared/data"),
        "output_dir": Path("/release/d3dgs"),
        "train_id": "actorshq_actor02_seq2",
        "frames": 300,
        "iterations": 10000,
        "checkpoint_every": 10,
        "experiment": "release_v1",
    }
    arguments.update(overrides)
    return build_training_command(**arguments)


def _fake_trainer():
    def initialize(rgb):
        params = {
            "rgb_colors": torch.nn.Parameter(
                torch.tensor(rgb, dtype=torch.float32)
            )
        }
        return params, {}

    def get_loss(params, *args, **kwargs):
        return params["rgb_colors"].sum(), kwargs.get("variables", {})

    def initialize_per_timestep(params, variables, optimizer):
        variables["prev_col"] = params["rgb_colors"].detach()
        return params, variables

    def params2cpu(params, is_initial_timestep):
        return {
            "rgb_colors": np.asarray(
                params["rgb_colors"].detach().cpu().tolist(),
                dtype=np.float32,
            )
        }

    return types.SimpleNamespace(
        initialize_params_from_pointcloud=lambda *a, **k: initialize(
            [[0.0, 0.25, 1.0]]
        ),
        initialize_params_random=lambda *a, **k: initialize(
            [[0.1, 0.5, 0.9]]
        ),
        get_loss=get_loss,
        report_progress=lambda params, *a, **k: params["rgb_colors"],
        initialize_per_timestep=initialize_per_timestep,
        params2cpu=params2cpu,
        save_training_state=lambda path, *a, **k: Path(path).write_text(
            "state", encoding="utf-8"
        ),
        load_training_state=lambda path: (
            0,
            {
                "rgb_colors": torch.nn.Parameter(
                    torch.tensor([[-2.0, 0.0, 2.0]])
                )
            },
            {},
            object(),
        ),
    )


class ColorBoundModeTest(unittest.TestCase):
    def test_sigmoid_decode_is_bounded_and_differentiable(self):
        raw = torch.tensor([[-10.0, 0.0, 10.0]], requires_grad=True)
        decoded = decode_rgb_colors(raw)
        self.assertTrue(torch.all(decoded > 0.0))
        self.assertTrue(torch.all(decoded < 1.0))
        decoded.sum().backward()
        self.assertTrue(torch.all(raw.grad > 0.0))

    def test_fresh_initialization_encodes_rgb_exactly_once(self):
        trainer = _fake_trainer()
        _install_color_bound_mode(trainer, "sigmoid_rgb")
        params, _ = trainer.initialize_params_from_pointcloud()
        decoded = decode_rgb_colors(params["rgb_colors"]).detach()
        expected = torch.tensor([[1e-4, 0.25, 1.0 - 1e-4]])
        torch.testing.assert_close(decoded, expected, atol=1e-6, rtol=0.0)

    def test_loss_and_progress_receive_decoded_rgb(self):
        trainer = _fake_trainer()
        _install_color_bound_mode(trainer, "sigmoid_rgb")
        raw = torch.nn.Parameter(torch.zeros((1, 3)))
        params = {"rgb_colors": raw}
        loss, _ = trainer.get_loss(params)
        self.assertAlmostEqual(loss.item(), 1.5)
        loss.backward()
        torch.testing.assert_close(raw.grad, torch.full((1, 3), 0.25))
        torch.testing.assert_close(
            trainer.report_progress(params), torch.full((1, 3), 0.5)
        )

    def test_temporal_reference_and_public_output_are_decoded(self):
        trainer = _fake_trainer()
        _install_color_bound_mode(trainer, "sigmoid_rgb")
        raw = torch.nn.Parameter(torch.tensor([[-2.0, 0.0, 2.0]]))
        params = {"rgb_colors": raw}
        params, variables = trainer.initialize_per_timestep(
            params, {}, object()
        )
        expected = torch.sigmoid(raw).detach()
        torch.testing.assert_close(variables["prev_col"], expected)
        output = trainer.params2cpu(params, False)
        np.testing.assert_allclose(
            output["rgb_colors"], np.asarray(expected.tolist())
        )

    def test_resume_keeps_raw_logits_and_requires_matching_marker(self):
        trainer = _fake_trainer()
        _install_color_bound_mode(trainer, "sigmoid_rgb")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "training_state_latest.pt"
            trainer.save_training_state(state, 0, {}, {}, object())
            before = torch.tensor([[-2.0, 0.0, 2.0]])
            _, params, _, _ = trainer.load_training_state(state)
            torch.testing.assert_close(params["rgb_colors"].detach(), before)
            marker = Path(f"{state}.color_bound_mode")
            self.assertEqual(marker.read_text(encoding="utf-8"), "sigmoid_rgb\n")

    def test_resume_rejects_unmarked_state(self):
        trainer = _fake_trainer()
        _install_color_bound_mode(trainer, "sigmoid_rgb")
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "legacy.pt"
            state.write_text("state", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "color-bound marker"):
                trainer.load_training_state(state)

    def test_default_mode_leaves_trainer_unchanged(self):
        trainer = _fake_trainer()
        original = trainer.get_loss
        _install_color_bound_mode(trainer, "none")
        self.assertIs(trainer.get_loss, original)

    def test_command_and_config_request_sigmoid_rgb(self):
        command = _training_command(color_bound_mode="sigmoid_rgb")
        self.assertEqual(
            command[command.index("--color-bound-mode") + 1], "sigmoid_rgb"
        )
        self.assertEqual(COLOR_BOUND_MODES, ("none", "sigmoid_rgb"))

        root = Path(__file__).parents[1]
        config = load_paper_config(
            root / "experiments/multiview_gaussian/reconstruction.yaml"
        )
        stage = config["methods"][0]["stages"][0]
        actorshq = config["datasets"][1]
        command = render_template(
            stage,
            {
                "case": actorshq["cases"][0],
                "dataset": actorshq,
                "run_dir": "/tmp/run",
            },
        )["command"]
        self.assertEqual(
            command[command.index("--color-bound-mode") + 1], "sigmoid_rgb"
        )


if __name__ == "__main__":
    unittest.main()
