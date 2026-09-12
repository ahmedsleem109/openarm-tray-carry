# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the tray vision policy, its dataset and its two control modes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openarm_gym.control.bimanual_tray import balance_law  # noqa: E402
from openarm_gym.policies.tray_vision import (  # noqa: E402
    BALL_SCALE,
    FRAME_STRIDE,
    TILT_SCALE,
    VELOCITY_SCALE,
    CameraEncoder,
    SpatialSoftmax,
    TrayVisionController,
    TrayVisionDataset,
    TrayVisionPolicy,
    collate,
    prepare_images,
)

CAMERAS = ("balancecam", "topcam")
SIZE = (84, 112)
PROPRIO = 25


@pytest.fixture
def episodes(tmp_path):
    """Two synthetic episodes whose frames encode their own timestep.

    Pixel [0, 0, 0] holds the step index, which lets the frame-stacking tests
    assert *which* frames were selected rather than only their shape.
    """
    steps = 12
    for seed in (0, 1):
        frames = {
            name: np.zeros((steps, *SIZE, 3), dtype=np.uint8) for name in CAMERAS
        }
        for name in CAMERAS:
            for t in range(steps):
                frames[name][t, 0, 0, 0] = t
        np.savez_compressed(
            tmp_path / f"episode_{seed:05d}.npz",
            labels=np.full((steps, 2), 0.01, dtype=np.float32),
            ball_xy=np.full((steps, 2), 0.02, dtype=np.float32),
            ball_vxy=np.full((steps, 2), 0.04, dtype=np.float32),
            proprio=np.zeros((steps, PROPRIO), dtype=np.float32),
            positions=np.zeros((steps, 3), dtype=np.float32),
            **{f"pixels_{n}": frames[n] for n in CAMERAS},
        )
    return tmp_path


# -------------------------------------------------------------------- the model


def test_spatial_softmax_finds_the_peak() -> None:
    """A single hot pixel must come back as its own normalised coordinate."""
    softmax = SpatialSoftmax(5, 5)
    features = torch.full((1, 1, 5, 5), -30.0)
    features[0, 0, 0, 4] = 30.0  # top-right corner
    out = softmax(features)
    assert out.shape == (1, 2)
    assert float(out[0, 0]) == pytest.approx(1.0, abs=1e-3)   # x = +1
    assert float(out[0, 1]) == pytest.approx(-1.0, abs=1e-3)  # y = -1


@pytest.mark.parametrize("size", [(84, 112), (64, 64), (100, 130)])
def test_camera_encoder_matches_its_softmax_grid_to_any_input_size(size) -> None:
    """Padded stride-2 convolutions round up, so the grid must be inferred.

    84 becomes 42, 21, 11 -- not 84 // 8 == 10 -- and assuming the latter is a
    shape error at the first forward pass.
    """
    encoder = CameraEncoder(6, size, width=8)
    out = encoder(torch.zeros(2, 6, *size))
    assert out.shape == (2, 16)


def test_policy_returns_a_2d_tilt_and_a_4d_ball_state() -> None:
    """The state head carries position *and* velocity: four numbers."""
    policy = TrayVisionPolicy(CAMERAS, SIZE, PROPRIO, width=8)
    images = {name: torch.zeros(3, 6, *SIZE) for name in CAMERAS}
    tilt, state = policy(images, torch.zeros(3, PROPRIO))
    assert tilt.shape == (3, 2)
    assert state.shape == (3, 4)


def test_prepare_images_flattens_the_stack_into_channels() -> None:
    """``(B, S, H, W, 3)`` uint8 becomes ``(B, 3S, H, W)`` in [-0.5, 0.5]."""
    frames = np.full((2, 2, 4, 5, 3), 255, dtype=np.uint8)
    frames[:, 0] = 0
    out = prepare_images(frames)
    assert out.shape == (2, 6, 4, 5)
    assert float(out[0, 0].min()) == pytest.approx(-0.5)
    assert float(out[0, 3].max()) == pytest.approx(0.5)


# ------------------------------------------------------------------ the dataset


def test_dataset_scales_position_and_velocity_separately(episodes) -> None:
    """The 4-vector target is position/BALL_SCALE then velocity/VELOCITY_SCALE."""
    dataset = TrayVisionDataset(episodes, CAMERAS)
    _images, _proprio, tilt, state = dataset[5]
    assert state.shape == (4,)
    assert state[0] == pytest.approx(0.02 / BALL_SCALE)
    assert state[2] == pytest.approx(0.04 / VELOCITY_SCALE)
    assert tilt[0] == pytest.approx(0.01 / TILT_SCALE)


def test_dataset_stacks_the_strided_frames_oldest_first(episodes) -> None:
    """Sample t must pair frame t-stride with frame t, in that order."""
    dataset = TrayVisionDataset(episodes, CAMERAS)
    # Sample index 5 of the first episode.
    images, _proprio, _tilt, _state = dataset[5]
    stamps = [int(frame[0, 0, 0]) for frame in images["topcam"]]
    assert stamps == [5 - FRAME_STRIDE, 5]


def test_dataset_clamps_the_stack_at_the_episode_start(episodes) -> None:
    """Step 0 repeats its own frame rather than reaching into nothing.

    A repeated frame implies zero ball velocity, which at the start of an episode
    -- the settled hold after the lift -- is very nearly true.
    """
    dataset = TrayVisionDataset(episodes, CAMERAS)
    images, _proprio, _tilt, _state = dataset[0]
    stamps = [int(frame[0, 0, 0]) for frame in images["topcam"]]
    assert stamps == [0, 0]


def test_normalisation_floors_a_channel_that_barely_moves(episodes) -> None:
    """A near-constant channel must not be divided by ~0.

    The synthetic episodes hold proprioception at exactly zero, so an epsilon
    added to the standard deviation would divide drift by 1e-6 and turn a
    micrometre into an input of order 10^3 -- one that differs between training
    and inference. A floor keeps it finite and small.
    """
    dataset = TrayVisionDataset(episodes, CAMERAS)
    assert float(dataset.proprio_std.min()) >= 1e-3
    _images, proprio, _tilt, _state = dataset[3]
    assert np.all(np.isfinite(proprio))
    assert float(np.abs(proprio).max()) < 10.0


def test_prepare_images_rejects_an_unstacked_batch() -> None:
    """A missing stack axis would otherwise unpack into the wrong dimensions."""
    with pytest.raises(ValueError, match=r"\(B, S, H, W, 3\)"):
        prepare_images(np.zeros((2, 4, 5, 3), dtype=np.uint8))


def test_controller_takes_the_control_rate_it_differentiates_at() -> None:
    """The differencing scale factor has to match the loop it drives."""
    fast = _controller()
    slow = TrayVisionController(
        TrayVisionPolicy(CAMERAS, SIZE, PROPRIO, width=8),
        np.zeros(PROPRIO), np.ones(PROPRIO), control_hz=25.0,
    )
    assert fast.control_hz == 50.0
    assert slow.control_hz == 25.0
    # Same position history, half the rate, half the velocity.
    for controller in (fast, slow):
        controller.reset()
        for step in range(8):
            controller._differenced_velocity(np.array([0.001 * step, 0.0]))
    assert slow.velocity_estimate[0] == pytest.approx(
        fast.velocity_estimate[0] / 2.0, rel=1e-6
    )


def test_dataset_indexes_every_episode(episodes) -> None:
    """Both episodes contribute samples, and collation stacks them."""
    dataset = TrayVisionDataset(episodes, CAMERAS)
    assert len(dataset) == 24
    images, proprio, tilt, state = collate([dataset[0], dataset[7]])
    assert images["topcam"].shape == (2, 6, *SIZE)
    assert proprio.shape == (2, PROPRIO)
    assert tilt.shape == (2, 2)
    assert state.shape == (2, 4)


# --------------------------------------------------------------- the controller


def _controller(mode: str = "estimator") -> TrayVisionController:
    policy = TrayVisionPolicy(CAMERAS, SIZE, PROPRIO, width=8)
    return TrayVisionController(policy, np.zeros(PROPRIO), np.ones(PROPRIO), mode=mode)


def _observation(stamp: int) -> dict:
    pixels = {}
    for name in CAMERAS:
        frame = np.zeros((*SIZE, 3), dtype=np.uint8)
        frame[0, 0, 0] = stamp
        pixels[name] = frame
    return {
        "qpos": np.zeros(18, dtype=np.float32),
        "tray_pos": np.zeros(3, dtype=np.float32),
        "tray_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "pixels": pixels,
    }


def test_controller_stacks_frames_the_same_way_the_dataset_does() -> None:
    """Inference and training must agree on frame order, or nothing works.

    A reversed stack trains and validates perfectly and then fails closed loop,
    which is the most expensive way to get this wrong.
    """
    controller = _controller()
    for stamp in range(6):
        stack = controller._stack("topcam", _observation(stamp)["pixels"]["topcam"])
    stamps = [int(frame[0, 0, 0]) for frame in stack]
    assert stamps == [5 - FRAME_STRIDE, 5]


def test_controller_reset_forgets_the_previous_episode() -> None:
    """Otherwise a new episode's first frames stack against the old episode's."""
    controller = _controller()
    for stamp in range(6):
        controller._stack("topcam", _observation(stamp)["pixels"]["topcam"])
    controller.reset()
    stack = controller._stack("topcam", _observation(99)["pixels"]["topcam"])
    assert [int(f[0, 0, 0]) for f in stack] == [99, 99]


def test_estimator_mode_runs_its_estimate_through_the_classical_law() -> None:
    """The setpoint must be exactly ``balance_law`` of the predicted state.

    This is what keeps the loop gain correct by construction: a tilt regressor was
    measured to output about half the required magnitude, and half the gain is a
    different controller rather than a slightly worse one.
    """
    controller = _controller("estimator")
    setpoint = controller(_observation(0))
    expected = balance_law(controller.ball_estimate, controller.velocity_estimate)
    assert np.allclose(setpoint, expected)


def test_tilt_mode_returns_the_tilt_head_unscaled_by_the_law() -> None:
    """The end-to-end baseline bypasses the balance law entirely."""
    controller = _controller("tilt")
    setpoint = controller(_observation(0))
    law = balance_law(controller.ball_estimate, controller.velocity_estimate)
    assert setpoint.shape == (2,)
    # Untrained weights make an exact match vanishingly unlikely.
    assert not np.allclose(setpoint, law)


def test_unknown_mode_is_rejected() -> None:
    """A typo in the mode must not silently fall back to a different controller."""
    with pytest.raises(ValueError, match="estimator"):
        _controller("balance")
