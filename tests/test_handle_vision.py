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

"""Tests for the handle detector that makes the grasp a vision problem."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry
from openarm_gym.policies.handle_vision import (
    HANDLE_NOMINAL,
    SIDES,
    HandleDetector,
    HandleEstimator,
    collate,
    decode_handles,
    encode_handles,
)

CAMERAS = ("topcam",)
IMAGE = (42, 56)


def _detector() -> HandleDetector:
    return HandleDetector(CAMERAS, IMAGE, width=8, hidden=16)


def test_encoding_a_label_and_decoding_it_returns_the_same_positions() -> None:
    """The label transform is the contract between the recorder and the grasp.

    A sign or an axis slipped here would train a network that is accurate on its
    own targets and points the grippers somewhere else.
    """
    handles = {"left": np.array([0.36, 0.14, 0.44]), "right": np.array([0.34, -0.16, 0.45])}
    recovered = decode_handles(encode_handles(handles))
    for side in SIDES:
        assert np.allclose(recovered[side], handles[side], atol=1e-6)


def test_the_nominal_layout_encodes_to_zero() -> None:
    """Predictions are offsets, so the authored layout has to be the origin."""
    nominal = {side: HANDLE_NOMINAL[i] for i, side in enumerate(SIDES)}
    assert np.allclose(encode_handles(nominal), np.zeros(6))


def test_the_detector_maps_one_frame_per_camera_to_six_numbers() -> None:
    """Left xyz then right xyz -- the layout :func:`decode_handles` expects."""
    detector = _detector()
    images, _ = collate(
        [({"topcam": np.zeros((*IMAGE, 3), dtype=np.uint8)}, np.zeros(6, np.float32))] * 3
    )
    assert images["topcam"].shape == (3, 3, *IMAGE)
    assert detector(images).shape == (3, 6)


def test_the_estimator_refuses_an_observation_without_its_cameras() -> None:
    """Silently estimating from the wrong cameras would look like a bad model."""
    estimator = HandleEstimator(_detector())
    with pytest.raises(ValueError, match="topcam"):
        estimator({"pixels": {"balancecam": np.zeros((*IMAGE, 3), dtype=np.uint8)}})


def test_the_estimator_reports_its_own_error_against_the_truth() -> None:
    """The error is reported, never fed back -- that is what keeps this vision."""
    detector = _detector()
    with torch.no_grad():
        detector.head.weight.zero_()
        detector.head.bias.zero_()
    estimator = HandleEstimator(detector)
    estimate = estimator({"pixels": {"topcam": np.zeros((*IMAGE, 3), dtype=np.uint8)}})
    # A zeroed head predicts no offset, so it returns exactly the nominal layout.
    for i, side in enumerate(SIDES):
        assert np.allclose(estimate[side], HANDLE_NOMINAL[i], atol=1e-6)
    truth = {side: HANDLE_NOMINAL[i] + np.array([0.01, 0.0, 0.0]) for i, side in enumerate(SIDES)}
    assert estimator.error(truth) == pytest.approx(0.01, abs=1e-6)


def test_an_estimate_equal_to_the_truth_grasps_exactly_as_the_privileged_path() -> None:
    """The two paths differ only in where the numbers came from.

    Worth pinning: it is what lets the visual grasp and the privileged grasp be
    compared on the same layouts and any difference be attributed to the estimate.
    """
    carry = BimanualTrayCarry(scene_path())
    carry.reset()
    carry.randomize_layout(np.random.default_rng(5))
    truth = {side: carry.handle_pos(side).copy() for side in SIDES}
    told = carry.grasp(handles=truth)
    told_pose = carry.tray_pose[0].copy()

    carry.reset()
    carry.randomize_layout(np.random.default_rng(5))
    read = carry.grasp()
    carry.close()

    assert np.allclose(told_pose, carry.tray_pose[0], atol=1e-9)
    for side in SIDES:
        assert told[side]["force"] == pytest.approx(read[side]["force"], rel=1e-9)
