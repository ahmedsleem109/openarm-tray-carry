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

"""Tests for the shared camera rig and the video writer."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from openarm_gym.assets import scene_path

from openarm_gym.vision import CameraRig, write_mp4



@pytest.fixture(scope="module")
def scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """A posed tray scene, shared by every test in the module."""
    model = mujoco.MjModel.from_xml_path(scene_path())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_rig_renders_every_camera_at_the_requested_size(scene) -> None:
    """Each named camera comes back as an ``(H, W, 3)`` uint8 image."""
    model, data = scene
    rig = CameraRig(model, ("balancecam", "topcam"), image_size=(48, 64))
    images = rig.render_all(data)
    assert set(images) == {"balancecam", "topcam"}
    for image in images.values():
        assert image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
    # The two cameras look at the scene from different places, so identical
    # output would mean the camera argument is being ignored.
    assert not np.array_equal(images["balancecam"], images["topcam"])
    rig.close()


def test_unknown_camera_is_rejected_at_construction(scene) -> None:
    """A typo in a camera name must fail immediately, not at the first render."""
    model, _data = scene
    with pytest.raises(ValueError, match="frontcam"):
        CameraRig(model, ("frontcam",))


def test_sensor_noise_is_off_unless_asked_for(scene) -> None:
    """A rig with no noise configured renders deterministically.

    This is what keeps every measurement recorded before the sensor model existed
    comparable with one taken after it.
    """
    model, data = scene
    rig = CameraRig(model, ("topcam",), image_size=(48, 64))
    assert not rig.noisy
    rng = np.random.default_rng(0)
    first = rig.render(data, "topcam", rng)
    second = rig.render(data, "topcam", np.random.default_rng(1))
    assert np.array_equal(first, second)
    rig.close()


def test_sensor_noise_perturbs_pixels_but_only_with_a_generator(scene) -> None:
    """Noise needs an explicit rng, so a debugging render stays clean."""
    model, data = scene
    rig = CameraRig(
        model, ("topcam",), image_size=(48, 64), gain_noise=0.05, read_noise=3.0
    )
    clean = rig.render(data, "topcam", None)
    noisy = rig.render(data, "topcam", np.random.default_rng(0))
    assert rig.noisy
    assert not np.array_equal(clean, noisy)
    # Still an image: within range, and recognisably the same view.
    assert noisy.dtype == np.uint8
    assert float(np.mean(np.abs(noisy.astype(int) - clean.astype(int)))) < 40.0
    rig.close()


def test_write_mp4_encodes_the_frames_it_is_given(scene, tmp_path) -> None:
    """A short rollout encodes to a playable file with the right frame count."""
    import av

    model, data = scene
    rig = CameraRig(model, ("balancecam",), image_size=(64, 85))
    frames = [rig.render(data, "balancecam") for _ in range(10)]
    path = write_mp4(tmp_path / "clip.mp4", frames, fps=25)
    assert path.stat().st_size > 0
    with av.open(str(path)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    assert decoded == len(frames)
    # H.264 needs even dimensions, and 85 is odd, so the writer crops one column.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.height, stream.codec_context.width) == (64, 84)
    rig.close()


def test_write_mp4_rejects_frames_of_differing_size(tmp_path) -> None:
    """H.264 fixes the frame size at the start of the stream.

    Without the check the odd frame is cropped into nonsense rather than
    reported, which reads as a corrupted render rather than a caller mistake.
    """
    frames = [np.zeros((16, 16, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)]
    with pytest.raises(ValueError, match="same shape"):
        write_mp4(tmp_path / "ragged.mp4", frames)


def test_write_mp4_rejects_an_empty_rollout(tmp_path) -> None:
    """Encoding nothing is a mistake worth reporting, not an empty file."""
    with pytest.raises(ValueError, match="no frames"):
        write_mp4(tmp_path / "empty.mp4", [])
