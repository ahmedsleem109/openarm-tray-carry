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

"""Offscreen multi-camera rendering, shared by the gym envs and the controllers.

One :class:`CameraRig` owns a single ``mujoco.Renderer`` and drives it over
several named cameras, because a renderer allocates a framebuffer and a GL
context: making one per camera costs ~0.9 s each and a chunk of the little RAM
this machine has spare.

The rig also applies the **camera sensor model**. The torque channel already
carries noise proportional to each actuator's range, but the cameras used to
return simulator-perfect pixels, which makes a vision policy trained on them
brittle in exactly the way sim-to-real cares about: it can key on absolute pixel
values that no real sensor reproduces. :meth:`render` optionally applies

* a per-frame **gain**, standing in for auto-exposure and lighting drift, and
* per-pixel zero-mean **read noise**.

Both are off by default so existing measurements do not move.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

#: Default offscreen resolution, matching :class:`~openarm_gym.env.OpenArmEnv`.
DEFAULT_IMAGE_SIZE = (240, 320)


class CameraRig:
    """Render named cameras offscreen from one lazily-created renderer.

    The renderer is created on first use rather than in ``__init__`` so that
    constructing an environment stays cheap and headless-safe when no camera was
    asked for.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: tuple[str, ...] = (),
        *,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        gain_noise: float = 0.0,
        read_noise: float = 0.0,
    ) -> None:
        """Record what to render and how noisy the sensor is.

        Args:
            model: the compiled model the cameras belong to.
            camera_names: cameras to render, in observation order.
            image_size: ``(height, width)`` in pixels.
            gain_noise: standard deviation of the per-frame multiplicative gain,
                as a fraction (``0.05`` is +-5% exposure drift).
            read_noise: standard deviation of the per-pixel additive noise, in
                8-bit levels (``2.0`` is a couple of levels of grain).

        """
        self._model = model
        self.camera_names = tuple(camera_names)
        self.image_size = tuple(image_size)
        self.gain_noise = float(gain_noise)
        self.read_noise = float(read_noise)
        self._renderer: mujoco.Renderer | None = None

        for name in self.camera_names:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"Camera '{name}' not found in model")

    def __bool__(self) -> bool:
        """Whether this rig renders anything at all."""
        return bool(self.camera_names)

    @property
    def noisy(self) -> bool:
        """Whether the sensor model would alter the rendered pixels."""
        return self.gain_noise > 0.0 or self.read_noise > 0.0

    def render(
        self,
        data: mujoco.MjData,
        name: str,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Render one camera as an ``(H, W, 3)`` uint8 array.

        ``rng`` supplies the sensor noise. Passing ``None`` renders the clean
        image even when noise is configured, which is what a deterministic test
        or a debugging render wants.
        """
        if self._renderer is None:
            height, width = self.image_size
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        self._renderer.update_scene(data, camera=name)
        frame = self._renderer.render()
        if rng is None or not self.noisy:
            return frame
        return self._apply_sensor_noise(frame, rng)

    def render_all(
        self, data: mujoco.MjData, rng: np.random.Generator | None = None
    ) -> dict[str, np.ndarray]:
        """Render every configured camera, keyed by name."""
        return {name: self.render(data, name, rng) for name in self.camera_names}

    def _apply_sensor_noise(
        self, frame: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply one frame's gain and per-pixel read noise, clipped to uint8."""
        out = frame.astype(np.float32)
        if self.gain_noise > 0.0:
            out *= 1.0 + rng.normal(0.0, self.gain_noise)
        if self.read_noise > 0.0:
            out += rng.normal(0.0, self.read_noise, size=out.shape)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    def close(self) -> None:
        """Release the renderer and its GL context."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def write_mp4(
    path: str | Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int = 50,
    crf: int = 20,
) -> Path:
    """Encode RGB frames to an H.264 MP4.

    Uses PyAV, which lerobot already pulls in as its video decoder, so watching a
    rollout costs no new dependency. H.264 needs even dimensions, so an odd frame
    size is cropped by a row or column rather than rescaled.
    """
    import av

    if not frames:
        raise ValueError("no frames to encode")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError(
            f"every frame must have the same shape; got {sorted(shapes)}. "
            "H.264 fixes the frame size at the start of the stream, so mixed "
            "sizes would be cropped into nonsense rather than rejected."
        )
    height, width = frames[0].shape[:2]
    height -= height % 2
    width -= width % 2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for frame in frames:
            picture = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame[:height, :width]), format="rgb24"
            )
            container.mux(stream.encode(picture))
        container.mux(stream.encode())
    return path
