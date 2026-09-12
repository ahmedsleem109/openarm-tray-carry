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

"""A small vision policy that emits tray tilt setpoints, and its dataset.

This is the piece that turns the classical tray controller into a *vision*
controller. It is deliberately small, and the reasons are structural rather than
a compromise:

**The controller shrinks the action space to two numbers.** A policy that had to
emit 14 joint angles would need either a batched GPU renderer (there is none --
no ``madrona_mjx``) or far more data than this machine can collect. Because
:class:`~openarm_gym.control.bimanual_tray.BimanualTrayCarry` turns one tray pose
into both arms' targets, the only thing left to learn is the ``(roll, pitch)``
tilt -- so behaviour cloning from a few thousand frames is enough.

**Two frames, not one.** The expert's label is a PD law over the ball's position
*and velocity*. A single image cannot show velocity, so a single-frame policy can
at best imitate the P term and would be structurally unable to reproduce the
damping that keeps the ball from overshooting. Each camera therefore sees a stack
of :data:`FRAME_STACK` frames spaced :data:`FRAME_STRIDE` control steps apart.

**Spatial softmax, not a flattened feature map.** The task is "where is the
ball", i.e. localisation, and a spatial-softmax head returns exactly that: the
expected ``(x, y)`` of each channel's activation. It is also tiny -- two numbers
per channel instead of a 6x7 map -- which keeps the head small enough to train on
a few thousand samples without overfitting the background.

**The estimator is a detector, and is best trained like one.** In ``estimator``
mode the network answers "where is the ball on the tray, and how fast is it
moving". That is *static perception*: the control is already handled by
:func:`~openarm_gym.control.bimanual_tray.balance_law`. Training it on an
expert's rollouts is what creates the coverage problem, because a stabilising
controller keeps the ball centred and 90% of the frames are the same picture --
measured RMS ball offset of 10.5 mm against a 75 mm half-tray, and a closed-loop
sight error of 72 mm against 4.6 mm on the expert's own distribution.

``scripts/record_tray_detector.py`` samples the ball's position and velocity
**uniformly** instead, so coverage is uniform by construction and there is no
policy generating the distribution for covariate shift to act on. Those files
store each sample's frame stack explicitly, because the samples are independent
rather than consecutive.

**The velocity is differenced, not regressed.** The network has a velocity head
and it does not work: measured 11.7 mm/s RMSE against 12.7 mm/s for a predictor
that ignores the images and always outputs the training mean. That is not a
training failure, it is an observability one -- the ball moves at about 33 mm/s,
so over the 40 ms between stacked frames it travels 1.3 mm, which at the overhead
camera's 10 mm per pixel is a **tenth of a pixel**. The velocity is simply not in
the input.

Differencing the *position* estimate over :data:`DIFF_SPAN` control steps and
smoothing it recovers the damping instead, and it is measured to be enough:
against the true ball state the classical loop scores 6/6 on held-out plans, with
a dead velocity channel it scores 4/6, and with velocity differenced from a
position estimate carrying the policy's own 4.2 mm of error it scores 6/6 again
-- with the ball held *closer* to centre (3.3 cm peak) than the exact-state
controller manages (4.3 cm), because differencing low-passes the estimate.

The span matters and the filter matters: differencing over 2 steps unsmoothed is
3/6, because at that spacing the displacement is smaller than the position
error and the derivative is mostly noise.

**Two ways to close the loop, and the difference is not cosmetic.** The network
has two heads, and :class:`TrayVisionController` can be driven by either:

``mode="tilt"``
    Regress the expert's ``(roll, pitch)`` directly -- textbook behaviour cloning.
    Measured here: it tracks the expert to 5.6 mrad open loop, and then fails
    every closed-loop episode. The reason is visible in the predictions, which
    come out at roughly **half** the magnitude of the labels: an MSE regressor on
    a small dataset shrinks toward the mean, and half the loop gain is a
    different controller, not a slightly worse one.

``mode="estimator"`` (the default)
    Regress the ball's tray-frame position *and velocity*, then hand them to
    :func:`~openarm_gym.control.bimanual_tray.balance_law` -- the same arithmetic
    the classical controller runs. The gains stay exact by construction, the
    shrinkage lands on the state estimate where it belongs, and the error is
    readable in millimetres instead of milliradians. This is also what
    requirement 2 of the project actually asks for: a control algorithm applied,
    with vision supplying the state, rather than an end-to-end regression.

The estimate is why the failure mode is diagnosable at all: a bad closed-loop
result splits cleanly into "did not see the ball" and "did not act on it".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..control.bimanual_tray import balance_law

#: Frames per camera in one observation, and their spacing in control steps.
#: At 50 Hz a stride of 2 is 40 ms, which is long enough for a rolling ball to
#: move a measurable number of pixels and short enough to stay a local estimate.
FRAME_STACK = 2
FRAME_STRIDE = 2

#: Targets are scaled to O(1) before the loss sees them: tilt by the controller's
#: own clamp, ball offset by roughly the tray's half-width. Raw values are
#: 0.01-0.05, and an MSE on those is dominated by float noise in the optimiser.
TILT_SCALE = 0.20
BALL_SCALE = 0.10
#: Ball speed in the tray frame rarely exceeds this, so it scales the velocity
#: half of the state target to the same O(1) range as the position half.
VELOCITY_SCALE = 0.40

#: How the controller gets the velocity the balance law damps with. Differencing
#: the position estimate is the default and the measured reason is below.
DIFF_SPAN = 5
DIFF_ALPHA = 0.3


class SpatialSoftmax(nn.Module):
    """Return the expected pixel coordinate of each channel's activation."""

    def __init__(self, height: int, width: int) -> None:
        """Cache the normalised coordinate grids the expectation runs over."""
        super().__init__()
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("grid_x", xs.reshape(1, 1, -1), persistent=False)
        self.register_buffer("grid_y", ys.reshape(1, 1, -1), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` activations to ``(B, 2C)`` keypoints."""
        batch, channels = features.shape[:2]
        weights = torch.softmax(features.reshape(batch, channels, -1), dim=-1)
        x = (weights * self.grid_x).sum(-1)
        y = (weights * self.grid_y).sum(-1)
        return torch.cat([x, y], dim=1)


class CameraEncoder(nn.Module):
    """Four strided convolutions into a spatial-softmax keypoint vector."""

    def __init__(self, in_channels: int, image_size: tuple[int, int], width: int = 32) -> None:
        """Build the stack and size its softmax grid from ``image_size``."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=1, padding=1),
        )
        # Infer the feature-map size rather than assuming image_size // 8: a
        # padded stride-2 convolution rounds *up*, so 84 becomes 42, 21, 11 --
        # not 10 -- and the softmax grid has to match exactly.
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, in_channels, *image_size))
        self.softmax = SpatialSoftmax(probe.shape[2], probe.shape[3])
        self.out_dim = 2 * width

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, H, W)`` images to ``(B, 2 * width)`` keypoints."""
        return self.softmax(self.conv(images))


class TrayVisionPolicy(nn.Module):
    """Images plus proprioception to a ``(roll, pitch)`` tray tilt setpoint.

    Also predicts the ball's tray-frame position and velocity from the same
    features. That head is not a diagnostic afterthought: in the default
    ``estimator`` mode it is the head that actually closes the loop, through the
    classical balance law. Training both heads costs almost nothing and lets the
    two control modes be compared on identical weights.
    """

    def __init__(
        self,
        cameras: tuple[str, ...],
        image_size: tuple[int, int],
        proprio_dim: int,
        *,
        width: int = 32,
        hidden: int = 256,
        frame_stack: int = FRAME_STACK,
    ) -> None:
        """Build one encoder per camera and the shared setpoint head."""
        super().__init__()
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)
        self.frame_stack = frame_stack
        self.encoders = nn.ModuleDict(
            {
                name: CameraEncoder(3 * frame_stack, self.image_size, width)
                for name in self.cameras
            }
        )
        feature_dim = sum(e.out_dim for e in self.encoders.values()) + proprio_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.tilt_head = nn.Linear(hidden, 2)
        #: Four outputs: the ball's in-plane position and velocity in the tray
        #: frame, scaled by BALL_SCALE and VELOCITY_SCALE respectively.
        self.ball_head = nn.Linear(hidden, 4)

    def forward(
        self, images: dict[str, torch.Tensor], proprio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tilt, ball state)`` predictions, both in scaled units.

        ``tilt`` is ``(B, 2)`` and ``ball state`` is ``(B, 4)`` -- position then
        velocity.
        """
        features = [self.encoders[name](images[name]) for name in self.cameras]
        features.append(proprio)
        hidden = self.trunk(torch.cat(features, dim=1))
        return self.tilt_head(hidden), self.ball_head(hidden)


def augment(images: dict[str, torch.Tensor], gain: float, read: float) -> None:
    """Apply the camera sensor model to a batch, in place.

    Mirrors :class:`~openarm_gym.vision.CameraRig`'s model -- a per-frame gain and
    per-pixel read noise -- but in the network's ``[-0.5, 0.5]`` units and on
    whatever device the batch is already on, which is far cheaper than rendering
    noisy pixels during collection. It lives here rather than in a training
    script because both detectors train against the same sensor model, and two
    copies of it would drift.
    """
    for tensor in images.values():
        if gain > 0.0:
            scale = 1.0 + torch.randn(
                tensor.shape[0], 1, 1, 1, device=tensor.device
            ) * gain
            tensor.add_(0.5).mul_(scale).sub_(0.5)
        if read > 0.0:
            tensor.add_(torch.randn_like(tensor) * (read / 255.0))
        tensor.clamp_(-0.5, 0.5)


def prepare_images(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert stacked uint8 frames to the network's float input layout.

    Takes ``(B, S, H, W, 3)`` and returns ``(B, 3 * S, H, W)`` in ``[-0.5, 0.5]``.
    """
    tensor = torch.as_tensor(frames)
    if tensor.ndim != 5 or tensor.shape[-1] != 3:
        raise ValueError(
            f"expected (B, S, H, W, 3) stacked frames, got {tuple(tensor.shape)}"
        )
    batch, stack, height, width, channels = tensor.shape
    out = tensor.permute(0, 1, 4, 2, 3).reshape(batch, stack * channels, height, width)
    return out.float().div_(255.0).sub_(0.5)


class TrayVisionDataset(torch.utils.data.Dataset):
    """Frames and labels from the recorded episodes, held in RAM as uint8.

    The episodes are loaded eagerly, because the alternative -- decompressing an
    ``.npz`` per batch -- is slower than the whole training run. uint8 is what
    makes that affordable: 28 episodes of two 84x112 cameras is about 700 MB as
    uint8 and 2.8 GB as float32, and this machine has 3-5 GB spare.
    """

    def __init__(
        self,
        root: Path,
        cameras: tuple[str, ...],
        *,
        episodes: list[Path] | None = None,
        frame_stack: int = FRAME_STACK,
        frame_stride: int = FRAME_STRIDE,
    ) -> None:
        """Load every episode under ``root`` and index its valid samples."""
        self.cameras = tuple(cameras)
        self.frame_stack = frame_stack
        self.frame_stride = frame_stride
        paths = sorted(episodes if episodes is not None else root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz episodes under {root}")

        self.pixels: list[dict[str, np.ndarray]] = []
        #: Per episode: True when the file stores each sample's frame stack
        #: explicitly, as (N, S, H, W, 3), instead of a trajectory to index into.
        #: Detector datasets are independent samples, so there is no "previous
        #: frame" to reach back to -- stacking them by stride would pair every
        #: sample with an unrelated one.
        self.prestacked: list[bool] = []
        self.labels: list[np.ndarray] = []
        self.ball: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []
        for episode, path in enumerate(paths):
            with np.load(path) as blob:
                frames = {n: blob[f"pixels_{n}"] for n in self.cameras}
                self.pixels.append(frames)
                stacked = {a.ndim == 5 for a in frames.values()}
                if len(stacked) != 1:
                    raise ValueError(f"{path}: cameras disagree on frame layout")
                self.prestacked.append(stacked.pop())
                self.labels.append(blob["labels"].astype(np.float32))
                # Position and velocity, each scaled to O(1), as one 4-vector.
                self.ball.append(
                    np.concatenate(
                        [
                            blob["ball_xy"].astype(np.float32) / BALL_SCALE,
                            blob["ball_vxy"].astype(np.float32) / VELOCITY_SCALE,
                        ],
                        axis=1,
                    )
                )
                self.proprio.append(blob["proprio"].astype(np.float32))
            self.index += [(episode, t) for t in range(len(self.labels[-1]))]

        flat = np.concatenate(self.proprio)
        self.proprio_mean = flat.mean(0)
        # A *floor*, not an epsilon. Adding 1e-6 to a near-constant channel still
        # divides by ~1e-6, so a joint that happens not to move in the recording
        # turns micrometre-scale drift into an input of order 10^3 -- and one
        # that differs between training and inference. The floor is in the units
        # of the channels (radians and metres), so 1 mm / 1 mrad of spread is the
        # least that counts as signal.
        self.proprio_std = np.maximum(flat.std(0), 1e-3)

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Return one sample's ``(frame stacks, proprio, tilt, ball)``."""
        episode, t = self.index[i]
        if self.prestacked[episode]:
            images = {name: self.pixels[episode][name][t] for name in self.cameras}
        else:
            # Clamp at the episode start rather than skipping those samples: the
            # first frames are the settled hold after the lift, so a repeated
            # frame there implies zero ball velocity, which is very nearly true.
            offsets = [
                max(0, t - k * self.frame_stride)
                for k in reversed(range(self.frame_stack))
            ]
            images = {
                name: self.pixels[episode][name][offsets] for name in self.cameras
            }
        proprio = (self.proprio[episode][t] - self.proprio_mean) / self.proprio_std
        return (
            images,
            proprio.astype(np.float32),
            self.labels[episode][t] / TILT_SCALE,
            self.ball[episode][t],
        )


def collate(batch: list) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack samples, converting the frame stacks to network input layout."""
    names = batch[0][0].keys()
    images = {
        name: prepare_images(np.stack([b[0][name] for b in batch])) for name in names
    }
    proprio = torch.as_tensor(np.stack([b[1] for b in batch]))
    tilt = torch.as_tensor(np.stack([b[2] for b in batch]))
    ball = torch.as_tensor(np.stack([b[3] for b in batch]))
    return images, proprio, tilt, ball


class TrayVisionController:
    """Run a trained policy as a drop-in for the classical balancer.

    Holds the frame history the network expects, so it can be handed straight to
    :func:`~openarm_gym.control.tray_task.run_plan` as its ``tilt_policy``.
    Call :meth:`reset` between episodes or the first frames of a new episode are
    stacked against the last frames of the previous one.

    ``mode`` picks which head closes the loop -- see the module docstring for why
    ``"estimator"`` is the default and ``"tilt"`` is kept as the baseline.
    """

    def __init__(
        self,
        policy: TrayVisionPolicy,
        proprio_mean: np.ndarray,
        proprio_std: np.ndarray,
        *,
        device: str = "cpu",
        frame_stride: int = FRAME_STRIDE,
        mode: str = "estimator",
        velocity_source: str = "difference",
        diff_span: int = DIFF_SPAN,
        diff_alpha: float = DIFF_ALPHA,
        control_hz: float = 50.0,
    ) -> None:
        """Bind a trained policy to its normalisation statistics.

        ``velocity_source`` selects where the balance law's damping term comes
        from: ``"difference"`` differences the position estimate over
        ``diff_span`` control steps and smooths it with a first-order filter of
        gain ``diff_alpha``; ``"head"`` uses the network's velocity output, which
        is kept only so the comparison can be re-run.

        ``control_hz`` must match the rate of the controller being driven -- it
        is the scale factor on the differenced velocity, so getting it wrong
        scales the damping term by the ratio without any other symptom.
        """
        if mode not in ("estimator", "tilt"):
            raise ValueError(f"mode must be 'estimator' or 'tilt', got {mode!r}")
        if velocity_source not in ("difference", "head"):
            raise ValueError(
                f"velocity_source must be 'difference' or 'head', got {velocity_source!r}"
            )
        self.policy = policy.to(device).eval()
        self.device = device
        self.mode = mode
        self.proprio_mean = np.asarray(proprio_mean, dtype=np.float32)
        self.proprio_std = np.asarray(proprio_std, dtype=np.float32)
        self.frame_stride = frame_stride
        self.velocity_source = velocity_source
        self.diff_span = int(diff_span)
        self.diff_alpha = float(diff_alpha)
        self.control_hz = float(control_hz)
        self._history: dict[str, list[np.ndarray]] = {}
        self._positions: list[np.ndarray] = []
        #: Last ball position estimate in the tray frame, metres.
        self.ball_estimate = np.zeros(2, dtype=np.float32)
        #: Last ball velocity estimate in the tray frame, m/s.
        self.velocity_estimate = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        """Forget the frame history and the differencing window."""
        self._history = {}
        self._positions = []
        self.ball_estimate = np.zeros(2, dtype=np.float32)
        self.velocity_estimate = np.zeros(2, dtype=np.float32)

    def _differenced_velocity(self, position: np.ndarray) -> np.ndarray:
        """Update and return the filtered derivative of the position estimate.

        Returns the previous value until the window has filled, which is a few
        control steps of no damping at the start of an episode -- harmless, since
        the tray is still settling into the lift and the ball is barely moving.
        """
        self._positions.append(np.asarray(position, dtype=np.float64))
        if len(self._positions) > self.diff_span + 1:
            del self._positions[0]
        if len(self._positions) > self.diff_span:
            raw = (
                (self._positions[-1] - self._positions[0])
                * self.control_hz
                / self.diff_span
            )
            self.velocity_estimate = (
                self.diff_alpha * raw + (1.0 - self.diff_alpha) * self.velocity_estimate
            )
        return self.velocity_estimate

    def _stack(self, name: str, frame: np.ndarray) -> np.ndarray:
        """Append a frame and return the strided stack the network wants."""
        span = 1 + self.frame_stride * (self.policy.frame_stack - 1)
        history = self._history.setdefault(name, [])
        history.append(frame)
        if len(history) > span:
            del history[0]
        while len(history) < span:
            history.insert(0, history[0])
        return np.stack(
            [history[-1 - k * self.frame_stride] for k in reversed(range(self.policy.frame_stack))]
        )

    @torch.no_grad()
    def __call__(self, obs: dict) -> np.ndarray:
        """Return the ``(roll, pitch)`` setpoint for one observation."""
        images = {
            name: prepare_images(self._stack(name, obs["pixels"][name])[None]).to(self.device)
            for name in self.policy.cameras
        }
        flat = np.concatenate([obs["qpos"], obs["tray_pos"], obs["tray_quat"]])
        proprio = (flat.astype(np.float32) - self.proprio_mean) / self.proprio_std
        tilt, state = self.policy(
            images, torch.as_tensor(proprio[None]).to(self.device)
        )
        state = state[0].cpu().numpy()
        self.ball_estimate = state[:2] * BALL_SCALE
        if self.velocity_source == "head":
            self.velocity_estimate = state[2:] * VELOCITY_SCALE
        else:
            self._differenced_velocity(self.ball_estimate)
        if self.mode == "tilt":
            return tilt[0].cpu().numpy().astype(np.float64) * TILT_SCALE
        return balance_law(self.ball_estimate, self.velocity_estimate)


def load_controller(
    checkpoint: Path,
    *,
    device: str = "cpu",
    mode: str = "estimator",
    velocity_source: str = "difference",
    control_hz: float = 50.0,
) -> TrayVisionController:
    """Rebuild a controller from a training checkpoint."""
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = TrayVisionPolicy(
        tuple(blob["cameras"]),
        tuple(blob["image_size"]),
        int(blob["proprio_dim"]),
        width=int(blob["width"]),
        frame_stack=int(blob["frame_stack"]),
    )
    policy.load_state_dict(blob["state_dict"])
    return TrayVisionController(
        policy,
        blob["proprio_mean"],
        blob["proprio_std"],
        device=device,
        frame_stride=int(blob["frame_stride"]),
        mode=mode,
        velocity_source=velocity_source,
        control_hz=control_hz,
    )
