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

"""See where the tray's handles are, so the grasp does not have to be told.

``BimanualTrayCarry.grasp`` reads the handle posts straight out of the simulator.
That made the *balancing* half of this task a vision problem while the *grasping*
half quietly stayed a privileged-state one -- and grasping is the half where the
five requirements are hardest, since it is the only part that touches the object.
This module closes that gap: a small convolutional detector that looks at the
scene from the arms' retracted pose and returns both handle positions in world
coordinates, which is exactly the input ``grasp`` needs.

It is a **separate, smaller network** from the ball policy rather than another
head on it, for reasons that are specific rather than stylistic:

* The ball estimator runs every control step on a two-frame stack, because the
  ball moves. The handles do not move before they are grasped, so this runs
  **once per episode, on one frame**. Sharing an encoder would pay the stack's
  cost forever to serve a question asked once.
* Their training distributions have nothing in common. The ball detector is
  trained with the tray held in the air mid-carry; this one is trained with the
  arms retracted and the tray on the table, which is the only view it will ever
  be asked about.
* It keeps the comparison clean. The privileged path stays available, so
  "grasped from pixels" and "grasped from the simulator" can be measured on the
  same layouts.

There is deliberately **no proprioception input**. The arms are at the same
retracted pose in every sample, so the channel carries no information, and a
normaliser dividing by a near-zero standard deviation is a known failure in this
repository.

The output is world coordinates, which is well-posed here because both cameras
are fixed to the world: the scene attaches them in ``worldbody``, so a pixel
maps to a ray in a frame that never moves. On real hardware this is the extrinsic
calibration step, and it is where the sim-to-real caveat for this piece lives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .tray_vision import CameraEncoder, prepare_images

#: The handle positions the scene is authored with, as ``(left, right)`` world
#: coordinates. Predictions are offsets from these, so the network starts near
#: the answer and the loss is not dominated by a constant it has to relearn.
HANDLE_NOMINAL = np.array(
    [[0.35, 0.15, 0.44], [0.35, -0.15, 0.44]], dtype=np.float32
)
#: Metres of offset that correspond to one unit of network output. Roughly the
#: layout envelope, so targets arrive at the loss as O(1) numbers.
HANDLE_SCALE = 0.05

SIDES = ("left", "right")


def encode_handles(handles: dict[str, np.ndarray]) -> np.ndarray:
    """Pack a ``{side: xyz}`` mapping into the network's 6-vector target."""
    stacked = np.stack([np.asarray(handles[s], dtype=np.float32) for s in SIDES])
    return ((stacked - HANDLE_NOMINAL) / HANDLE_SCALE).reshape(6)


def decode_handles(prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Undo :func:`encode_handles`, returning world positions per side."""
    offsets = np.asarray(prediction, dtype=np.float64).reshape(2, 3) * HANDLE_SCALE
    world = offsets + HANDLE_NOMINAL
    return {side: world[i] for i, side in enumerate(SIDES)}


class HandleDetector(nn.Module):
    """One frame per camera to both handle positions, in world coordinates."""

    def __init__(
        self,
        cameras: tuple[str, ...],
        image_size: tuple[int, int],
        *,
        width: int = 32,
        hidden: int = 128,
    ) -> None:
        """Build one single-frame encoder per camera and the shared head."""
        super().__init__()
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)
        self.encoders = nn.ModuleDict(
            {name: CameraEncoder(3, self.image_size, width) for name in self.cameras}
        )
        feature_dim = sum(e.out_dim for e in self.encoders.values())
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        #: Six outputs: left xyz then right xyz, as scaled offsets from nominal.
        self.head = nn.Linear(hidden, 6)

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return ``(B, 6)`` scaled handle offsets."""
        features = [self.encoders[name](images[name]) for name in self.cameras]
        return self.head(self.trunk(torch.cat(features, dim=1)))


class HandleDataset(torch.utils.data.Dataset):
    """Frames and handle labels, loaded from the recorder's ``.npz`` files.

    Held in RAM as uint8 and converted per batch. The whole dataset is a few
    thousand single frames -- far smaller than the ball policy's stacks, which is
    the other reason this network is worth keeping separate.
    """

    def __init__(self, paths: list[Path], cameras: tuple[str, ...]) -> None:
        """Load and concatenate every file."""
        self.cameras = tuple(cameras)
        frames: dict[str, list[np.ndarray]] = {n: [] for n in self.cameras}
        labels = []
        for path in paths:
            with np.load(path) as blob:
                for name in self.cameras:
                    frames[name].append(blob[f"pixels_{name}"])
                labels.append(blob["handles"])
        self.pixels = {n: np.concatenate(v) for n, v in frames.items()}
        self.labels = np.concatenate(labels).astype(np.float32)

    def __len__(self) -> int:
        """Return how many samples the dataset holds."""
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Return one sample's frames per camera and its label."""
        return {n: v[i] for n, v in self.pixels.items()}, self.labels[i]


def collate(batch: list) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Stack samples into the network's batch layout."""
    # prepare_images speaks the stacked-frame layout the ball policy uses, so a
    # single frame goes in as a stack of one rather than growing a second path
    # through it.
    images = {
        name: prepare_images(np.stack([sample[0][name] for sample in batch])[:, None])
        for name in batch[0][0]
    }
    labels = torch.from_numpy(np.stack([sample[1] for sample in batch]))
    return images, labels


class HandleEstimator:
    """Runtime wrapper: an observation in, ``{side: world xyz}`` out.

    Built to be handed straight to :meth:`BimanualTrayCarry.grasp` as its
    ``handles`` argument, which is the whole interface between this module and
    the controller.
    """

    def __init__(self, detector: HandleDetector, *, device: str = "cpu") -> None:
        """Put the detector in eval mode on ``device``."""
        self.detector = detector.to(device).eval()
        self.device = device
        #: The last estimate produced, for reporting its error against truth.
        self.estimate: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, obs: dict) -> dict[str, np.ndarray]:
        """Estimate both handle positions from one observation's pixels."""
        missing = [n for n in self.detector.cameras if n not in obs.get("pixels", {})]
        if missing:
            raise ValueError(
                f"the handle detector needs cameras {missing}, which this "
                "observation does not carry; build the controller with "
                "camera_names covering the checkpoint's cameras"
            )
        images = {
            name: prepare_images(obs["pixels"][name][None, None]).to(self.device)
            for name in self.detector.cameras
        }
        prediction = self.detector(images).cpu().numpy()[0]
        self.estimate = decode_handles(prediction)
        return self.estimate

    def error(self, truth: dict[str, np.ndarray]) -> float:
        """Mean distance between the last estimate and the true positions, m."""
        if not self.estimate:
            return float("nan")
        return float(
            np.mean([np.linalg.norm(self.estimate[s] - truth[s]) for s in SIDES])
        )


def load_handle_detector(
    checkpoint: Path, *, device: str = "cpu"
) -> HandleEstimator:
    """Rebuild an estimator from a training checkpoint."""
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    detector = HandleDetector(
        tuple(blob["cameras"]),
        tuple(blob["image_size"]),
        width=int(blob["width"]),
    )
    detector.load_state_dict(blob["state_dict"])
    return HandleEstimator(detector, device=device)
