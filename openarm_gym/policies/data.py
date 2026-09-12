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

"""Loading OpenArm demonstrations for policy training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

#: ImageNet statistics, matching the pretrained ResNet trunk.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dataset(root: str | Path, repo_id: str, chunk_size: int):
    """Open a recorded LeRobotDataset, serving action chunks.

    ``delta_timestamps`` is what makes each sample carry the next
    ``chunk_size`` actions rather than a single step, which is the supervision
    an action-chunking policy needs.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # older releases
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    fps = 50
    meta_path = root / "meta" / "info.json"
    if meta_path.exists():
        import json

        fps = int(json.loads(meta_path.read_text())["fps"])

    delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)


def camera_keys(dataset) -> list[str]:
    """Return the dataset's camera feature keys, in a stable order."""
    return sorted(
        key for key in dataset.features if key.startswith("observation.images.")
    )


class Normalizer:
    """Standardise proprioception and actions using dataset statistics."""

    def __init__(self, stats: dict, keys: tuple[str, ...], device: str) -> None:
        """Cache per-key mean and standard deviation as tensors."""
        self.mean: dict[str, Tensor] = {}
        self.std: dict[str, Tensor] = {}
        for key in keys:
            entry = stats[key]
            mean = torch.as_tensor(entry["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(entry["std"], dtype=torch.float32, device=device)
            self.mean[key] = mean.flatten()
            # A joint that never moves has zero spread; guard the divide.
            self.std[key] = torch.clamp(std.flatten(), min=1e-4)

    def normalize(self, key: str, value: Tensor) -> Tensor:
        """Standardise a tensor for one key."""
        return (value - self.mean[key]) / self.std[key]

    def denormalize(self, key: str, value: Tensor) -> Tensor:
        """Undo :meth:`normalize`."""
        return value * self.std[key] + self.mean[key]


def normalize_images(images: Tensor) -> Tensor:
    """Apply ImageNet normalisation to ``(..., 3, H, W)`` images in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(3, 1, 1)
    return (images - mean) / std


def stack_cameras(batch: dict, keys: list[str]) -> Tensor:
    """Stack per-camera image tensors into ``(B, cameras, 3, H, W)``."""
    return torch.stack([batch[key] for key in keys], dim=1)
