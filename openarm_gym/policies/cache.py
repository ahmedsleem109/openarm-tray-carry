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

"""An in-memory cache of a recorded dataset.

Decoding PNGs in the dataloader held training to 0.5 steps/s against 5.2 on
synthetic tensors -- the GPU was idle more than 90% of the time. These datasets
are small enough to sit in RAM as ``uint8`` (8.5k frames across two cameras is
about 630 MB), so they are read once, cached to disk in tensor form, and served
from memory thereafter.

Action chunks are assembled here rather than by LeRobot's ``delta_timestamps``,
and are clamped to the end of their own episode: the last action repeats rather
than bleeding into the next episode, which would teach the policy to drive from
one task layout into another.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

#: Bump when the cache layout changes, so stale files are rebuilt.
CACHE_VERSION = 1


def _open_raw(root: Path, repo_id: str):
    """Open a recorded dataset without action chunking."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # older releases
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id, root=root)


class CachedChunkDataset(Dataset):
    """Serve ``(images, state, torque, action_chunk)`` tuples from memory."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        chunk_size: int,
        *,
        cache_path: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        """Load the dataset into memory, building the cache if needed."""
        self.root = Path(root)
        self.chunk_size = chunk_size
        cache = Path(cache_path) if cache_path else self.root / "tensor_cache.pt"

        payload = None
        if cache.exists():
            candidate = torch.load(cache, map_location="cpu", weights_only=False)
            if candidate.get("version") == CACHE_VERSION:
                payload = candidate
                if verbose:
                    print(f"  cache hit: {cache}")

        if payload is None:
            payload = self._build(repo_id, verbose=verbose)
            payload["version"] = CACHE_VERSION
            torch.save(payload, cache)
            if verbose:
                print(f"  cache written: {cache}")

        self.images: dict[str, torch.Tensor] = payload["images"]
        self.state: torch.Tensor = payload["state"]
        self.torque: torch.Tensor = payload["torque"]
        self.actions: torch.Tensor = payload["actions"]
        self.episode_index: torch.Tensor = payload["episode_index"]
        self.stats: dict = payload["stats"]
        self.camera_keys: list[str] = payload["camera_keys"]
        self.num_episodes: int = payload["num_episodes"]

        # Last frame index of each frame's own episode, for clamping chunks.
        episodes = self.episode_index
        last = torch.zeros_like(episodes)
        end = len(episodes) - 1
        for i in range(len(episodes) - 1, -1, -1):
            if i < len(episodes) - 1 and episodes[i] != episodes[i + 1]:
                end = i
            last[i] = end
        self.episode_end = last

    def _build(self, repo_id: str, *, verbose: bool) -> dict:
        """Read every frame once and stack it into tensors."""
        raw = _open_raw(self.root, repo_id)
        camera_keys = sorted(
            key for key in raw.features if key.startswith("observation.images.")
        )
        total = raw.num_frames
        if verbose:
            print(f"  caching {total} frames from {raw.num_episodes} episodes...")

        first = raw[0]
        height, width = first[camera_keys[0]].shape[-2:]
        images = {
            key: torch.empty(total, 3, height, width, dtype=torch.uint8)
            for key in camera_keys
        }
        state = torch.empty(total, first["observation.state"].shape[-1])
        torque = torch.empty(total, first["observation.torque"].shape[-1])
        actions = torch.empty(total, first["action"].shape[-1])
        episode_index = torch.empty(total, dtype=torch.long)

        for i in range(total):
            item = raw[i]
            for key in camera_keys:
                # Frames arrive as float in [0, 1]; store bytes to save memory.
                images[key][i] = (item[key] * 255.0).round().clamp(0, 255).to(
                    torch.uint8
                )
            state[i] = item["observation.state"]
            torque[i] = item["observation.torque"]
            actions[i] = item["action"]
            episode_index[i] = int(item["episode_index"])
            if verbose and (i + 1) % 2000 == 0:
                print(f"    {i + 1}/{total}", flush=True)

        return {
            "images": images,
            "state": state,
            "torque": torque,
            "actions": actions,
            "episode_index": episode_index,
            "stats": raw.meta.stats,
            "camera_keys": camera_keys,
            "num_episodes": raw.num_episodes,
        }

    def __len__(self) -> int:
        """Return the number of frames."""
        return self.state.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one training sample."""
        end = int(self.episode_end[index])
        stop = min(index + self.chunk_size, end + 1)
        chunk = self.actions[index:stop]
        if chunk.shape[0] < self.chunk_size:
            # Hold the final action rather than running past the episode.
            pad = chunk[-1:].expand(self.chunk_size - chunk.shape[0], -1)
            chunk = torch.cat([chunk, pad], dim=0)

        return {
            "images": torch.stack([self.images[key][index] for key in self.camera_keys]),
            "observation.state": self.state[index],
            "observation.torque": self.torque[index],
            "action": chunk,
        }
