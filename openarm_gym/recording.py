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

"""Record scripted-expert rollouts as a LeRobotDataset.

The dataset schema is the interchange format the wider ecosystem reads, and it
is what Enactic's own ``openarm-dataset-convert`` targets, so demonstrations
collected here drop into the same tooling as real-robot recordings.

Two collection details matter more than the format:

*Exploration noise with clean labels.* Each step executes the expert's action
plus Gaussian joint noise, but records the expert's *clean* action at the state
the arm actually reached. Because the experts re-plan every step, that label is
the correction back onto the plan. Training on it teaches recovery; training on
noiseless rollouts teaches a single trajectory.

*Successful episodes only.* A failed scripted rollout is not a demonstration of
anything, so episodes are filtered on the task's success predicate before they
are written.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np

#: Joint names, in the 16-value driver order.
JOINT_NAMES = [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"] + [
    f"left_joint{i}" for i in range(1, 8)
] + ["left_gripper"]


def _import_lerobot():
    """Return ``LeRobotDataset``, trying both module layouts."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # older releases
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


@dataclasses.dataclass
class EpisodeStats:
    """Outcome of one recorded rollout."""

    seed: int
    success: bool
    steps: int
    total_reward: float
    instruction: str


def build_features(
    camera_names: tuple[str, ...],
    image_size: tuple[int, int],
    use_videos: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot feature schema for an OpenArm recording.

    Datasets this small are cheaper stored as images than as video: encoding
    dominates collection time, and decoding dominates the training dataloader.
    """
    height, width = image_size
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        # The channel the force-conditioning ablation switches on and off.
        "observation.torque": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": list(JOINT_NAMES),
        },
    }
    for name in camera_names:
        features[f"observation.images.{name}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def rollout(
    env,
    expert,
    seed: int,
    *,
    action_noise: float = 0.0,
    on_frame: Callable[[dict[str, Any], np.ndarray], None] | None = None,
) -> EpisodeStats:
    """Run one episode, optionally handing each frame to a recorder.

    The frame passed to ``on_frame`` pairs the observation with the expert's
    clean action; the environment is stepped with a noisy version of it.
    """
    obs, _info = env.reset(seed=seed)
    expert.reset(seed=seed)
    instruction = env.instruction

    total = 0.0
    steps = 0
    success = False
    while True:
        action = expert.act()
        if on_frame is not None:
            on_frame(obs, action)
        obs, reward, terminated, truncated, info = env.step(
            expert.perturb(action, action_noise)
        )
        total += reward
        steps += 1
        if terminated or truncated:
            success = bool(info["is_success"])
            break

    return EpisodeStats(
        seed=seed,
        success=success,
        steps=steps,
        total_reward=total,
        instruction=instruction,
    )


class DatasetRecorder:
    """Collect successful expert rollouts into a LeRobotDataset."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        fps: int,
        camera_names: tuple[str, ...],
        image_size: tuple[int, int],
        use_videos: bool = True,
        image_writer_threads: int = 4,
    ) -> None:
        """Open a dataset on disk, creating it if it is not already there.

        Reopening rather than failing makes collection resumable, so a long run
        can be taken in short chunks and survives being interrupted.
        """
        lerobot_dataset = _import_lerobot()
        self.root = Path(root)
        self.camera_names = tuple(camera_names)

        if (self.root / "meta" / "info.json").exists():
            self.dataset = lerobot_dataset(repo_id, root=self.root)
            self.resumed_from = self.dataset.num_episodes
        else:
            self.dataset = lerobot_dataset.create(
                repo_id=repo_id,
                fps=fps,
                root=self.root,
                features=build_features(self.camera_names, image_size, use_videos),
                use_videos=use_videos,
                # Frames are written one PNG per camera per step before being
                # encoded to video. Doing that synchronously dominates
                # collection time, so hand it to a thread pool.
                image_writer_threads=image_writer_threads,
                # Flush episode metadata every episode. The default batches ten
                # of them, and a run interrupted mid-batch leaves a half-written
                # metadata parquet that no longer opens -- which loses the whole
                # dataset, not just the unflushed episodes.
                metadata_buffer_size=1,
            )
            self.resumed_from = 0

        self.episodes: list[EpisodeStats] = []

    def _frame(self, obs: dict[str, Any], action: np.ndarray) -> dict[str, Any]:
        """Map an environment observation onto the dataset schema."""
        frame: dict[str, Any] = {
            "observation.state": np.asarray(obs["agent_pos"], dtype=np.float32),
            "observation.velocity": np.asarray(obs["agent_vel"], dtype=np.float32),
            "observation.torque": np.asarray(obs["agent_torque"], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
        pixels = obs.get("pixels", {})
        for name in self.camera_names:
            frame[f"observation.images.{name}"] = pixels[name]
        return frame

    def record(
        self,
        env,
        expert,
        seed: int,
        *,
        action_noise: float = 0.0,
        max_steps: int | None = None,
    ) -> EpisodeStats:
        """Run one episode and write it only if the expert succeeded.

        Frames are held in a compact list of ``uint8`` arrays and handed to the
        dataset only once the episode is known to be a success. Streaming into
        the dataset instead makes it write one PNG per camera per step -- 1800
        files for a 900-step rollout -- and then delete them all again whenever
        the rollout fails, which is most of the cost of collection.
        """
        obs, _info = env.reset(seed=seed)
        expert.reset(seed=seed)
        instruction = env.instruction
        limit = max_steps or env.MAX_STEPS

        buffer: list[dict[str, Any]] = []
        total = 0.0
        steps = 0
        success = False
        while True:
            action = expert.act()
            buffer.append(self._frame(obs, action))
            obs, reward, terminated, truncated, info = env.step(
                expert.perturb(action, action_noise)
            )
            total += reward
            steps += 1
            if terminated or truncated or steps >= limit:
                success = bool(info["is_success"])
                break

        stats = EpisodeStats(
            seed=seed,
            success=success,
            steps=steps,
            total_reward=total,
            instruction=instruction,
        )

        if success:
            for frame in buffer:
                # lerobot carries the language instruction as a frame field.
                self.dataset.add_frame({**frame, "task": instruction})
            self.dataset.save_episode()
            self.episodes.append(stats)

        return stats
