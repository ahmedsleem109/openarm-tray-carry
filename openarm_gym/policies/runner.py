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

"""Run a trained action-chunking policy in closed loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .act import ACTPolicy
from .data import normalize_images


class PolicyRunner:
    """Drive an environment with a trained policy.

    Chunks are combined by temporal ensembling: every step the policy proposes
    the next ``chunk_size`` actions, and the action actually executed is an
    exponentially weighted average of all the predictions made for this
    timestep. Executing chunks open-loop instead leaves a visible discontinuity
    each time a new chunk starts.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        ensemble_weight: float = 0.01,
    ) -> None:
        """Load a checkpoint saved by ``scripts/train_policy.py``."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)

        self.config = payload["config"]
        self.cameras = payload["cameras"]
        self.camera_names = [key.rsplit(".", 1)[-1] for key in self.cameras]
        self.chunk_size = int(self.config["chunk_size"])
        self.use_torque = bool(self.config["use_torque"])
        self.ensemble_weight = ensemble_weight

        self.policy = ACTPolicy(**self.config, pretrained_vision=False).to(self.device)
        self.policy.load_state_dict(payload["state_dict"])
        self.policy.eval()

        self.stats = {
            key: (
                torch.tensor(value["mean"], device=self.device),
                torch.tensor(value["std"], device=self.device),
            )
            for key, value in payload["stats"].items()
        }
        self._pending: list[tuple[int, np.ndarray]] = []
        self._step = 0

    # ------------------------------------------------------------- plumbing

    def reset(self) -> None:
        """Clear the ensemble buffer between episodes."""
        self._pending = []
        self._step = 0

    def _normalize(self, key: str, value: np.ndarray) -> torch.Tensor:
        """Standardise one observation vector."""
        mean, std = self.stats[key]
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return ((tensor - mean) / std).unsqueeze(0)

    def _denormalize(self, key: str, value: torch.Tensor) -> np.ndarray:
        """Undo the action standardisation."""
        mean, std = self.stats[key]
        return (value * std + mean).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """Return the action to execute for this observation."""
        pixels = obs["pixels"]
        images = np.stack([pixels[name] for name in self.camera_names])
        tensor = torch.as_tensor(images, device=self.device).float().div_(255.0)
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        tensor = normalize_images(tensor)

        state = self._normalize("observation.state", obs["agent_pos"])
        torque = self._normalize("observation.torque", obs["agent_torque"])

        predicted, _mu, _logvar = self.policy(tensor, state, torque)
        chunk = self._denormalize("action", predicted)

        self._pending.append((self._step, chunk))
        # Only chunks that still cover the current step can contribute.
        self._pending = [
            (start, actions)
            for start, actions in self._pending
            if self._step - start < self.chunk_size
        ]

        votes = []
        weights = []
        for start, actions in self._pending:
            age = self._step - start
            votes.append(actions[age])
            weights.append(np.exp(-self.ensemble_weight * age))
        weights = np.asarray(weights)
        action = np.average(np.stack(votes), axis=0, weights=weights)

        self._step += 1
        return action.astype(np.float32)


def evaluate(
    env,
    runner: PolicyRunner,
    seeds: list[int],
) -> dict[str, Any]:
    """Run a policy over a set of layouts and return the outcome per seed."""
    outcomes = []
    lengths = []
    for seed in seeds:
        obs, _info = env.reset(seed=seed)
        runner.reset()
        steps = 0
        success = False
        while True:
            obs, _reward, terminated, truncated, info = env.step(runner.act(obs))
            steps += 1
            if terminated or truncated:
                success = bool(info["is_success"])
                break
        outcomes.append(success)
        lengths.append(steps)

    successes = int(np.sum(outcomes))
    return {
        "successes": successes,
        "episodes": len(seeds),
        "rate": successes / max(len(seeds), 1),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "outcomes": outcomes,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a success rate.

    Wilson rather than normal-approximation: with 50 evaluation episodes and
    rates near 0 or 1, the normal interval runs past the ends of the scale.
    """
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    spread = (
        z
        * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))
