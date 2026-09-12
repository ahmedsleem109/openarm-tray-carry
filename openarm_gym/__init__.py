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

"""Gymnasium environments for the OpenArm v2 MuJoCo scenes."""

from gymnasium.envs.registration import register

from .env import DRIVER_DIM, OpenArmEnv
from .ik import PoseController
from .registry import CONTACT_RICH, CONTROL_TASKS, TASKS, make
from .tasks.move_puck import MovePuckEnv
from .tasks.peg_socket import PegSocketEnv
from .tasks.valve import ValveEnv

__all__ = [
    "CONTACT_RICH",
    "CONTROL_TASKS",
    "DRIVER_DIM",
    "MovePuckEnv",
    "OpenArmEnv",
    "PegSocketEnv",
    "PoseController",
    "TASKS",
    "ValveEnv",
    "make",
]

for _name, (_env_cls, _) in TASKS.items():
    register(
        id=f"OpenArm/{_name}-v0",
        entry_point=f"openarm_gym.tasks.{_name}:{_env_cls.__name__}",
        max_episode_steps=_env_cls.MAX_STEPS,
    )
