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

"""Task name to environment and scripted expert."""

from __future__ import annotations

from .experts.move_puck import MovePuckExpert
from .experts.peg_socket import PegSocketExpert
from .experts.valve import ValveExpert
from .tasks.move_puck import MovePuckEnv
from .tasks.peg_socket import PegSocketEnv
from .tasks.valve import ValveEnv

#: Every task, by short name.
TASKS = {
    "peg_socket": (PegSocketEnv, PegSocketExpert),
    "move_puck": (MovePuckEnv, MovePuckExpert),
    "valve": (ValveEnv, ValveExpert),
}

#: Tasks where contact feedback should plausibly matter, and the control task
#: where it should not. Used by the force-conditioning ablation.
CONTACT_RICH = ("peg_socket", "valve")
CONTROL_TASKS = ("move_puck",)


def make(name: str, **kwargs):
    """Build a task environment and its scripted expert by name."""
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}; known tasks: {sorted(TASKS)}")
    env_cls, expert_cls = TASKS[name]
    env = env_cls(**kwargs)
    return env, expert_cls(env)
