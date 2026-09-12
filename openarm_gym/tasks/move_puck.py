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

"""Move a puck to a commanded goal, on the OpenArm Cell workspace.

The scene provides a 70x70x44 mm puck and two goal markers, green and blue.
Which goal is commanded varies per episode and is carried as a language
instruction, so the task is goal-conditioned rather than fixed.

The puck can be pushed or picked and placed. Both are legitimate, and the
scripted expert does both -- the resulting demonstrations are multimodal, which
is what action-chunking and diffusion policies are built to model and what a
single-strategy planner would never produce.

It also serves as the negative control for the force-conditioning ablation:
pushing a light puck across a table needs no force feedback, so a torque channel
that "helps" here is measuring a leak, not contact.
"""

from __future__ import annotations

import numpy as np

from ..env import OpenArmEnv

#: Puck half-extents, from the scene's ``puck_geom``.
PUCK_HALF = np.array([0.035, 0.035, 0.022])

#: Goal marker radius, from the scene's goal sites.
GOAL_RADIUS = 0.05

#: Cell table top, in world coordinates.
TABLE_TOP_Z = 1.005

#: The puck counts as delivered when its centre sits on the goal marker the
#: scene draws, which is 50 mm across.
SUCCESS_RADIUS = GOAL_RADIUS

GOALS = ("goal_a", "goal_b")
GOAL_COLOURS = {"goal_a": "green", "goal_b": "blue"}


class MovePuckEnv(OpenArmEnv):
    """Deliver the puck to whichever goal marker the instruction names."""

    SCENE = "cell/move_puck_cell_scene.xml"
    MAX_STEPS = 900

    def __init__(self, **kwargs) -> None:
        """Resolve the scene's task bodies and cache their authored poses."""
        super().__init__(**kwargs)

        self._puck_body = self.body_id("puck")
        self._puck_geom = self.geom_id("puck_geom")
        self._goal_bodies = {name: self.body_id(name) for name in GOALS}

        puck_joint = self.model.body_jntadr[self._puck_body]
        self._puck_qpos = int(self.model.jnt_qposadr[puck_joint])
        self._puck_dof = int(self.model.jnt_dofadr[puck_joint])

        self._fingers = {
            side: (
                self.body_id(f"openarm_{side}_ee_inner_finger"),
                self.body_id(f"openarm_{side}_ee_outer_finger"),
            )
            for side in ("right", "left")
        }

        self._nominal_puck = self.model.body_pos[self._puck_body].copy()
        self._nominal_mass = float(self.model.body_mass[self._puck_body])
        self._nominal_friction = self.model.geom_friction[self._puck_geom].copy()

        self.goal_name = GOALS[0]

    # ------------------------------------------------------------- geometry

    @property
    def puck_pos(self) -> np.ndarray:
        """Return the puck's centre in world coordinates."""
        return self.data.xpos[self._puck_body].copy()

    @property
    def goal_pos(self) -> np.ndarray:
        """Return the commanded goal marker's position."""
        return self.data.xpos[self._goal_bodies[self.goal_name]].copy()

    @property
    def resting_height(self) -> float:
        """Return the puck centre height when it sits flat on the table."""
        return TABLE_TOP_Z + PUCK_HALF[2]

    @property
    def puck_yaw(self) -> float:
        """Return the puck's rotation about the vertical, in radians."""
        forward = self.data.xmat[self._puck_body].reshape(3, 3)[:, 0]
        return float(np.arctan2(forward[1], forward[0]))

    def grasp_target(self) -> np.ndarray:
        """Return the world point the fingertips should pinch on the puck."""
        puck = self.puck_pos
        return np.array([puck[0], puck[1], self.resting_height])

    def place_target(self) -> np.ndarray:
        """Return where the puck's centre should end up."""
        goal = self.goal_pos
        return np.array([goal[0], goal[1], self.resting_height])

    def is_grasped(self) -> bool:
        """Return whether either hand has both fingertips on the puck."""
        return any(
            self.touching(self._puck_geom, inner) and self.touching(self._puck_geom, outer)
            for inner, outer in self._fingers.values()
        )

    def goal_distance(self) -> float:
        """Return the puck's planar distance from the commanded goal."""
        return float(np.linalg.norm(self.puck_pos[:2] - self.goal_pos[:2]))

    # ----------------------------------------------------------- task hooks

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Pick a goal, place the puck and randomize its physics."""
        self.goal_name = GOALS[int(rng.integers(len(GOALS)))]
        self.instruction = (
            f"move the puck to the {GOAL_COLOURS[self.goal_name]} goal"
        )

        puck_xy = self._nominal_puck[:2] + rng.uniform([-0.04, -0.05], [0.04, 0.05])
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self.data.qpos[self._puck_qpos : self._puck_qpos + 3] = [
            puck_xy[0],
            puck_xy[1],
            self.resting_height,
        ]
        self.data.qpos[self._puck_qpos + 3 : self._puck_qpos + 7] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        self.data.qvel[self._puck_dof : self._puck_dof + 6] = 0.0

        if self.domain_randomize:
            self.model.body_mass[self._puck_body] = self._nominal_mass * rng.uniform(
                0.6, 1.6
            )
            self.model.geom_friction[self._puck_geom] = (
                self._nominal_friction * rng.uniform(0.6, 1.5)
            )
        else:
            self.model.body_mass[self._puck_body] = self._nominal_mass
            self.model.geom_friction[self._puck_geom] = self._nominal_friction

    def compute_reward(self) -> float:
        """Return a staged reach / contact / deliver reward."""
        from ..experts.base import GRASP_OFFSET

        reach = float(
            np.linalg.norm(
                self.ik.grasp_point(self.data, "right", GRASP_OFFSET)
                - self.grasp_target()
            )
        )
        reward = 0.2 * (1.0 - np.tanh(6.0 * reach))
        reward += 1.5 * (1.0 - np.tanh(4.0 * self.goal_distance()))
        if self.is_success():
            reward += 5.0
        return reward

    def is_success(self) -> bool:
        """Return whether the puck rests inside the commanded goal marker."""
        puck = self.puck_pos
        if self.goal_distance() > SUCCESS_RADIUS:
            return False
        return abs(puck[2] - self.resting_height) < 0.015
