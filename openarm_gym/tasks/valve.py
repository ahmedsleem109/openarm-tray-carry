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

"""Turn a valve lever to a commanded angle, on the OpenArm Cell workspace.

The scene provides a fixed pipe carrying a lever that turns about the vertical
axis, with a graspable post at the lever tip 62 mm out from the hub. Turn
direction and magnitude vary per episode and are carried as a language
instruction.

This is the task where contact feedback should matter most. The lever has
damping and friction loss, the grasp is a fingertip pinch on a 14 mm post, and
the end-effector has to follow a constrained arc rather than a free path: push
along the wrong tangent and the jaws slip off the post. Joint torque carries
that constraint directly.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..env import OpenArmEnv

#: Lever length: distance from the turn axis to the graspable post, from the
#: scene's ``valve_grip`` geom.
LEVER_RADIUS = 0.062

#: Height of the post's centre above the valve base.
POST_HEIGHT = 0.065

#: The valve counts as set when it is within this of the commanded angle.
ANGLE_TOLERANCE = 0.15


class ValveEnv(OpenArmEnv):
    """Turn the valve lever to the commanded angle, using the right arm."""

    SCENE = "cell/valve_cell_scene.xml"
    MAX_STEPS = 900

    def __init__(self, **kwargs) -> None:
        """Resolve the valve joint and its geometry."""
        super().__init__(**kwargs)

        self._valve_body = self.body_id("valve")
        self._base_body = self.body_id("valve_base")
        self._grip_geom = self.geom_id("valve_grip")

        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "valve_turn"
        )
        if joint_id < 0:
            raise ValueError("Joint 'valve_turn' not found in the valve scene")
        self._valve_qpos = int(self.model.jnt_qposadr[joint_id])
        self._valve_dof = int(self.model.jnt_dofadr[joint_id])
        self._valve_range = self.model.jnt_range[joint_id].copy()

        self._fingers = {
            side: (
                self.body_id(f"openarm_{side}_ee_inner_finger"),
                self.body_id(f"openarm_{side}_ee_outer_finger"),
            )
            for side in ("right", "left")
        }

        self._nominal_base = self.model.body_pos[self._base_body].copy()
        self._nominal_damping = float(self.model.dof_damping[self._valve_dof])
        self._nominal_friction = float(self.model.dof_frictionloss[self._valve_dof])

        self.target_angle = 0.0
        self.start_angle = 0.0

    # ------------------------------------------------------------- geometry

    @property
    def angle(self) -> float:
        """Return the valve's current turn angle, in radians."""
        return float(self.data.qpos[self._valve_qpos])

    @property
    def axis_pos(self) -> np.ndarray:
        """Return the turn axis in world coordinates, at post height."""
        base = self.data.xpos[self._base_body].copy()
        base[2] += POST_HEIGHT
        return base

    def post_pos(self, angle: float | None = None) -> np.ndarray:
        """Return the graspable post's centre at a given (or current) angle."""
        if angle is None:
            return self.data.geom_xpos[self._grip_geom].copy()
        axis = self.axis_pos
        return axis + LEVER_RADIUS * np.array(
            [np.cos(angle), np.sin(angle), 0.0]
        )

    def angle_error(self) -> float:
        """Return the absolute difference from the commanded angle."""
        return abs(self.angle - self.target_angle)

    def is_grasped(self) -> bool:
        """Return whether either hand has both fingertips on the post."""
        return any(
            self.touching(self._grip_geom, inner) and self.touching(self._grip_geom, outer)
            for inner, outer in self._fingers.values()
        )

    # ----------------------------------------------------------- task hooks

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Command a turn angle and randomize the valve's resistance."""
        self.start_angle = float(rng.uniform(-0.25, 0.25))
        magnitude = float(rng.uniform(0.7, 1.4))
        direction = 1.0 if rng.random() < 0.5 else -1.0
        self.target_angle = float(
            np.clip(
                self.start_angle + direction * magnitude,
                self._valve_range[0] + 0.1,
                self._valve_range[1] - 0.1,
            )
        )
        turn = "anticlockwise" if self.target_angle > self.start_angle else "clockwise"
        self.instruction = f"turn the valve {turn}"

        self.data.qpos[self._valve_qpos] = self.start_angle
        self.data.qvel[self._valve_dof] = 0.0

        base_xy = self._nominal_base[:2] + rng.uniform([-0.02, -0.03], [0.02, 0.03])
        self.model.body_pos[self._base_body, :2] = base_xy

        if self.domain_randomize:
            self.model.dof_damping[self._valve_dof] = (
                self._nominal_damping * rng.uniform(0.5, 2.0)
            )
            self.model.dof_frictionloss[self._valve_dof] = (
                self._nominal_friction * rng.uniform(0.4, 2.5)
            )
        else:
            self.model.dof_damping[self._valve_dof] = self._nominal_damping
            self.model.dof_frictionloss[self._valve_dof] = self._nominal_friction

    def compute_reward(self) -> float:
        """Return a staged reach / grasp / turn reward."""
        from ..experts.base import GRASP_OFFSET

        reach = float(
            np.linalg.norm(
                self.ik.grasp_point(self.data, "right", GRASP_OFFSET) - self.post_pos()
            )
        )
        reward = 0.2 * (1.0 - np.tanh(8.0 * reach))

        travelled = abs(self.angle - self.start_angle)
        required = max(abs(self.target_angle - self.start_angle), 1e-6)
        reward += 1.5 * float(np.clip(travelled / required, 0.0, 1.0))
        reward += 0.5 * (1.0 - np.tanh(3.0 * self.angle_error()))

        if self.is_success():
            reward += 5.0
        return reward

    def is_success(self) -> bool:
        """Return whether the valve sits at the commanded angle."""
        return self.angle_error() < ANGLE_TOLERANCE
