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

"""Peg-in-socket insertion on the OpenArm Cell workspace.

The scene ships a 14 mm-radius cylindrical peg and an octagonal socket of
20 mm inner apothem, so the radial clearance is 6 mm: an insertion tight
enough that contact feedback matters, which is the point of running it on a
backdrivable arm.
"""

from __future__ import annotations

import numpy as np

from ..env import OpenArmEnv

#: Distance from the control-point site to the fingertip pinch, along the tool
#: axis. Measured from the model's finger collision geoms.
GRASP_OFFSET = 0.155

#: Peg dimensions, from the scene's ``peg_geom``.
PEG_RADIUS = 0.014
PEG_HALF_LENGTH = 0.045

#: Socket inner apothem and wall height, from the scene's ``socket_*`` geoms.
SOCKET_APOTHEM = 0.020
SOCKET_WALL_HEIGHT = 0.05

#: Cell table top, in world coordinates.
TABLE_TOP_Z = 1.005

#: How far below the peg's top rim the fingertips pinch. Grasping high keeps
#: the fingers clear of the socket's 50 mm walls during insertion.
GRASP_INSET = 0.008

#: A peg counts as seated when its centre is within this of the seated height.
SEAT_TOLERANCE = 0.015
#: ...and its axis is within this of vertical.
UPRIGHT_TOLERANCE = 0.9


class PegSocketEnv(OpenArmEnv):
    """Pick the peg up and insert it into the socket, using the right arm."""

    SCENE = "cell/peg_socket_cell_scene.xml"
    MAX_STEPS = 900

    def __init__(self, **kwargs) -> None:
        """Resolve the scene's task bodies and cache their authored poses."""
        super().__init__(**kwargs)

        self._peg_body = self.body_id("peg")
        self._socket_body = self.body_id("socket")
        self._peg_geom = self.geom_id("peg_geom")
        self._socket_site = self.site_id("socket_center")

        peg_joint = self.model.body_jntadr[self._peg_body]
        self._peg_qpos = int(self.model.jnt_qposadr[peg_joint])
        self._peg_dof = int(self.model.jnt_dofadr[peg_joint])

        self._fingers = {
            side: (
                self.body_id(f"openarm_{side}_ee_inner_finger"),
                self.body_id(f"openarm_{side}_ee_outer_finger"),
            )
            for side in ("right", "left")
        }

        self._nominal_peg = self.model.body_pos[self._peg_body].copy()
        self._nominal_socket = self.model.body_pos[self._socket_body].copy()
        self._nominal_peg_mass = float(self.model.body_mass[self._peg_body])
        self._nominal_peg_friction = self.model.geom_friction[self._peg_geom].copy()

    # ------------------------------------------------------------- geometry

    @property
    def peg_pos(self) -> np.ndarray:
        """Return the peg's centre in world coordinates."""
        return self.data.xpos[self._peg_body].copy()

    @property
    def peg_axis(self) -> np.ndarray:
        """Return the peg's long axis (its body Z) in world coordinates."""
        return self.data.xmat[self._peg_body].reshape(3, 3)[:, 2].copy()

    @property
    def socket_pos(self) -> np.ndarray:
        """Return the socket base in world coordinates."""
        return self.data.xpos[self._socket_body].copy()

    @property
    def seated_height(self) -> float:
        """Return the peg centre height for a peg resting inside the socket."""
        return TABLE_TOP_Z + PEG_HALF_LENGTH

    def grasp_target(self) -> np.ndarray:
        """Return the world point the fingertips should pinch.

        Near the top of the peg, so the fingers clear the socket walls on the
        way down: the peg hangs below the pinch and enters the socket while the
        fingertips stay above the 50 mm walls.
        """
        top = self.peg_pos + self.peg_axis * PEG_HALF_LENGTH
        return top - self.peg_axis * GRASP_INSET

    def place_target(self) -> np.ndarray:
        """Return the world point the peg's centre should end up at."""
        socket = self.socket_pos
        return np.array([socket[0], socket[1], self.seated_height])

    def is_grasped(self) -> bool:
        """Return whether either hand has both fingertips on the peg."""
        return any(
            self.touching(self._peg_geom, inner) and self.touching(self._peg_geom, outer)
            for inner, outer in self._fingers.values()
        )

    # ----------------------------------------------------------- task hooks

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Randomize the peg pose, the socket position and the peg's physics."""
        self.instruction = "insert the peg into the socket"
        peg_xy = self._nominal_peg[:2] + rng.uniform(
            [-0.03, -0.025], [0.03, 0.025]
        )
        for _ in range(10):
            socket_xy = self._nominal_socket[:2] + rng.uniform(
                [-0.03, -0.025], [0.03, 0.025]
            )
            if np.linalg.norm(peg_xy - socket_xy) > 0.09:
                break

        # The socket is a fixed body, so it moves via the model, not qpos.
        self.model.body_pos[self._socket_body, :2] = socket_xy

        yaw = rng.uniform(-np.pi, np.pi)
        self.data.qpos[self._peg_qpos : self._peg_qpos + 3] = [
            peg_xy[0],
            peg_xy[1],
            TABLE_TOP_Z + PEG_HALF_LENGTH,
        ]
        self.data.qpos[self._peg_qpos + 3 : self._peg_qpos + 7] = [
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ]
        self.data.qvel[self._peg_dof : self._peg_dof + 6] = 0.0

        if self.domain_randomize:
            self.model.body_mass[self._peg_body] = self._nominal_peg_mass * rng.uniform(
                0.7, 1.4
            )
            self.model.geom_friction[self._peg_geom] = (
                self._nominal_peg_friction * rng.uniform(0.75, 1.25)
            )
        else:
            self.model.body_mass[self._peg_body] = self._nominal_peg_mass
            self.model.geom_friction[self._peg_geom] = self._nominal_peg_friction

    def compute_reward(self) -> float:
        """Return a staged reach / grasp / insert reward."""
        reach = float(
            np.linalg.norm(
                self.ik.grasp_point(self.data, "right", GRASP_OFFSET)
                - self.grasp_target()
            )
        )
        reward = 0.2 * (1.0 - np.tanh(8.0 * reach))

        if self.is_grasped():
            place = float(np.linalg.norm(self.peg_pos - self.place_target()))
            reward += 1.0 + 1.0 * (1.0 - np.tanh(8.0 * place))

        if self.is_success():
            reward += 5.0
        return reward

    def is_success(self) -> bool:
        """Return whether the peg is seated upright inside the socket."""
        peg = self.peg_pos
        socket = self.socket_pos
        radial = float(np.linalg.norm(peg[:2] - socket[:2]))
        if radial > SOCKET_APOTHEM - PEG_RADIUS + 0.006:
            return False
        if abs(peg[2] - self.seated_height) > SEAT_TOLERANCE:
            return False
        return abs(float(self.peg_axis[2])) > UPRIGHT_TOLERANCE
