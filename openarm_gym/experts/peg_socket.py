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

"""Scripted peg-in-socket expert.

Uses a *side* grasp, holding the tool horizontal. A top-down grasp is
unreachable here -- the arm hangs from the cell lifter and cannot get its wrist
above a peg standing on the table, which an orientation sweep shows as a 0.5-1.0
residual on every straight-down target against ~1e-5 on horizontal ones.

The jaws stay closed while travelling: open they are 95 mm across and knock the
peg over before the grasp, closed they are 9 mm.
"""

from __future__ import annotations

import numpy as np

from ..tasks.peg_socket import PEG_HALF_LENGTH, GRASP_INSET, PegSocketEnv
from .base import Waypoint, WaypointExpert

#: Jaw opening used to clear the 28 mm peg on approach.
OPEN = 0.57

#: How far the peg hangs below the fingertips once grasped.
PEG_HANG = PEG_HALF_LENGTH - GRASP_INSET


class PegSocketExpert(WaypointExpert):
    """Pick the peg up and seat it in the socket."""

    SIDE = "right"
    #: Track the peg through "approach"; commit from "advance" onwards.
    TRACK_UNTIL = 2

    def __init__(self, env: PegSocketEnv, **kwargs) -> None:
        """Bind to a peg-socket environment."""
        super().__init__(env, **kwargs)
        self._grasp_bias = np.zeros(3)
        self._place_bias = np.zeros(3)
        self._safe_pinch = np.zeros(3)

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Sample the perception biases and the initial retreat point."""
        noise = self.params.waypoint_noise
        self._grasp_bias = rng.normal(0.0, noise, size=3) if noise > 0 else np.zeros(3)
        self._place_bias = rng.normal(0.0, noise, size=3) if noise > 0 else np.zeros(3)
        # Back straight off along the tool axis before going anywhere else, so
        # the hand never sweeps sideways through a peg standing next to it.
        start = self.pinch_now()
        self._safe_pinch = start.copy()

    def plan(self) -> list[Waypoint]:
        """Return the pick-and-insert waypoints for the current world state."""
        env: PegSocketEnv = self.env
        params = self.params

        grasp = self.tracked("grasp", env.grasp_target() + self._grasp_bias)
        place_centre = env.place_target() + self._place_bias
        place = np.array(
            [place_centre[0], place_centre[1], place_centre[2] + PEG_HANG]
        )

        back = self.tool_dir * params.standoff
        up = np.array([0.0, 0.0, params.lift_height])
        safe = self._safe_pinch - self.tool_dir * (params.standoff + 0.04)

        return [
            Waypoint(safe, 0.0, name="retract"),
            Waypoint(grasp - back + up, 0.0, name="clear"),
            Waypoint(grasp - back, OPEN, name="approach"),
            Waypoint(grasp, OPEN, speed=params.slow_speed, name="advance"),
            Waypoint(grasp, 0.0, dwell=35, name="grasp"),
            Waypoint(grasp + up, 0.0, name="lift"),
            Waypoint(place + up, 0.0, name="transfer"),
            Waypoint(place + up, 0.0, dwell=30, name="align"),
            Waypoint(place, 0.0, speed=params.slow_speed, dwell=35, name="insert"),
            Waypoint(place, OPEN, dwell=20, name="release"),
            Waypoint(place - back, OPEN, name="retreat"),
        ]
