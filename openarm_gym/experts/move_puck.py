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

"""Scripted puck-delivery expert, with two strategies.

Each episode picks either a push or a pick-and-place, so the demonstration set
for this task is genuinely multimodal: two different action sequences are
correct from the same starting observation. That is the distribution
action-chunking and diffusion policies exist to model, and a single-strategy
scripted planner never produces it.
"""

from __future__ import annotations

import numpy as np

from ..tasks.move_puck import PUCK_HALF, MovePuckEnv
from .base import Waypoint, WaypointExpert

#: Jaw opening used on approach. The puck is 70 mm across the faces and 99 mm
#: across the diagonal, so the jaws are opened most of the way: a narrower gap
#: catches a corner and shoves the puck instead of straddling it.
OPEN = 0.92

#: Which arm reaches which goal. goal_a sits at y=-0.16, goal_b at y=+0.16.
GOAL_SIDES = {"goal_a": "right", "goal_b": "left"}

#: How far behind the puck's trailing face the pusher lines up.
PUSH_LEAD = 0.055

#: The puck rides ahead of the fingertips while being pushed, by its own half
#: width plus the pad thickness. Measured by sweeping the offset and reading the
#: signed along-push error: 0.030 lands the puck within 5 +/- 14 mm of the goal
#: along the push axis.
PUSH_CONTACT_OFFSET = 0.030

#: Waypoint index at which the push re-aims on the puck's actual position.
_REAIM_INDEX = 3


class MovePuckExpert(WaypointExpert):
    """Deliver the puck to the commanded goal by pushing or by pick-and-place."""

    SIDE = "right"
    ORIENTATIONS = (
        (-90.0, 0.0),
        (-75.0, 0.0),
        (-60.0, 0.0),
        (-60.0, -30.0),
        (-60.0, 30.0),
        (-60.0, 60.0),
        (-60.0, -60.0),
        (-45.0, 0.0),
    )

    def __init__(self, env: MovePuckEnv, **kwargs) -> None:
        """Bind to a puck environment."""
        super().__init__(env, **kwargs)
        self.strategy = "push"
        self._bias = np.zeros(3)
        self._reaim: np.ndarray | None = None

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Choose the arm, the strategy and the perception bias.

        The goals sit either side of the midline and each arm only reaches its
        own: pick-and-place to the far goal with the near arm drops the puck
        short every time. Choosing by goal makes the task properly bimanual.
        """
        self._use_side(GOAL_SIDES[self.env.goal_name])
        self.strategy = "push" if rng.random() < 0.5 else "place"

        if self.strategy == "place":
            # Line the jaws up with the puck's faces. Approaching a rotated box
            # at an arbitrary yaw walks a corner into one finger, which shoves
            # the puck out from under the grasp -- it was the whole of the left
            # arm's 0/10. Any of the four face normals will do, so offer all of
            # them and let the IK residual pick the one the arm can reach.
            puck_yaw = np.degrees(self.env.puck_yaw)
            self.ORIENTATIONS = tuple(
                (pitch, float(puck_yaw + 90.0 * quarter))
                for pitch in (-90.0, -75.0, -60.0)
                for quarter in range(4)
            )
        else:
            # Pushing wants the tool lined up with the direction of travel, not
            # with the puck's faces: face-aligning the pusher cost it 2/2.
            heading = self.env.place_target()[:2] - self.env.grasp_target()[:2]
            bearing = np.degrees(np.arctan2(heading[1], heading[0]))
            # Offer both the travel-aligned candidates and the generic set, and
            # let the residual search choose: neither alone covers both arms.
            self.ORIENTATIONS = tuple(
                (pitch, float(bearing + offset))
                for pitch in (-90.0, -75.0, -60.0)
                for offset in (0.0, -30.0, 30.0)
            ) + MovePuckExpert.ORIENTATIONS
        noise = self.params.waypoint_noise
        self._bias = rng.normal(0.0, noise, size=3) if noise > 0 else np.zeros(3)
        self._reaim = None

    def plan(self) -> list[Waypoint]:
        """Return the waypoints for whichever strategy this episode uses."""
        if self.strategy == "push":
            return self._push_plan()
        return self._place_plan()

    def _push_plan(self) -> list[Waypoint]:
        """Line up behind the puck and sweep it onto the goal."""
        env: MovePuckEnv = self.env
        params = self.params

        goal = env.place_target() + self._bias
        up = np.array([0.0, 0.0, params.lift_height])

        def lead_from(point: np.ndarray) -> np.ndarray:
            """Return the unit heading from a point towards the goal."""
            direction = goal[:2] - point[:2]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                return np.array([1.0, 0.0, 0.0])
            return np.array([direction[0] / norm, direction[1] / norm, 0.0])

        live = env.grasp_target() + self._bias

        # First segment: line up behind the puck and drive most of the way.
        start = self.tracked("puck", live)
        lead0 = lead_from(start)
        behind = start - lead0 * (PUSH_CONTACT_OFFSET + PUSH_LEAD)
        midway = start + (goal - start) * 0.55 - lead0 * PUSH_CONTACT_OFFSET

        # Second segment: re-aim once from wherever the puck actually ended up.
        # A narrow fingertip pushing a 70 mm box veers -- 42 mm of lateral drift,
        # measured -- so an open-loop push misses. Continuously chasing the puck
        # is worse still: the hand swings around behind it and knocks it further
        # (2/12). One correction, taken at the midpoint, splits the difference.
        if self._index < _REAIM_INDEX:
            self._reaim = None
        elif self._reaim is None:
            self._reaim = live.copy()
        second = live if self._reaim is None else self._reaim
        lead1 = lead_from(second)
        regroup = second - lead1 * (PUSH_CONTACT_OFFSET + 0.035)
        finish = goal - lead1 * PUSH_CONTACT_OFFSET

        return [
            Waypoint(behind + up, 0.0, name="clear"),
            Waypoint(behind, 0.0, name="setup"),
            Waypoint(midway, 0.0, speed=params.slow_speed, dwell=10, name="push"),
            Waypoint(regroup, 0.0, dwell=8, name="reaim"),
            Waypoint(finish, 0.0, speed=params.slow_speed, dwell=25, name="deliver"),
            Waypoint(finish - lead1 * 0.05 + up, 0.0, name="retreat"),
        ]

    def _place_plan(self) -> list[Waypoint]:
        """Grasp the puck, carry it and set it down on the goal."""
        env: MovePuckEnv = self.env
        params = self.params

        grasp = self.tracked("grasp", env.grasp_target() + self._bias)
        place = env.place_target() + self._bias
        back = self.tool_dir * params.standoff
        up = np.array([0.0, 0.0, params.lift_height])
        # Set down a little high and let the puck drop the last millimetres,
        # rather than driving it into the table.
        release = place + np.array([0.0, 0.0, 0.004])

        return [
            Waypoint(grasp - back + up, 0.0, name="clear"),
            Waypoint(grasp - back, OPEN, name="approach"),
            Waypoint(grasp, OPEN, speed=params.slow_speed, name="advance"),
            Waypoint(grasp, 0.0, dwell=35, name="grasp"),
            Waypoint(grasp + up, 0.0, name="lift"),
            Waypoint(release + up, 0.0, name="transfer"),
            Waypoint(release, 0.0, speed=params.slow_speed, dwell=25, name="lower"),
            Waypoint(release, OPEN, dwell=20, name="release"),
            Waypoint(release - back + up * 0.5, OPEN, name="retreat"),
        ]

    @property
    def puck_clearance(self) -> float:
        """Return the puck's half-height, for reference by callers."""
        return float(PUCK_HALF[2])
