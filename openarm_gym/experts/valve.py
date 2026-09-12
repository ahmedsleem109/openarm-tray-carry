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

"""Scripted valve-turning expert.

The post at the lever tip travels on a circle, so the routine lays a chain of
waypoints along that arc rather than a straight line: the base follower
interpolates straight between targets, and a straight chord across a 60-degree
sweep would drag the jaws off the post.

The tool does *not* rotate with the lever. Yawing the wrist to keep a constant
pose relative to the lever sounds right and measures worse -- 6/12 against 9/12
-- because the post is a cylinder that spins freely in the jaws anyway, so the
rotation buys no grip and only adds wrist disturbance mid-sweep.

The post is approached tangentially, from the side the lever is turning away
from. Radial approaches do not work: the hub sits between the robot and the post
at the lever's rest angle, so coming in along the radius from the near side
means driving through the hub, and from the far side it asks the arm to reach
past the valve and point back at itself, which it cannot do -- that variant
scored 0/12 against 9/12 for the tangential one.
"""

from __future__ import annotations

import numpy as np

from ..tasks.valve import LEVER_RADIUS, ValveEnv
from .base import Waypoint, WaypointExpert

#: Jaw opening that clears the 14 mm post.
OPEN = 0.45

#: Waypoints laid along the turn arc. Enough that the chord error between
#: consecutive targets stays under a millimetre for the longest sweep.
ARC_STEPS = 8

#: Radians of overshoot past the commanded angle. Zero: the lever lags the hand
#: under its damping and friction and tends to settle short, but driving past
#: the target and letting it settle back measures worse, not better (12/20
#: against 14/20 at 0.10 rad) -- friction holds the lever wherever it is pushed
#: rather than springing back.
ARC_OVERSHOOT = 0.0


class ValveExpert(WaypointExpert):
    """Grasp the lever post and sweep it to the commanded angle."""

    SIDE = "right"
    ORIENTATIONS = (
        (-90.0, 0.0),
        (-90.0, 45.0),
        (-90.0, -45.0),
        (-75.0, 0.0),
        (-75.0, 45.0),
        (-75.0, -45.0),
        (-60.0, 0.0),
        (-60.0, 45.0),
        (-60.0, -45.0),
        (-60.0, 90.0),
        (-60.0, -90.0),
    )

    def __init__(self, env: ValveEnv, **kwargs) -> None:
        """Bind to a valve environment."""
        super().__init__(env, **kwargs)
        self._bias = np.zeros(3)
        self._tangent = np.array([0.0, 1.0, 0.0])

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Sample the perception bias and pick the tangential approach side."""
        noise = self.params.waypoint_noise
        self._bias = rng.normal(0.0, noise, size=3) if noise > 0 else np.zeros(3)

        # Stand off on the side the lever is turning away from, so the hand is
        # never in the path of the sweep.
        env: ValveEnv = self.env
        start = env.start_angle
        tangent = np.array([-np.sin(start), np.cos(start), 0.0])
        self._tangent = -tangent if env.target_angle > start else tangent

    def _post_at(self, angle: float) -> np.ndarray:
        """Return the post centre at a turn angle, with the perception bias."""
        return self.env.post_pos(angle) + self._bias

    def plan(self) -> list[Waypoint]:
        """Return approach, grasp, arc-sweep and release waypoints."""
        env: ValveEnv = self.env
        params = self.params

        start = env.start_angle
        target = env.target_angle
        grasp = self.tracked("grasp", self._post_at(start))

        standoff = self._tangent * params.standoff
        up = np.array([0.0, 0.0, params.lift_height])

        waypoints = [
            Waypoint(grasp + standoff + up, 0.0, name="clear"),
            Waypoint(grasp + standoff, OPEN, name="approach"),
            Waypoint(grasp, OPEN, speed=params.slow_speed, name="advance"),
            Waypoint(grasp, 0.0, dwell=35, name="grasp"),
        ]

        # Sweep the arc. Turning slowly keeps the pinch loaded against the
        # lever's damping instead of snatching it off the post.
        sweep = target - start
        driven = target + np.sign(sweep) * ARC_OVERSHOOT
        for step in range(1, ARC_STEPS + 1):
            angle = start + (driven - start) * step / ARC_STEPS
            waypoints.append(
                Waypoint(
                    self._post_at(angle),
                    0.0,
                    speed=params.slow_speed,
                    dwell=6,
                    tol=0.006,
                    name=f"turn{step}",
                )
            )

        settled = self._post_at(target)
        release_dir = settled - env.axis_pos
        release_dir /= max(float(np.linalg.norm(release_dir)), 1e-6)

        waypoints += [
            Waypoint(settled, 0.0, dwell=25, name="hold"),
            Waypoint(settled, OPEN, dwell=20, name="release"),
            Waypoint(
                settled + release_dir * (LEVER_RADIUS * 0.5) + up,
                OPEN,
                name="retreat",
            ),
        ]
        return waypoints
