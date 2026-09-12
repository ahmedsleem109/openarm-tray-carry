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

"""Shared machinery for the scripted waypoint experts.

Subclasses implement :meth:`WaypointExpert.plan`, returning the current list of
:class:`Waypoint` targets. ``plan`` is called *every step*, not once per
episode, which is what makes the resulting demonstrations worth learning from:

* When noise pushes the arm off course, the expert re-plans from where things
  actually are, so the data contains genuine error-correction rather than an
  open-loop trajectory replayed from a stale plan. That is the DART recipe --
  perturb during collection, label with the expert's clean action at the
  perturbed state -- and it is why a policy trained on this data has to learn
  closed-loop control instead of memorising a path.
* When an object is knocked, the waypoints follow it.

Targets are fingertip pinch points in world coordinates; the control-point pose
is derived by pushing back up the tool axis, so waypoints read in task
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..ik import TOOL_AXIS_LOCAL, quat_from_euler_zyx

#: Fingertip pinch offset from the control-point site, along the tool axis.
GRASP_OFFSET = 0.155

#: Gripper commands per arm. The right finger actuator's range is
#: ``[-0.7854, 0]`` and opens towards negative values; the left mirrors it.
GRIPPER_LIMIT = 0.7854


@dataclass
class Waypoint:
    """One target in a scripted routine."""

    #: World-frame fingertip pinch target.
    pinch: np.ndarray
    #: Jaw opening, 0 (closed) to 1 (fully open).
    opening: float = 0.0
    #: Travel speed in m/s; ``None`` uses the expert's default.
    speed: float | None = None
    #: Steps to hold after arriving, letting the arm and payload settle.
    dwell: int = 4
    #: Arrival tolerance in metres.
    tol: float = 0.005
    #: Tool orientation for this target. ``None`` uses the episode's chosen
    #: orientation. Routines that follow a curve -- a valve lever sweeping an
    #: arc -- must rotate the tool with it, or the jaws slide off.
    quat: np.ndarray | None = None
    #: Optional label, recorded alongside the demonstration.
    name: str = ""


@dataclass
class _Params:
    """Per-episode expert parameters, resampled at every reset."""

    speed: float = 0.25
    slow_speed: float = 0.06
    standoff: float = 0.09
    lift_height: float = 0.07
    waypoint_noise: float = 0.0
    extra: dict = field(default_factory=dict)


class WaypointExpert:
    """Speed-limited waypoint follower with live re-planning."""

    #: Which arm this routine drives.
    SIDE = "right"
    #: Candidate tool orientations, as ``(pitch_deg, yaw_deg)``. The one that
    #: solves every waypoint with the smallest worst-case IK residual is chosen
    #: at reset, so no task has to hand-tune an approach angle.
    ORIENTATIONS: tuple[tuple[float, float], ...] = (
        (-90.0, 0.0),
        (-75.0, 0.0),
        (-60.0, 0.0),
        (-60.0, -30.0),
        (-60.0, 30.0),
        (-45.0, 0.0),
        (-30.0, 0.0),
    )
    #: Ranges the per-episode parameters are drawn from.
    SPEED_RANGE = (0.18, 0.32)
    SLOW_SPEED_RANGE = (0.04, 0.08)
    STANDOFF_RANGE = (0.075, 0.105)
    LIFT_RANGE = (0.06, 0.09)

    #: Timeout per waypoint, in steps, past its dwell.
    TIMEOUT = 140

    #: Waypoint index up to which object-derived targets follow the object
    #: live. Past it they freeze at their last value.
    #:
    #: Live tracking is what puts error-correction in the data, but it has to
    #: stop once the arm commits to the object. Track through the final
    #: approach and the gripper nudges the object, the target follows the
    #: object, and the arm chases it across the table -- a runaway that
    #: bulldozed the peg 18 cm before this cut it off.
    TRACK_UNTIL = 1

    def __init__(self, env, *, waypoint_noise: float = 0.0) -> None:
        """Bind the expert to an environment."""
        self.env = env
        self.waypoint_noise = waypoint_noise
        self._dt = 1.0 / env.control_hz

        self.side = self.SIDE
        self._use_side(self.SIDE)

        self.params = _Params()
        self._rot = np.eye(3)
        self.grasp_quat = quat_from_euler_zyx(0.0, -np.pi / 2, 0.0)
        self.tool_dir = np.array([1.0, 0.0, 0.0])

        self._index = 0
        self._steps = 0
        self._frozen: dict[str, np.ndarray] = {}
        self._cursor = np.zeros(3)
        self._target = np.zeros(3)
        self._rng = np.random.default_rng()

    def _use_side(self, side: str) -> None:
        """Point this expert at one arm, fixing its action slots."""
        if side not in ("left", "right"):
            raise ValueError(f"Invalid side: {side!r}")
        self.side = side
        self._joint_slice = slice(0, 7) if side == "right" else slice(8, 15)
        self._gripper_index = 7 if side == "right" else 15
        # The right gripper opens towards negative values, the left positive.
        self._gripper_sign = -1.0 if side == "right" else 1.0

    # ------------------------------------------------------------- geometry

    def _set_orientation(self, pitch_deg: float, yaw_deg: float) -> None:
        """Fix the tool orientation used for every waypoint this episode."""
        self.grasp_quat = quat_from_euler_zyx(
            0.0, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
        )
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, self.grasp_quat)
        self._rot = rot.reshape(3, 3)
        self.tool_dir = self._rot @ TOOL_AXIS_LOCAL

    def control_pos(
        self, pinch: np.ndarray, rot: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the control-point position that puts the pinch at ``pinch``."""
        rotation = self._rot if rot is None else rot
        return pinch - rotation @ (TOOL_AXIS_LOCAL * GRASP_OFFSET)

    def pose_for(self, waypoint: Waypoint) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(quat, rot)`` a waypoint should be reached with."""
        if waypoint.quat is None:
            return self.grasp_quat, self._rot
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, waypoint.quat)
        return waypoint.quat, rot.reshape(3, 3)

    def pinch_now(self) -> np.ndarray:
        """Return where the fingertips actually are."""
        return self.env.ik.grasp_point(self.env.data, self.side, GRASP_OFFSET)

    def tracked(self, key: str, value: np.ndarray) -> np.ndarray:
        """Follow an object-derived target live, then freeze it once committed.

        See :attr:`TRACK_UNTIL`.
        """
        if self._index <= self.TRACK_UNTIL or key not in self._frozen:
            self._frozen[key] = np.asarray(value, dtype=np.float64).copy()
        return self._frozen[key]

    def jitter(self, point: np.ndarray) -> np.ndarray:
        """Add the configured waypoint noise to a target."""
        if self.params.waypoint_noise <= 0.0:
            return np.asarray(point, dtype=np.float64)
        return np.asarray(point, dtype=np.float64) + self._rng.normal(
            0.0, self.params.waypoint_noise, size=3
        )

    # --------------------------------------------------------------- resets

    def reset(self, seed: int | None = None) -> None:
        """Resample parameters, choose an orientation and restart the routine."""
        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._steps = 0
        self._frozen = {}
        self._use_side(self.SIDE)

        rng = self._rng
        self.params = _Params(
            speed=rng.uniform(*self.SPEED_RANGE),
            slow_speed=rng.uniform(*self.SLOW_SPEED_RANGE),
            standoff=rng.uniform(*self.STANDOFF_RANGE),
            lift_height=rng.uniform(*self.LIFT_RANGE),
            waypoint_noise=self.waypoint_noise,
        )
        self._on_reset(rng)
        self._choose_orientation()
        self._cursor = self.pinch_now()

    def _on_reset(self, rng: np.random.Generator) -> None:
        """Make the task-specific per-episode choices, if there are any."""

    def _choose_orientation(self) -> None:
        """Pick the tool orientation that reaches every waypoint best."""
        best = (np.inf, self.ORIENTATIONS[0])
        for pitch, yaw in self.ORIENTATIONS:
            self._set_orientation(pitch, yaw)
            worst = 0.0
            for waypoint in self.plan():
                quat, rot = self.pose_for(waypoint)
                _joints, residual = self.env.ik.solve(
                    self.env.data,
                    self.side,
                    self.control_pos(waypoint.pinch, rot),
                    quat,
                    iters=60,
                )
                worst = max(worst, residual)
                if worst >= best[0]:
                    break
            if worst < best[0]:
                best = (worst, (pitch, yaw))
        self._set_orientation(*best[1])
        self.orientation_residual = float(best[0])

    # --------------------------------------------------------------- policy

    def plan(self) -> list[Waypoint]:
        """Return the routine's waypoints, given the current world state."""
        raise NotImplementedError

    def _gripper_command(self, opening: float) -> float:
        """Map a 0-1 jaw opening onto this arm's actuator command."""
        return self._gripper_sign * float(np.clip(opening, 0.0, 1.0)) * GRIPPER_LIMIT

    def act(self) -> np.ndarray:
        """Return the next 16-value joint position command.

        This is the *clean* expert action at the current state. Collection adds
        exploration noise on top via :meth:`perturb`, so the recorded label is
        the correction the expert would make from wherever the arm ended up.
        """
        waypoints = self.plan()
        self._index = min(self._index, len(waypoints) - 1)
        waypoint = waypoints[self._index]
        self._target = waypoint.pinch

        speed = waypoint.speed if waypoint.speed is not None else self.params.speed
        delta = waypoint.pinch - self._cursor
        distance = float(np.linalg.norm(delta))
        step = speed * self._dt
        if distance <= step:
            self._cursor = waypoint.pinch.copy()
            at_waypoint = True
        else:
            self._cursor = self._cursor + delta * (step / distance)
            at_waypoint = False

        quat, rot = self.pose_for(waypoint)
        joints, _residual = self.env.ik.solve(
            self.env.data,
            self.side,
            self.control_pos(self._cursor, rot),
            quat,
            iters=80,
        )

        arrived = float(np.linalg.norm(self.pinch_now() - waypoint.pinch)) < waypoint.tol
        self._steps += 1
        ready = at_waypoint and (arrived or self._steps > self.TIMEOUT)
        if ready and self._steps >= waypoint.dwell and self._index < len(waypoints) - 1:
            self._index += 1
            self._steps = 0

        action = self.env.driver_position()
        action[self._joint_slice] = joints
        action[self._gripper_index] = self._gripper_command(waypoint.opening)
        return action

    def perturb(self, action: np.ndarray, scale: float) -> np.ndarray:
        """Add exploration noise to an action, for DART-style collection."""
        if scale <= 0.0:
            return action
        noisy = np.array(action, dtype=np.float64)
        noisy[self._joint_slice] += self._rng.normal(0.0, scale, size=7)
        return np.clip(noisy, self.env.action_space.low, self.env.action_space.high)

    @property
    def phase(self) -> str:
        """Return the name of the waypoint currently being followed."""
        waypoints = self.plan()
        return waypoints[min(self._index, len(waypoints) - 1)].name

    @property
    def finished(self) -> bool:
        """Return whether the routine has reached its last waypoint."""
        return self._index >= len(self.plan()) - 1
