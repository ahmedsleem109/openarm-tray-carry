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

"""Coordinated two-arm control of a grasped tray, and ball balancing on it.

Both grippers grasp handle posts on a tray by friction -- there is no weld, and
arm collision geometry stays enabled. Once the grasp closes, each gripper is
rigidly related to the tray, so the two arms stop being independent: a single
commanded **tray pose** determines both arms' targets. That is the coordination.
Inconsistent arm motion would fight through the tray, so the controller never
commands the arms separately.

The ball is then controlled by tilting the tray. On a surface tilted by a small
angle the in-plane acceleration is ``g * sin(t) * cos(t) ~= g * t``, so a PD law
on the ball's position in the tray frame maps directly to a commanded tilt:

    a_desired = -kp * ball_xy - kd * ball_vxy
    pitch(about y) =  a_x / g        roll(about x) = -a_y / g

This is classical control, no learning. It exists both as a working carry
controller and as the baseline any learned policy has to beat -- and, since
:func:`balance_law` is callable on its own, as the law a *vision* policy drives
by estimating the ball state rather than regressing the tilt.

The controller also carries the optional machinery that makes the task harder and
more transferable, all of it off by default so the measured numbers above stay
reproducible: :meth:`BimanualTrayCarry.nudge_ball` for mid-episode disturbances,
a yaw goal through :func:`tray_quat`, camera observations through
:meth:`BimanualTrayCarry.observe`, and torque-level actuation, command latency,
backlash and dynamics randomization from :mod:`openarm_gym.realism`.

Measured facts this relies on (see the module tests):

* The jaws close along the world y axis and reach along +x at the home pose.
* The finger pads sit ~0.125 m ahead of the wrist site, not the 0.155 m
  ``GRASP_OFFSET`` the peg experts use.
* ``PoseController.solve`` drives the wrist *site*, so a desired pad position
  must be converted with ``site = pad - tool_dir * PAD_OFFSET``.
* Reaching in at the handle centre scrapes the gripper's palm along the tray's
  top face and shoves the tray 30 mm forward. :meth:`BimanualTrayCarry.grasp`
  reaches in above that and settles down.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import mujoco

from ..ik import PoseController, quat_conj, quat_mul
from ..realism import ActuatorModel, DynamicsRandomizer, DynamicsRanges, JointTorqueServo
from ..vision import CameraRig

#: Finger-pad centre ahead of the wrist site, in metres. Measured: 0.115 m with
#: the jaws open and 0.135 m closed, so this sits between the two and keeps the
#: post between the pads as the fingers swing shut.
PAD_OFFSET = 0.125

#: Tool orientation at the home pose, and the world direction the jaws reach.
TOOL_QUAT = np.array([0.7071, 0.0, -0.7071, 0.0])
TOOL_DIR = np.array([1.0, 0.0, 0.0])

QPOS = {"left": slice(0, 7), "right": slice(9, 16)}
CTRL = {"left": slice(0, 7), "right": slice(8, 15)}
FINGER_QPOS = {"left": (7, 8), "right": (16, 17)}
FINGER_CTRL = {"left": 7, "right": 15}
#: The right gripper opens toward negative values, the left toward positive.
OPEN = {"left": 0.7854, "right": -0.7854}
SHUT = {"left": 0.0, "right": 0.0}

#: Height above the handle centre at which the jaws reach in, and the height at
#: which they close. Measured: reaching in at the handle centre makes the
#: gripper's palm scrape the tray's top face and shove the tray 30 mm forward,
#: which is 8x the displacement of reaching in high and settling down.
APPROACH_LIFT = 0.020
CLOSE_LIFT = 0.0

GRAVITY = 9.81

#: Free-joint addresses of the two loose bodies, which the scene fixes.
TRAY_QPOS = slice(18, 25)
TRAY_QVEL = slice(18, 24)


@dataclass(frozen=True)
class LayoutRanges:
    """How far the tray's starting pose may be jittered, about the keyframe.

    The bounds are measured, and the x range is **asymmetric in the direction
    that is easy to get backwards**. The static IK note -- handle pairs solved
    cleanly to x = 0.35, degrading by x = 0.40 -- describes reaching *in* to a
    tray on the table, and it suggests there is room to move the tray closer and
    none to move it away. The carry measures the opposite. Flying three
    randomized plans at each offset, with the tray lifted and moving:

    ========  ==================  ==================
    x offset  tray tracking       ball
    ========  ==================  ==================
    -0.04     7.0-9.5 mm          lost on 2 of 3
    -0.03     7.2-16.3 mm         lost on 1 of 3
    -0.02     7.2-15.1 mm         kept
    0.00      2.1-6.4 mm          kept
    +0.02     2.5-3.9 mm          kept
    +0.05     2.0-3.3 mm          kept
    ========  ==================  ==================

    Pulling the tray **closer** is what hurts: the arms fold up, tracking
    degrades, and the tray jerks hard enough to throw the ball. Reaching further
    out is free over the range tested. So the envelope leans outward.

    A handle sits 0.15 m out along y, so a tray yaw of ``t`` also moves the far
    handle ``0.15 sin(t)`` along x -- 7.5 mm at the yaw bound, which is inside
    the slack the x bound leaves.
    """

    #: Metres along world x, added to the authored 0.35.
    x: tuple[float, float] = (-0.015, 0.04)
    #: Metres along world y. Moves both handles together, to |y| in 0.13-0.17.
    y: tuple[float, float] = (-0.02, 0.02)
    #: Radians about world z. At 0.05 rad the far handle moves 7.5 mm along x.
    yaw: tuple[float, float] = (-0.05, 0.05)

#: Gains of the classical ball balancer. Named rather than left as method
#: defaults because a vision *estimator* driving the same law has to use the same
#: numbers -- that is the whole point of estimating the state instead of
#: regressing the tilt, so a second copy of these would silently break it.
BALANCE_KP = 12.0
BALANCE_KD = 4.5
BALANCE_MAX_TILT = 0.20


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix for a unit quaternion."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=np.float64))
    return mat.reshape(3, 3)


def tilt_quat(roll: float, pitch: float) -> np.ndarray:
    """Quaternion for a small roll about x then pitch about y.

    The inverse of the 2-D setpoint a balancing policy emits, so a learned
    ``(roll, pitch)`` can be handed to :meth:`BimanualTrayCarry.command_tray`
    exactly as the classical controller's is.
    """
    qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0])
    qp = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    return quat_mul(qp, qr)


def balance_law(
    ball_xy: np.ndarray,
    ball_vxy: np.ndarray,
    *,
    kp: float = BALANCE_KP,
    kd: float = BALANCE_KD,
    max_tilt: float = BALANCE_MAX_TILT,
    feedforward: np.ndarray | None = None,
) -> np.ndarray:
    """Return the PD balance law's tilt: ball state in, ``(roll, pitch)`` out.

    Separated from :meth:`BimanualTrayCarry.balance_setpoint` so that a vision
    policy which *estimates* the ball state can be run through exactly this
    arithmetic instead of learning a tilt directly. That is worth the indirection:
    a regressor trained on tilt with an MSE loss shrinks its outputs toward the
    mean -- measured at roughly half the required magnitude -- and half the loop
    gain is a qualitatively different controller. Estimating the state keeps the
    gains exact by construction, and leaves the error where it can be read in
    millimetres.
    """
    accel = -kp * np.asarray(ball_xy, dtype=np.float64) - kd * np.asarray(
        ball_vxy, dtype=np.float64
    )
    if feedforward is not None:
        accel = accel + np.asarray(feedforward, dtype=np.float64)
    pitch = np.clip(accel[0] / GRAVITY, -max_tilt, max_tilt)
    roll = np.clip(-accel[1] / GRAVITY, -max_tilt, max_tilt)
    return np.array([roll, pitch])


def tray_quat(roll: float, pitch: float, yaw: float = 0.0) -> np.ndarray:
    """Full commanded tray orientation: a balance tilt inside a yaw goal.

    Order matters. The balance law works entirely in the **tray frame** -- the
    ball's offset and velocity are expressed there -- so its roll and pitch are
    about the tray's own axes, not the world's. Composing as ``yaw * tilt``
    applies the tilt in the yawed frame, which is what keeps the balancer correct
    once the tray is turned. The other order would silently rotate the balancer's
    feedback by the yaw angle and turn a stable loop into a spiral.

    Yaw is free of the ball dynamics in a way roll and pitch are not: rotating
    about the gravity vector adds no in-plane acceleration, so a yaw goal composes
    with balancing instead of competing with it.
    """
    qy = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return quat_mul(qy, tilt_quat(roll, pitch))


class BimanualTrayCarry:
    """Grasp a tray with both arms, then carry and balance a ball on it."""

    #: Cameras a vision policy sees: the angled view that shows the tray face and
    #: the ball, and the overhead view that resolves the ball's in-plane offset.
    VISION_CAMERAS = ("balancecam", "topcam")

    def __init__(
        self,
        scene_path: str,
        *,
        control_hz: float = 50.0,
        camera_names: tuple[str, ...] = (),
        image_size: tuple[int, int] = (84, 112),
        camera_gain_noise: float = 0.0,
        camera_read_noise: float = 0.0,
        torque_control: bool = False,
        torque_limit_scale: float = 1.0,
        latency_steps: int = 0,
        backlash: float = 0.0,
    ) -> None:
        """Load the scene, prepare the IK solver, and pick the actuation model.

        The sim-to-real arguments all default to off, so every number measured for
        the position-controlled carry stays reproducible:

        * ``torque_control`` replaces the scene's position actuators with
          :class:`~openarm_gym.realism.JointTorqueServo`. The grip becomes a
          commanded torque, which is what makes :meth:`regulate_grip` meaningful
          -- see its docstring.
        * ``torque_limit_scale`` shrinks every joint's usable torque, as a
          fraction of the modelled DM motor's range. Only has an effect under
          ``torque_control``, since otherwise MuJoCo's own ``forcerange`` applies.
        * ``latency_steps`` and ``backlash`` add command delay and transmission
          slack, independently per arm.
        """
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ik = PoseController(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.cameras = CameraRig(
            self.model,
            camera_names,
            image_size=image_size,
            gain_noise=camera_gain_noise,
            read_noise=camera_read_noise,
        )

        self._tray = self._body("tray")
        self._ball = self._body("ball")
        self._handle = {
            "left": self._geom("handle_left"),
            "right": self._geom("handle_right"),
        }
        #: Pad pose relative to the tray frame, captured when the grasp closes.
        self._grasp_rel: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        #: Live finger command per hand, moved by the grip force regulator.
        self._grip_cmd: dict[str, float] = dict(SHUT)
        #: Last two commanded tray positions, for the acceleration feedforward.
        self._cmd_history: list[np.ndarray] = []

        # Actuator-level realism, one model per arm so that their delay lines and
        # backlash states stay independent.
        self.actuators = {
            side: ActuatorModel(latency_steps=latency_steps, backlash=backlash)
            for side in ("left", "right")
        }
        self.torque: dict[str, JointTorqueServo] | None = None
        self._nominal_kp: dict[str, np.ndarray] = {}
        if torque_control:
            self.torque = {
                side: self._build_torque_servo(side, torque_limit_scale)
                for side in ("left", "right")
            }
            for servo in self.torque.values():
                servo.engage()
            self._nominal_kp = {s: v.kp.copy() for s, v in self.torque.items()}
        self._dynamics: DynamicsRandomizer | None = None
        #: Called after every control step, including those inside `grasp`.
        #: `run_plan` records the carry itself, but the approach and the close
        #: happen inside the controller, so a demo that wants to *show* the
        #: grasp has no other way to reach those steps.
        self.on_step: Callable[[BimanualTrayCarry], None] | None = None

    def _build_torque_servo(self, side: str, limit_scale: float) -> JointTorqueServo:
        """Build one arm's torque servo over its seven joints plus its finger.

        The finger joins the same servo because commanding a position SHUT against
        a post wider than the closed jaws simply saturates the actuator, so the
        PD-plus-limit form reproduces the measured 42 N grip while making the
        commanded grip *torque* the quantity actually being set.
        """
        actuators = np.array(
            list(range(CTRL[side].start, CTRL[side].stop)) + [FINGER_CTRL[side]],
            dtype=np.intp,
        )
        finger_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint1"
        )
        dofs = np.append(
            self.ik.arm_dof_indices(side), self.model.jnt_dofadr[finger_joint]
        )
        qpos = np.append(
            self.ik.arm_qpos_indices(side), self.model.jnt_qposadr[finger_joint]
        )
        return JointTorqueServo(
            self.model, actuators, dofs, qpos, torque_limit_scale=limit_scale
        )

    def randomize_dynamics(
        self, rng: np.random.Generator, ranges: DynamicsRanges | None = None
    ) -> dict[str, np.ndarray]:
        """Resample mass, friction, actuator gain and damping; return the scales.

        Call after :meth:`reset` and before :meth:`grasp`, so that one episode is
        flown entirely under one draw. Scoped to the bodies and geoms the task
        actually involves, plus the arm's own actuators and DOFs -- randomizing
        the floor's friction would be noise in the literal sense.
        """
        if self._dynamics is None:
            self._dynamics = DynamicsRandomizer(
                self.model,
                bodies=(self._tray, self._ball),
                geoms=(
                    self._geom("tray_top"),
                    self._handle["left"],
                    self._handle["right"],
                    self._geom("ball_geom"),
                ),
                dofs=np.arange(18, dtype=np.intp),
                ranges=ranges or DynamicsRanges(),
            )
        applied = self._dynamics.sample(rng)
        if self.torque is not None:
            # engage() zeroed the position gains the randomizer just scaled, so
            # the torque law would never see them; scale the servos' own gains by
            # the same draw instead. Actuator order is left 0-7 then right 8-15,
            # which is the order each servo was built in.
            for side, lo in (("left", 0), ("right", 8)):
                self.torque[side].kp = (
                    self._nominal_kp[side] * applied["gain"][lo : lo + 8]
                )
        return applied

    # -- model lookups ---------------------------------------------------------

    def _body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _geom(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    # -- state -----------------------------------------------------------------

    @property
    def tray_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and orientation of the tray."""
        return self.data.xpos[self._tray].copy(), self.data.xquat[self._tray].copy()

    @property
    def grasped(self) -> bool:
        """Whether the jaws have closed and the grasp transform is captured."""
        return bool(self._grasp_rel)

    def tray_yaw(self) -> float:
        """Return the tray's heading about the world z axis, in radians.

        Read from the rotation matrix rather than the quaternion so that a tray
        which is also tilted still reports the heading alone.
        """
        rot = _quat_to_mat(self.tray_pose[1])
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def handle_pos(self, side: str) -> np.ndarray:
        """World position of one handle post."""
        return self.data.geom_xpos[self._handle[side]].copy()

    def ball_in_tray(self) -> tuple[np.ndarray, np.ndarray]:
        """Ball position and velocity expressed in the tray frame."""
        tray_pos, tray_quat = self.tray_pose
        rot = _quat_to_mat(tray_quat)
        rel = rot.T @ (self.data.xpos[self._ball] - tray_pos)
        # Free-joint velocities: tray at qvel[18:24], ball at qvel[24:30].
        vel = rot.T @ (self.data.qvel[24:27] - self.data.qvel[18:21])
        return rel, vel

    def ball_on_tray(self, half_x: float = 0.075, half_y: float = 0.15) -> bool:
        """Whether the ball is still on the tray's top face."""
        rel, _ = self.ball_in_tray()
        return bool(
            abs(rel[0]) < half_x + 0.02
            and abs(rel[1]) < half_y + 0.02
            and rel[2] > 0.0
        )

    # -- low-level control -----------------------------------------------------

    def observe(
        self, rng: np.random.Generator | None = None, *, pixels: bool = True
    ) -> dict:
        """Return what a vision policy is allowed to see.

        Deliberately **excludes the ball's state**, which is the whole point:
        :meth:`balance_tilt` reads the ball's pose out of the simulator, and a
        policy that imitates it has to recover that from pixels instead. The
        proprioceptive half is honest -- joint angles are measured on the real
        arm, and the tray pose follows from them by forward kinematics once the
        grasp is closed, so it is not privileged information.

        ``rng`` is passed through to the camera sensor model; ``None`` renders
        clean pixels. ``pixels=False`` skips rendering entirely, for the callers
        that only want the proprioceptive half -- rendering is by far the most
        expensive part of a control step.
        """
        tray_pos, tray_quat = self.tray_pose
        obs: dict = {
            "qpos": self.data.qpos[:18].astype(np.float32).copy(),
            "tray_pos": tray_pos.astype(np.float32),
            "tray_quat": tray_quat.astype(np.float32),
        }
        if pixels and self.cameras:
            obs["pixels"] = self.cameras.render_all(self.data, rng)
        return obs

    def nudge_ball(self, kick_xy: np.ndarray) -> None:
        """Shove the ball in the tray's plane, mid-episode.

        A disturbance the controller has to recover from is what separates active
        control from a static hold, and it is the cheapest way to make the task
        read as real.

        The kick is a **velocity change in m/s, in the tray frame**, so a
        disturbance means the same thing however the tray is tilted. Velocity
        rather than impulse because the ball weighs 2.73 g: a plausible-looking
        impulse of 0.004 N*s is a 1.5 m/s launch, and the same impulse would also
        stop meaning the same disturbance once dynamics randomization starts
        scaling the ball's mass. Measured recovery limits for the classical
        balancer: 0.30 m/s along the tray's short axis, 0.20 m/s along the long
        one, which is the harder direction because the carry is already moving
        that way.

        Applied to the velocity directly -- ``xfrc_applied`` would be cleared by
        the next :func:`mujoco.mj_step` anyway.
        """
        _tray_pos, tray_quat = self.tray_pose
        rot = _quat_to_mat(tray_quat)
        self.data.qvel[24:27] += rot @ np.array(
            [kick_xy[0], kick_xy[1], 0.0], dtype=np.float64
        )

    def close(self) -> None:
        """Release the camera rig's renderer."""
        self.cameras.close()


    def _site_target(self, pad_target: np.ndarray) -> np.ndarray:
        """Wrist-site target that places the finger pads at ``pad_target``."""
        return np.asarray(pad_target, dtype=np.float64) - TOOL_DIR * PAD_OFFSET

    def pad_pos(self, side: str) -> np.ndarray:
        """Return the current finger-pad centre for one arm."""
        return self.ik.grasp_point(self.data, side, PAD_OFFSET)

    def _servo(self, pad_targets: dict, quats: dict, grip: dict) -> None:
        """One control step: IK both arms, hold the grip, advance the sim."""
        targets = {}
        for side in ("left", "right"):
            q, _res = self.ik.solve(
                self.data, side, self._site_target(pad_targets[side]), quats[side]
            )
            # Eight values per arm, seven joints plus the finger, so that latency
            # and backlash reach the grip as well as the reach.
            targets[side] = self.actuators[side].apply(np.append(q, grip[side]))
            self.data.ctrl[CTRL[side]] = targets[side][:7]
            self.data.ctrl[FINGER_CTRL[side]] = targets[side][7]
        for _ in range(self.n_substeps):
            # Gravity compensation on the arm DOFs, as the real driver applies.
            # Without it the wrists sag and the tray tilts on its own.
            self.data.qfrc_applied[:18] = self.data.qfrc_bias[:18]
            if self.torque is not None:
                # Recomputed every substep: a 1 kHz torque loop under a 50 Hz
                # command stream, which is how the real driver runs. The servo's
                # torque already carries the bias term for its own DOFs.
                for side, servo in self.torque.items():
                    self.data.qfrc_applied[servo.dofs] = servo.torque(
                        self.data, targets[side]
                    )
            mujoco.mj_step(self.model, self.data)
        if self.on_step is not None:
            self.on_step(self)

    def _sweep(self, goal: dict, grip: dict, steps: int, quats: dict | None = None) -> None:
        """Interpolate the pads from where they are to ``goal`` over ``steps``."""
        quats = quats or {s: TOOL_QUAT for s in ("left", "right")}
        start = {s: self.pad_pos(s).copy() for s in ("left", "right")}
        for i in range(steps):
            a = (i + 1) / steps
            targets = {s: start[s] * (1 - a) + goal[s] * a for s in ("left", "right")}
            self._servo(targets, quats, grip)

    # -- the sequence ----------------------------------------------------------

    def reset(self) -> None:
        """Place the arms retracted and clear of the tray, jaws open.

        The scene's home keyframe puts the closed jaws inside the handle posts;
        starting there makes the contact impulse throw the tray across the room.
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._grasp_rel.clear()
        self._grip_cmd = dict(SHUT)
        self._cmd_history.clear()
        for actuator in self.actuators.values():
            actuator.reset()
        for side in ("left", "right"):
            sign = 1.0 if side == "left" else -1.0
            target = self._site_target([0.24, sign * 0.22, 0.58])
            q, _ = self.ik.solve(self.data, side, target, TOOL_QUAT, iters=200)
            self.data.qpos[QPOS[side]] = q
            self.data.ctrl[CTRL[side]] = q
            for qi in FINGER_QPOS[side]:
                self.data.qpos[qi] = OPEN[side]
            self.data.ctrl[FINGER_CTRL[side]] = OPEN[side]
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def randomize_layout(
        self, rng: np.random.Generator, ranges: LayoutRanges | None = None
    ) -> dict[str, float]:
        """Jitter where the tray starts, and re-seat the ball on it.

        Call after :meth:`reset` and before :meth:`grasp` -- which is what
        ``run_plan``'s ``on_reset`` hook is for. It has to be before the grasp
        because the grasp is what adapts: :meth:`grasp` reads the handle
        positions out of the simulator and approaches wherever they are, so a
        moved tray is still grasped, and every plan is expressed as an **offset
        from the tray pose at grasp time** so the manoeuvre is unchanged too.

        Why this matters more than it looks: ``reset`` restores one keyframe, so
        until now the tray started in exactly the same place in every episode
        ever recorded or evaluated. The learned estimator has therefore never
        seen a different starting configuration, and a detector that has only
        ever seen one tray pose may be reading the image or may be reading the
        scene's layout -- there is no way to tell them apart from a dataset that
        contains one layout.

        Draw ``rng`` from a generator **spawned separately** from the visual and
        dynamics ones. Layout moves the tray, and two streams sharing a source is
        how enabling one randomizer silently changes what another produces; this
        repository has already been bitten by it once.

        Returns:
            The offsets actually applied, in metres and radians.

        """
        ranges = ranges or LayoutRanges()
        dx = float(rng.uniform(*ranges.x))
        dy = float(rng.uniform(*ranges.y))
        yaw = float(rng.uniform(*ranges.yaw))

        # From the scene as authored, never from the live pose: sampling around
        # wherever the tray happens to be lets repeated episodes random-walk the
        # layout out of the IK envelope.
        home = self.model.key_qpos[0, TRAY_QPOS]
        self.data.qpos[TRAY_QPOS.start : TRAY_QPOS.start + 3] = home[:3] + [dx, dy, 0.0]
        spin = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.data.qpos[TRAY_QPOS.start + 3 : TRAY_QPOS.stop] = quat_mul(spin, home[3:])
        self.data.qvel[TRAY_QVEL] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # The ball was seated on the tray's old pose; left where it is it would
        # be resting on the table, and the grasp would then lift the tray out
        # from under it.
        self.place_ball()
        return {"x": dx, "y": dy, "yaw": yaw}

    def grasp(
        self,
        *,
        approach_lift: float = APPROACH_LIFT,
        close_lift: float = CLOSE_LIFT,
        handles: dict[str, np.ndarray] | None = None,
    ) -> dict:
        """Approach the handles from behind, close the jaws, and record the grasp.

        The approach reaches in **above** the handle centre and then settles to
        ``close_lift`` before closing. That detour is not stylistic: driving
        straight in at the handle centre puts the gripper's ``ee_base_link`` --
        its palm, not its fingers -- in contact with the tray's top face, and the
        reach then bulldozes the whole tray 30 mm forward before the jaws ever
        reach the posts. Reaching in 20 mm higher clears the palm and cuts the
        displacement to about 5 mm.

        Args:
            approach_lift: height above the handle centre to reach in at, metres.
                Zero reproduces the original bulldozing approach.
            close_lift: height above the handle centre to close the jaws at.
                Closing back at the centre puts the pads on the widest part of
                the post.
            handles: where the posts are, per side, in world coordinates.
                ``None`` reads them out of the simulator, which is privileged
                state and is the reason this argument exists: pass a *vision*
                estimate instead and the grasp becomes a perception problem
                rather than a scripted one. Nothing else about the approach
                changes, so the two are directly comparable on the same layout.
                Only the estimate is used -- the controller never peeks at the
                true positions to correct it mid-approach.

        Note the honest cost: the clean approach holds the post with **two**
        finger contacts per hand at 89 N, where the bulldozing one held it with
        four at 84 N. The extra two were a consequence of the fault, not a
        feature -- shoving the tray 30 mm forward drove the post deep between the
        fingers. Reaching deliberately further in reproduces the contacts and the
        displacement together, in a 1:1 trade, so there is nothing to recover
        there. Two contacts on a flat face is still form closure; one contact on
        a cylinder was what previously failed.

        Returns:
            A report of the resulting contact count and normal force per hand.

        """
        handles = (
            {s: np.asarray(handles[s], dtype=np.float64) for s in ("left", "right")}
            if handles is not None
            else {s: self.handle_pos(s) for s in ("left", "right")}
        )
        behind_x = 0.23

        def at(dz: float) -> dict:
            return {
                s: np.array([handles[s][0], handles[s][1], handles[s][2] + dz])
                for s in handles
            }

        self._sweep({s: np.array([behind_x, handles[s][1], 0.52]) for s in handles},
                    OPEN, int(0.4 * self.control_hz * 8))
        self._sweep(
            {s: np.array([behind_x, handles[s][1], handles[s][2] + approach_lift])
             for s in handles},
            OPEN, int(0.4 * self.control_hz * 8),
        )
        self._sweep(at(approach_lift), OPEN, int(0.5 * self.control_hz * 8))
        if close_lift != approach_lift:
            # Straight down, with the jaws still open. A vertical settle presses
            # on a tray that is still resting on the table; it does not push it.
            self._sweep(at(close_lift), OPEN, int(0.3 * self.control_hz * 8))
        self._sweep(at(close_lift), SHUT, int(0.5 * self.control_hz * 8))

        # Freeze the grasp: where each pad sits in the tray's frame.
        tray_pos, tray_quat = self.tray_pose
        rot = _quat_to_mat(tray_quat)
        for side in ("left", "right"):
            pad = self.pad_pos(side)
            _wrist, wq = self.ik.pose(self.data, side)
            self._grasp_rel[side] = (
                rot.T @ (pad - tray_pos),
                quat_mul(quat_conj(tray_quat), wq),
            )
        return self.grip_report()

    def grip_report(self) -> dict:
        """Contact count and total normal force between each hand and its post."""
        out = {s: {"contacts": 0, "force": 0.0} for s in ("left", "right")}
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""
            pair = f"{n1}|{n2}"
            if "finger" not in pair:
                continue
            for side in ("left", "right"):
                if f"handle_{side}" in pair:
                    ft = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, c, ft)
                    out[side]["contacts"] += 1
                    out[side]["force"] += abs(float(ft[0]))
        return out

    def regulate_grip(self, target_force: float = 40.0, gain: float = 2e-4) -> dict:
        """Close the loop on grip force by trimming the finger command.

        **Off by default, because measurement says it does not help here.** The
        posts are far wider than the jaws' closed aperture, so commanding SHUT
        already saturates the position error and the grip sits at the actuator's
        force limit. There is no headroom to squeeze harder -- a regulator can
        only open the jaws -- and enabling it made the worst-case grip fall from
        one finger contact to zero, i.e. it dropped the tray.

        It is kept because it becomes the right tool once the handles have
        proper form closure (a flange or groove) rather than relying on friction
        alone, and because real hardware must regulate force rather than command
        position into a rigid object. Raise ``target_force`` above the saturated
        value to make it a crush limiter rather than a grip booster.

        **Under ``torque_control`` the saturation is the feature, not the
        problem.** The finger's commanded torque is clipped to
        ``torque_limit_scale`` of the DM motor's 7 N*m range, and because the
        command saturates, that limit *is* the grip force. Measured on the
        randomized carries:

        ===================  ==========
        ``torque_limit_scale``  grip force
        ===================  ==========
        1.00                  41.7 N
        0.40                  30.3 N
        0.25                  18.9 N
        ===================  ==========

        So a commanded grip force is available today by setting the limit, which
        is what the real driver does, and this regulator is only needed for the
        case where the grip must be trimmed *below* saturation in closed loop.
        The task itself fails somewhere between 0.40 and 0.25: at 25% the carry
        drops to 3/6.
        """
        report = self.grip_report()
        for side in ("left", "right"):
            error = target_force - report[side]["force"]
            # OPEN and SHUT have opposite signs per hand, so step toward SHUT.
            direction = -np.sign(OPEN[side])
            self._grip_cmd[side] = float(
                np.clip(
                    self._grip_cmd[side] + direction * gain * error,
                    min(OPEN[side], SHUT[side]),
                    max(OPEN[side], SHUT[side]),
                )
            )
        return report

    def commanded_accel(self) -> np.ndarray:
        """Acceleration of the commanded tray path, from the last three commands."""
        if len(self._cmd_history) < 3:
            return np.zeros(3)
        p0, p1, p2 = self._cmd_history[-3:]
        return (p2 - 2.0 * p1 + p0) * (self.control_hz ** 2)

    def command_tray(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        *,
        grip_force: float | None = None,
    ) -> None:
        """Drive both arms so the grasped tray reaches the given pose.

        This is the coordination step: one tray pose in, both arms' targets out,
        derived from the grasp transform captured when the jaws closed.

        ``grip_force`` enables :meth:`regulate_grip`, which is off by default --
        see that method for why it does not currently help.
        """
        if not self._grasp_rel:
            raise RuntimeError("command_tray called before grasp()")
        pos = np.asarray(pos, dtype=np.float64)
        self._cmd_history.append(pos.copy())
        if len(self._cmd_history) > 3:
            self._cmd_history.pop(0)

        if grip_force is not None:
            self.regulate_grip(grip_force)
            grip = dict(self._grip_cmd)
        else:
            grip = SHUT

        rot = _quat_to_mat(quat)
        targets, quats = {}, {}
        for side in ("left", "right"):
            rel_pos, rel_quat = self._grasp_rel[side]
            targets[side] = pos + rot @ rel_pos
            quats[side] = quat_mul(quat, rel_quat)
        self._servo(targets, quats, grip)

    def place_ball(self, offset_xy: np.ndarray | None = None) -> None:
        """Seat the ball on the tray, at rest, in the tray's own frame.

        This is the episode's **initial condition**, and
        :func:`~openarm_gym.control.tray_task.run_plan` calls it with the plan's
        sampled offset so that each episode starts the balancer somewhere
        different. It is not a workaround: with the original smooth cylindrical
        handles the approach rolled the ball off before the jaws closed and tests
        had to re-seat it, but box handles made the tray slide rather than tilt
        and the ball now rides along, moving about 3 mm. Calling it with no
        offset simply re-centres the ball.
        """
        tray_pos, tray_quat = self.tray_pose
        rot = _quat_to_mat(tray_quat)
        local = np.array([0.0, 0.0, 0.025])          # tray half-thickness + radius
        if offset_xy is not None:
            local[:2] += np.asarray(offset_xy, dtype=np.float64)
        self.data.qpos[25:28] = tray_pos + rot @ local
        self.data.qpos[28:32] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[24:30] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def balance_setpoint(self, kp: float = BALANCE_KP, kd: float = BALANCE_KD,
                         max_tilt: float = BALANCE_MAX_TILT,
                         feedforward: bool = False) -> np.ndarray:
        """``(roll, pitch)`` in radians that drives the ball back to centre.

        The two-number form of :meth:`balance_tilt`, split out because this is
        exactly what a learned policy has to produce: the setpoint is 2-D, while
        the quaternion it becomes is 4-D with three constraints. Behaviour
        cloning regresses these two numbers.

        Returns level once the ball is off the tray, for the reason given in
        :meth:`balance_tilt`.
        """
        if not self.ball_on_tray():
            return np.zeros(2)
        rel, vel = self.ball_in_tray()
        feed = self.commanded_accel()[:2] if feedforward else None
        return balance_law(rel[:2], vel[:2], kp=kp, kd=kd, max_tilt=max_tilt, feedforward=feed)

    def balance_tilt(self, kp: float = BALANCE_KP, kd: float = BALANCE_KD,
                     max_tilt: float = BALANCE_MAX_TILT,
                     feedforward: bool = False) -> np.ndarray:
        """Tilt that drives the ball back to the tray centre (PD on ball state).

        ``feedforward`` adds the commanded tray acceleration, since a tray
        accelerating at ``a`` pushes the ball backwards in its own frame. It is
        off by default, and the reason is worth recording: with the original
        smooth cylindrical handles it looked essential, lifting a carry from
        4.9 cm of a commanded 10 cm to the full 10 cm. But that shortfall was
        the tray slipping in a weak friction grasp, not the control law -- the
        controller was compensating for a bad grasp. With box handles giving
        form closure, the reactive PD alone tracks the full 10 cm, and the
        feedforward's extra tilt authority instead costs grip margin
        (worst-case finger contacts fall from nine to four).

        Returns level if the ball is already off the tray: chasing a ball that
        has gone saturates the tilt and tears the grasp off the posts.
        """
        roll, pitch = self.balance_setpoint(kp, kd, max_tilt, feedforward)
        return tilt_quat(roll, pitch)
