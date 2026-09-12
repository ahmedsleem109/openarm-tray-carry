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

"""Carry plans, ball disturbances, and the single rollout driver they feed.

Everything that runs a tray-carry episode goes through :func:`run_plan`: data
collection, evaluating a learned policy against the classical one, and rendering
video. They differ only in arguments, which is the point -- a separate rollout
loop per purpose is how an evaluation quietly stops measuring the same task the
data was collected on.

A :class:`CarryPlan` is expressed as **offsets from the tray pose at grasp time**
rather than absolute positions, so the same plan is valid whatever the scene's
layout randomization did to where the tray started.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..vision import CameraRig
from .bimanual_tray import BimanualTrayCarry, tray_quat

#: Ramp length, in control steps, used to reach a waypoint. Commanding a 10 cm
#: step in one control step jerks the tray hard enough to throw the ball off, so
#: every waypoint is approached along a ramp and then held.
DEFAULT_RAMP = 40

#: A policy maps one observation to a ``(roll, pitch)`` tilt setpoint.
TiltPolicy = Callable[[dict], np.ndarray]


@dataclass(frozen=True)
class Waypoint:
    """One tray position target, held for a while after being reached."""

    #: Target position relative to the tray's pose when the grasp closed.
    offset: np.ndarray
    #: Total control steps spent on this waypoint, ramp included.
    steps: int
    #: Steps at the end of the window during which the target is already reached.
    ramp: int = DEFAULT_RAMP
    #: Tray yaw goal at this waypoint, radians about the world z axis. Ramped
    #: alongside the position, and composed *outside* the balance tilt -- see
    #: :func:`~openarm_gym.control.bimanual_tray.tray_quat`.
    yaw: float = 0.0


@dataclass(frozen=True)
class CarryPlan:
    """A full episode: where to carry the tray, and what goes wrong on the way."""

    waypoints: tuple[Waypoint, ...]
    #: Where the ball is seeded on the tray face, in the tray frame.
    ball_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: Control step -> in-plane velocity kick (m/s, tray frame) given to the ball.
    disturbances: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        """Total control steps in the episode."""
        return sum(w.steps for w in self.waypoints)


def straight_carry(lift: float = 0.10, sideways: float = 0.10) -> CarryPlan:
    """Return the fixed lift-then-carry manoeuvre the controller tests use.

    Kept as a named plan so the learned policy is evaluated on exactly the
    manoeuvre the classical baseline is asserted against.
    """
    return CarryPlan(
        waypoints=(
            Waypoint(np.array([0.0, 0.0, lift]), 120),
            Waypoint(np.array([0.0, sideways, lift]), 200),
        ),
        ball_offset=np.array([0.02, 0.04]),
    )


def random_carry(
    rng: np.random.Generator,
    *,
    n_waypoints: int = 2,
    lift: tuple[float, float] = (0.06, 0.13),
    reach: float = 0.10,
    steps: tuple[int, int] = (140, 220),
    ball_offset: float = 0.035,
    n_disturbances: int = 1,
    kick: tuple[float, float] = (0.10, 0.22),
    yaw: float = 0.0,
) -> CarryPlan:
    """Sample a carry: a lift, then waypoints in the tray's plane, plus shoves.

    The envelopes are deliberately narrow, and both bounds were measured rather
    than guessed:

    * IK reaches handle pairs cleanly at x = 0.30-0.35 and degrades to a 0.03
      residual by x = 0.40, so a plan that wandered further would be measuring
      the IK's edge rather than the balancer.
    * Travel along **x is limited hardest**, because x is the tray's *short*
      axis: the ball has 7.5 cm to the edge there against 15 cm along y. A
      +-4 cm x leg was enough to lose the ball with no disturbance at all.
    * ``kick`` stays inside what the classical balancer recovers from (0.30 m/s
      short-axis, 0.20 m/s long-axis), because behaviour cloning can only learn
      from episodes the expert survived.
    * ``yaw`` is the half-width of the tray rotation goal, in radians, and is
      **off by default**. A handle sits 0.15 m out, so yawing by an angle t moves
      it 0.15 sin(t) along x: at 10 degrees that is 26 mm, which already pushes
      the far handle to x = 0.38 where the IK residual starts to climb.
    """
    lift_h = float(rng.uniform(*lift))
    waypoints = [Waypoint(np.array([0.0, 0.0, lift_h]), int(rng.integers(110, 150)))]
    here = np.array([0.0, 0.0, lift_h])
    for _ in range(n_waypoints):
        # Stay within the reach envelope by sampling a displacement and clipping
        # the resulting position, rather than sampling an absolute target.
        step = rng.uniform(-reach, reach, size=3)
        step[2] *= 0.4
        nxt = here + step
        nxt[0] = float(np.clip(nxt[0], -0.025, 0.025))
        nxt[1] = float(np.clip(nxt[1], -0.11, 0.11))
        nxt[2] = float(np.clip(nxt[2], lift[0], lift[1]))
        waypoints.append(
            Waypoint(
                nxt.copy(),
                int(rng.integers(*steps)),
                yaw=float(rng.uniform(-yaw, yaw)) if yaw else 0.0,
            )
        )
        here = nxt

    total = sum(w.steps for w in waypoints)
    # Never in the first waypoint: the lift is still settling, and a shove there
    # measures the grasp rather than the balancer.
    first = waypoints[0].steps + 20
    last = max(total - 60, first + 1)
    # Sample the steps *without replacement*. Keying the schedule by step means
    # two draws landing on the same control step silently become one shove --
    # measured at 14 plans in 200 losing one of six.
    choices = np.arange(first, last)
    when = rng.choice(choices, size=min(n_disturbances, len(choices)), replace=False)
    disturbances: dict[int, np.ndarray] = {}
    for step in when:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        size = float(rng.uniform(*kick))
        disturbances[int(step)] = size * np.array([np.cos(angle), np.sin(angle)])

    return CarryPlan(
        waypoints=tuple(waypoints),
        ball_offset=rng.uniform(-ball_offset, ball_offset, size=2),
        disturbances=disturbances,
    )


@dataclass
class Rollout:
    """Everything one episode produced, whoever was driving the tilt."""

    survived: bool
    steps: int
    #: Control step at which the ball left the tray, or ``None``.
    lost_at: int | None
    #: (T, 2) ball offset in the tray frame -- the quantity vision must recover.
    ball_xy: np.ndarray
    #: (T, 2) ball velocity in the tray frame. Recorded because the balance law
    #: needs it: a policy that estimates position alone can only reproduce the P
    #: term, and the D term is what stops the ball overshooting.
    ball_vxy: np.ndarray
    #: (T, 2) tilt setpoint actually commanded, perturbation included.
    setpoints: np.ndarray
    #: (T, 2) tilt the classical controller would have commanded, always
    #: recorded so a learned policy's error is measurable in closed loop.
    labels: np.ndarray
    #: (T, 3) commanded tray position, as an offset from the grasp-time pose.
    positions: np.ndarray
    #: (T,) commanded tray yaw, radians.
    yaws: np.ndarray
    #: (T,) distance between the commanded and the achieved tray position.
    tracking: np.ndarray
    #: (T, 2) total finger normal force per hand.
    grip: np.ndarray
    #: (T, D) proprioception, matching :meth:`BimanualTrayCarry.observe`.
    proprio: np.ndarray
    #: camera name -> (T, H, W, 3) uint8, empty unless pixels were recorded.
    pixels: dict[str, np.ndarray] = field(default_factory=dict)
    #: Full-resolution video frames, empty unless a video camera was named.
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def mean_tracking(self) -> float:
        """Mean tray position tracking error, in metres."""
        return float(np.mean(self.tracking)) if len(self.tracking) else float("nan")

    @property
    def policy_error(self) -> float:
        """RMS difference between the commanded tilt and the classical one, rad."""
        if not len(self.setpoints):
            return float("nan")
        return float(np.sqrt(np.mean((self.setpoints - self.labels) ** 2)))


def flat_proprio(obs: dict) -> np.ndarray:
    """Flatten an observation's non-visual half into one vector.

    Fixed order, because a policy trained on one ordering and evaluated on
    another fails silently and looks like a modelling problem.
    """
    return np.concatenate(
        [obs["qpos"], obs["tray_pos"], obs["tray_quat"]]
    ).astype(np.float32)


def run_plan(
    carry: BimanualTrayCarry,
    plan: CarryPlan,
    *,
    tilt_policy: TiltPolicy | None = None,
    rng: np.random.Generator | None = None,
    on_reset: Callable[[BimanualTrayCarry], None] | None = None,
    handle_source: Callable[[BimanualTrayCarry], dict[str, np.ndarray]] | None = None,
    action_noise: float = 0.0,
    action_rng: np.random.Generator | None = None,
    record_pixels: bool = False,
    video: tuple[CameraRig, str] | None = None,
    stop_on_loss: bool = True,
) -> Rollout:
    """Grasp, then fly ``plan``, recording what happened at every control step.

    Args:
        carry: a controller. It is reset and re-grasped here, so one instance can
            fly many plans.
        plan: the manoeuvre and its disturbances.
        tilt_policy: given an observation, returns the ``(roll, pitch)`` setpoint.
            ``None`` uses :meth:`BimanualTrayCarry.balance_setpoint`, which is the
            classical baseline and the source of behaviour-cloning labels.
        rng: drives the camera sensor model. ``None`` renders clean pixels.
        on_reset: called with the controller after the reset and before the
            grasp. That window is the only place per-episode dynamics
            randomization can go -- :meth:`BimanualTrayCarry.randomize_dynamics`
            has to be applied to a settled model before the jaws close, or the
            grasp captures a transform under one draw and flies under another --
            and the reset and the grasp both live in here, so a caller has no
            other way to reach it.
        handle_source: given the controller, returns where the handle posts are.
            ``None`` lets :meth:`BimanualTrayCarry.grasp` read them out of the
            simulator, which is privileged state -- pass a vision estimator to
            make the grasp a perception problem. It is called once, after the
            reset and the layout draw and before the approach, which is the only
            moment at which the arms are clear of the tray and a camera can see
            both posts.
        action_noise: standard deviation, in radians, of a perturbation added to
            the tilt *actually commanded*, while ``labels`` keeps the clean
            setpoint. This is the repository's existing recipe for making
            demonstrations teach recovery, and here it is the difference between
            a usable dataset and an unusable one: a stabilising controller spends
            almost all its time at its own setpoint, so undisturbed
            demonstrations of this task carry an RMS ball offset of 10 mm against
            a 75 mm half-tray, and an estimator trained on them has never seen
            the states a closed loop visits in its first second. Shoving the ball
            barely helps -- the expert recentres it within a few tenths of a
            second -- but perturbing the *command* holds the ball off centre
            continuously, and every recorded label is then a correction from
            there.
        action_rng: generator for that perturbation. Kept separate from ``rng``
            on purpose: action noise moves the tray and camera noise does not, so
            sharing one stream would make enabling the camera sensor model change
            the physical trajectory -- the same class of bug as the visual
            randomization one this project already fixed.
        record_pixels: keep every frame of the rig's cameras in the result. A
            300-step episode at 84x112 over two cameras is about 17 MB, so this
            is opt-in -- RAM is the binding constraint on this machine.
        video: a ``(rig, camera name)`` pair to render a frame from at every
            control step. It takes its own rig because video wants a presentable
            resolution while the policy's cameras are deliberately small.
        stop_on_loss: end the episode once the ball is off the tray. The labels
            after that point are the level fallback and teach nothing. Set it
            ``False`` to fly the plan out regardless, which is what a
            side-by-side demo needs -- an empty tray still being carried as if
            nothing had happened is the clearest picture of what the controller
            is for.

    Returns:
        A :class:`Rollout` with per-step traces and the survival verdict.

    """
    carry.reset()
    if on_reset is not None:
        on_reset(carry)
    carry.grasp(handles=handle_source(carry) if handle_source is not None else None)
    if plan.ball_offset is not None:
        carry.place_ball(np.asarray(plan.ball_offset, dtype=np.float64))

    # Rendering dominates a control step, so only render when something consumes
    # the pixels: a driving policy, or the recorder.
    need_pixels = record_pixels or tilt_policy is not None
    if need_pixels and not carry.cameras:
        raise ValueError(
            "run_plan needs camera observations here (a tilt_policy is driving, or "
            "record_pixels is set) but the controller was built without any. "
            "Construct BimanualTrayCarry with camera_names=BimanualTrayCarry."
            "VISION_CAMERAS."
        )
    # A policy that differentiates its own estimates carries a control rate, and
    # a mismatch scales its damping term by the ratio with no other symptom.
    policy_hz = getattr(tilt_policy, "control_hz", None)
    if policy_hz is not None and abs(policy_hz - carry.control_hz) > 1e-9:
        raise ValueError(
            f"the tilt policy is configured for {policy_hz:g} Hz but the "
            f"controller runs at {carry.control_hz:g} Hz; pass "
            "control_hz=carry.control_hz when building the policy"
        )
    if action_noise > 0.0 and action_rng is None:
        action_rng = np.random.default_rng()
    origin = carry.tray_pose[0].copy()
    # The plan's yaw goals are offsets from the heading the tray was grasped at,
    # for the same reason its positions are offsets from where it was grasped.
    # Commanding an absolute yaw of zero instead orders the arms to untwist a
    # tray that started turned -- which is a jerk through the grasp on the first
    # control step and a standing fight afterwards. Measured on a layout drawn
    # 1.6 degrees off: tray tracking 4.5 mm -> 10.2 mm, and the ball is thrown
    # off well before the episode's disturbance arrives.
    yaw_origin = carry.tray_yaw()
    ball_xy, ball_vxy, setpoints, labels = [], [], [], []
    positions, yaws, tracking, grip, proprio = [], [], [], [], []
    pixel_frames: dict[str, list[np.ndarray]] = {n: [] for n in carry.cameras.camera_names}
    frames: list[np.ndarray] = []
    lost_at: int | None = None
    step = 0

    yaw_from = 0.0
    for waypoint in plan.waypoints:
        target = origin + waypoint.offset
        src = carry.tray_pose[0].copy()
        for i in range(waypoint.steps):
            if step in plan.disturbances:
                carry.nudge_ball(np.asarray(plan.disturbances[step], dtype=np.float64))

            alpha = min(1.0, (i + 1) / max(1, waypoint.steps - waypoint.ramp))
            commanded = src * (1.0 - alpha) + target * alpha
            yaw = yaw_origin + yaw_from * (1.0 - alpha) + waypoint.yaw * alpha

            # Observe before acting, so the recorded pixels are the ones the
            # policy's action was actually a response to -- and read the ball's
            # true state here too, at the same instant. Reading it after the step
            # instead paired every image with the *next* step's ball position,
            # which asks the estimator to predict 20 ms into the future and
            # quietly biases both the position and the velocity target.
            obs = carry.observe(rng, pixels=need_pixels)
            rel, vel = carry.ball_in_tray()
            label = carry.balance_setpoint()
            if tilt_policy is None:
                setpoint = label
            else:
                setpoint = np.asarray(tilt_policy(obs), dtype=np.float64).reshape(2)

            if action_noise > 0.0:
                setpoint = setpoint + action_rng.normal(0.0, action_noise, size=2)
            carry.command_tray(commanded, tray_quat(*setpoint, yaw))

            report = carry.grip_report()
            ball_xy.append(rel[:2].copy())
            ball_vxy.append(vel[:2].copy())
            setpoints.append(setpoint.copy())
            labels.append(label.copy())
            positions.append((commanded - origin).copy())
            yaws.append(yaw - yaw_origin)
            tracking.append(float(np.linalg.norm(carry.tray_pose[0] - commanded)))
            grip.append([report["left"]["force"], report["right"]["force"]])
            proprio.append(flat_proprio(obs))
            if record_pixels and "pixels" in obs:
                for name, image in obs["pixels"].items():
                    pixel_frames[name].append(image)
            if video is not None:
                video_rig, video_cam = video
                frames.append(video_rig.render(carry.data, video_cam))

            step += 1
            if not carry.ball_on_tray():
                # Record the loss once, and only stop if asked to. Breaking
                # regardless would make stop_on_loss=False a half-measure: the
                # waypoint still ended early, so two runs of one plan came back
                # different lengths and could not be compared frame by frame.
                if lost_at is None:
                    lost_at = step
                if stop_on_loss:
                    break
        yaw_from = waypoint.yaw
        if lost_at is not None and stop_on_loss:
            break

    return Rollout(
        survived=carry.ball_on_tray(),
        steps=step,
        lost_at=lost_at,
        ball_xy=np.asarray(ball_xy, dtype=np.float32),
        ball_vxy=np.asarray(ball_vxy, dtype=np.float32),
        setpoints=np.asarray(setpoints, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.float32),
        yaws=np.asarray(yaws, dtype=np.float32),
        tracking=np.asarray(tracking, dtype=np.float32),
        grip=np.asarray(grip, dtype=np.float32),
        proprio=np.asarray(proprio, dtype=np.float32),
        pixels={n: np.asarray(v, dtype=np.uint8) for n, v in pixel_frames.items() if v},
        frames=frames,
    )
