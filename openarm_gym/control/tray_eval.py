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

"""Named sim-to-real conditions, and the evaluation loop that flies them.

The realism machinery -- camera noise, torque actuation, torque limits, command
latency, transmission backlash and dynamics randomization -- is built and tested
in :mod:`openarm_gym.realism` and wired into
:class:`~openarm_gym.control.bimanual_tray.BimanualTrayCarry`, and all of it
defaults to off. Everything measured so far was therefore measured on clean
pixels and position control, the vision policy's headline survival rate
included. This module turns those knobs on **one named condition at a time** so
the drop can be reported rather than assumed.

A :class:`Condition` is the whole of what makes an episode harder, in one frozen
object, for two reasons:

* Three of the knobs are *constructor* arguments -- torque control, latency and
  backlash all change how a command reaches the joints, so they cannot be
  switched on an existing instance. A condition therefore owns a controller
  rather than configuring one, and the caller builds one per condition.
* The classical and the vision controller have to fly the **same** condition on
  the **same** seeds, or the comparison stops being one. Passing a single object
  to both is how that stays true when a knob is added later.

Dynamics randomization is the exception that shaped ``run_plan``'s ``on_reset``
hook: it is per-episode, and it has to land after the reset and before the
grasp, which is a window only the rollout driver holds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .bimanual_tray import BimanualTrayCarry
from .tray_task import CarryPlan, run_plan


@dataclass(frozen=True)
class Condition:
    """One sim-to-real setting: everything that makes an episode less ideal.

    Every field is off at its default, so ``Condition("clean")`` reproduces the
    position-controlled, clean-pixel numbers the rest of the project quotes.
    """

    name: str
    #: Per-episode multiplicative gain jitter and additive read noise on pixels.
    gain_noise: float = 0.0
    read_noise: float = 0.0
    #: Replace the scene's position actuators with a joint torque servo.
    torque_control: bool = False
    #: Usable fraction of the modelled DM motor's torque range. Only bites under
    #: ``torque_control``; otherwise MuJoCo's own ``forcerange`` applies.
    torque_limit_scale: float = 1.0
    #: Command delay, in control steps. One step is 20 ms at 50 Hz.
    latency_steps: int = 0
    #: Transmission slack, in radians.
    backlash: float = 0.0
    #: Resample mass, friction, actuator gain and damping once per episode.
    randomize_dynamics: bool = False
    #: Jitter where the tray starts. Not a sim-to-real knob -- it is scene
    #: variation, and it is kept out of the named presets for that reason -- but
    #: it belongs here because it composes with them the same way: per episode,
    #: in the window between the reset and the grasp.
    randomize_layout: bool = False

    @property
    def noisy_pixels(self) -> bool:
        """Whether this condition perturbs the camera images at all."""
        return self.gain_noise > 0.0 or self.read_noise > 0.0

    def summary(self) -> str:
        """One line naming only the knobs that are actually turned on."""
        parts = []
        if self.noisy_pixels:
            parts.append(f"camera gain {self.gain_noise:g} read {self.read_noise:g}")
        if self.torque_control:
            parts.append(f"torque at {self.torque_limit_scale * 100:.0f}% of rated")
        if self.latency_steps:
            parts.append(f"latency {self.latency_steps} step(s)")
        if self.backlash:
            parts.append(f"backlash {np.degrees(self.backlash):.2f} deg")
        if self.randomize_dynamics:
            parts.append("randomized dynamics")
        if self.randomize_layout:
            parts.append("randomized layout")
        return ", ".join(parts) if parts else "nothing enabled (baseline)"


#: The sweep. Each entry isolates one mechanism except ``all``, which is the
#: question actually being asked -- a real arm does not hand you its failure
#: modes one at a time. The magnitudes are the ones the classical controller was
#: already measured under (see STATUS.md), so the two tables line up: there,
#: 0.5 deg of backlash cost far more than 40 ms of latency did.
CONDITIONS: dict[str, Condition] = {
    "clean": Condition("clean"),
    "camera-noise": Condition("camera-noise", gain_noise=0.04, read_noise=2.0),
    "torque": Condition("torque", torque_control=True),
    "torque-40": Condition("torque-40", torque_control=True, torque_limit_scale=0.4),
    "latency-1": Condition("latency-1", latency_steps=1),
    "latency-2": Condition("latency-2", latency_steps=2),
    "backlash-0.5": Condition("backlash-0.5", backlash=float(np.radians(0.5))),
    "backlash-1.5": Condition("backlash-1.5", backlash=float(np.radians(1.5))),
    "dynamics": Condition("dynamics", randomize_dynamics=True),
    "all": Condition(
        "all",
        gain_noise=0.04,
        read_noise=2.0,
        torque_control=True,
        torque_limit_scale=0.4,
        latency_steps=1,
        backlash=float(np.radians(0.5)),
        randomize_dynamics=True,
    ),
}


def build_carry(
    scene_path: str,
    condition: Condition,
    *,
    camera_names: tuple[str, ...] = (),
    image_size: tuple[int, int] = (84, 112),
    control_hz: float = 50.0,
) -> BimanualTrayCarry:
    """Build the controller a condition asks for.

    Cameras are built even for the classical controller, which never reads them:
    an unrendered rig costs nothing, and one controller per *condition* rather
    than one per controller is what keeps the physics identical across a row.
    """
    return BimanualTrayCarry(
        scene_path,
        control_hz=control_hz,
        camera_names=camera_names,
        image_size=image_size,
        camera_gain_noise=condition.gain_noise,
        camera_read_noise=condition.read_noise,
        torque_control=condition.torque_control,
        torque_limit_scale=condition.torque_limit_scale,
        latency_steps=condition.latency_steps,
        backlash=condition.backlash,
    )


class LoggedVision:
    """A learned controller, logging how well its ball estimate tracks truth.

    The privileged ball state is read only to *report*; it is never fed back to
    the policy. That number is the one worth watching under a sensor model --
    survival says whether the loop held, and the sight error says whether it was
    the images or the actuators that broke.
    """

    def __init__(self, controller: Any, carry: BimanualTrayCarry) -> None:
        """Wrap a controller so each call records its sight error."""
        self.controller = controller
        self.carry = carry
        self.errors: list[float] = []

    @property
    def control_hz(self) -> float:
        """Forward the wrapped controller's rate, which ``run_plan`` checks.

        Without this the wrapper hides the attribute and the rate check passes
        vacuously -- which is the same silent damping-scale bug the check exists
        to catch.
        """
        return self.controller.control_hz

    def reset(self) -> None:
        """Clear the controller's frame history and this episode's log."""
        self.controller.reset()
        self.errors = []

    def __call__(self, obs: dict) -> np.ndarray:
        """Return the policy's setpoint, logging its ball-position error."""
        setpoint = self.controller(obs)
        truth = self.carry.ball_in_tray()[0][:2]
        self.errors.append(float(np.linalg.norm(self.controller.ball_estimate - truth)))
        return setpoint


def level_policy(_obs: dict) -> np.ndarray:
    """Never tilt. The floor: what the carry achieves with no balancing at all."""
    return np.zeros(2)


def episode_rngs(
    condition: Condition, seed: int, index: int
) -> tuple[np.random.Generator | None, Callable[[BimanualTrayCarry], None] | None]:
    """Return this episode's camera generator and its ``on_reset`` hook.

    Both are seeded from ``seed + index``, so episode ``i`` of one controller
    meets the same pixels and the same model as episode ``i`` of every other --
    which is the whole point of running the comparison on shared plans.

    The dynamics and layout draws are **spawned** rather than taken from the
    camera stream, and from each other. Both move the tray and camera noise does
    not, so a shared generator would make enabling the sensor model change the
    physics; this repository has been bitten by exactly that once already (see
    STATUS.md).
    """
    rng = np.random.default_rng(seed + index) if condition.noisy_pixels else None
    if not (condition.randomize_dynamics or condition.randomize_layout):
        return rng, None
    # Two spawned children, one per randomizer, so that turning either on leaves
    # the other's draws bit-for-bit identical.
    dyn_rng, layout_rng = np.random.default_rng(seed + index).spawn(2)

    def on_reset(carry: BimanualTrayCarry) -> None:
        # Layout first: it re-seats the ball, and a dynamics draw scales masses
        # and frictions that the re-seating does not depend on, so the order is
        # only fixed to keep the sequence reproducible.
        if condition.randomize_layout:
            carry.randomize_layout(layout_rng)
        if condition.randomize_dynamics:
            carry.randomize_dynamics(dyn_rng)

    return rng, on_reset


def evaluate_controller(
    carry: BimanualTrayCarry,
    plans: list[CarryPlan],
    policy: Any | None,
    *,
    condition: Condition,
    seed: int,
    label: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fly every plan with one controller and summarise what happened.

    ``policy`` is ``None`` for the classical baseline, which is ``run_plan``'s
    own default and therefore reads the ball's state out of the simulator.
    """
    survived, peaks, tilt_errors, sight_errors, tracking, grips = 0, [], [], [], [], []
    for i, plan in enumerate(plans):
        if hasattr(policy, "reset"):
            policy.reset()
        rng, on_reset = episode_rngs(condition, seed, i)
        rollout = run_plan(carry, plan, tilt_policy=policy, rng=rng, on_reset=on_reset)
        survived += rollout.survived
        peaks.append(float(np.abs(rollout.ball_xy).max()))
        tracking.append(rollout.mean_tracking)
        grips.append(float(np.mean(rollout.grip)) if len(rollout.grip) else float("nan"))
        if policy is not None:
            tilt_errors.append(rollout.policy_error)
        if isinstance(policy, LoggedVision):
            sight = float(np.mean(policy.errors)) if policy.errors else float("nan")
            sight_errors.append(sight)
        if verbose:
            print(
                f"  {label or 'controller':<21} ep {i:2d}  "
                f"surv={str(rollout.survived):<5} {rollout.steps:4d}/{plan.steps:4d} steps"
                f"  peak |ball| {peaks[-1] * 100:5.1f} cm"
                + (f"  sight err {sight_errors[-1] * 1000:5.1f} mm" if sight_errors else ""),
                flush=True,
            )
    return {
        "survived": survived,
        "episodes": len(plans),
        "mean_peak_ball_m": float(np.mean(peaks)),
        "mean_tracking_m": float(np.mean(tracking)),
        "mean_grip_n": float(np.nanmean(grips)),
        "tilt_rmse_vs_classical_rad": float(np.mean(tilt_errors)) if tilt_errors else 0.0,
        "ball_sight_rmse_m": float(np.mean(sight_errors)) if sight_errors else None,
    }


def add_condition_args(parser: Any) -> None:
    """Add the sim-to-real flags, all defaulting to ``None`` for "not given"."""
    parser.add_argument(
        "--condition",
        default="clean",
        choices=sorted(CONDITIONS),
        help="named sim-to-real preset; the individual flags below override it",
    )
    parser.add_argument("--gain-noise", type=float, default=None)
    parser.add_argument("--read-noise", type=float, default=None)
    parser.add_argument("--torque-control", action="store_true", default=None)
    parser.add_argument("--torque-limit-scale", type=float, default=None)
    parser.add_argument("--latency-steps", type=int, default=None)
    parser.add_argument(
        "--backlash", type=float, default=None, help="transmission slack, radians"
    )
    parser.add_argument("--randomize-dynamics", action="store_true", default=None)
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        default=None,
        help="jitter the tray's starting pose; composes with any preset",
    )


def condition_from_args(args: Any) -> Condition:
    """Build a condition from a named preset plus explicit command-line overrides.

    An override applies only when its flag was actually given, so
    ``--condition backlash-0.5`` keeps its backlash while adding
    ``--latency-steps 2`` on top of it still means what it says.
    """
    condition = CONDITIONS[getattr(args, "condition", "clean")]
    overrides = {
        field: value
        for field, value in (
            ("gain_noise", args.gain_noise),
            ("read_noise", args.read_noise),
            ("torque_control", args.torque_control),
            ("torque_limit_scale", args.torque_limit_scale),
            ("latency_steps", args.latency_steps),
            ("backlash", args.backlash),
            ("randomize_dynamics", args.randomize_dynamics),
            ("randomize_layout", args.randomize_layout),
        )
        if value is not None
    }
    if not overrides:
        return condition
    return replace(
        condition,
        name=condition.name if condition.name != "clean" else "custom",
        **overrides,
    )


class Stopwatch:
    """Elapsed seconds since construction, for reporting what a sweep cost."""

    def __init__(self) -> None:
        """Start counting."""
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Seconds since this stopwatch was made."""
        return time.perf_counter() - self.started
