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

"""The three things standing between this controller and real hardware.

Each class here exists because of a specific way simulation flatters a
controller, and each is **off by default** so that no existing measurement moves
when this module is imported.

:class:`JointTorqueServo`
    The scenes drive the arms with MuJoCo ``position`` actuators. A position
    servo commanded into a rigid grasped object is the single least transferable
    thing in this project: the solver will find whatever force satisfies the
    position target, so a controller can quietly rely on infinite stiffness that
    no DM motor has. This replaces them with explicit joint torque, computed from
    the model's own gains so behaviour starts out identical, but now passing
    through a torque limit that can be lowered to the real motor's value.

:class:`ActuatorModel`
    Commands on real hardware arrive late, and real gearboxes have slack. Both
    are pure phase loss, which is exactly what destabilises a derivative term --
    so a balancer tuned without them can be tuned into a region that does not
    exist on hardware.

:class:`DynamicsRandomizer`
    Mass, friction and actuator gain are all guesses in the scene file. Visual
    domain randomization already exists (and its RNG bug is fixed); this is the
    dynamics half, which is the half that matters for a *contact* task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


class JointTorqueServo:
    """Drive joints with explicit torque instead of MuJoCo position actuators.

    Construction only *reads* the model. Call :meth:`engage` to actually
    neutralise the position actuators; until then the object is inert, so it can
    be built unconditionally and enabled by a flag.

    The default gains are the model's own ``kp``/``kv``, which makes the torque
    law reproduce what the position actuator was already computing -- the point
    is not to change the behaviour but to change the *interface*, so that

    * the torque limit is applied by us, explicitly, and can be lowered to a real
      motor's continuous rating rather than its peak;
    * gravity compensation is part of the commanded torque, as it is in the real
      driver, rather than a separate ``qfrc_applied`` term;
    * latency and backlash have somewhere to live.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        actuators: np.ndarray,
        dofs: np.ndarray,
        qpos: np.ndarray,
        *,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        torque_limit_scale: float = 1.0,
    ) -> None:
        """Read the model's gains and force limits for the given actuators.

        Args:
            model: the compiled model, mutated by :meth:`engage`.
            actuators: actuator ids to take over, in order.
            dofs: the DOF index each actuator drives.
            qpos: the qpos index each actuator drives.
            kp: position gains; defaults to the model's ``gainprm[:, 0]``.
            kd: velocity gains; defaults to the model's ``-biasprm[:, 2]``.
            torque_limit_scale: fraction of each actuator's ``forcerange`` the
                servo may command. Below 1.0 this is a real constraint -- the DM
                motors' continuous rating is well under their peak.

        """
        self.model = model
        self.actuators = np.asarray(actuators, dtype=np.intp)
        self.dofs = np.asarray(dofs, dtype=np.intp)
        self.qpos = np.asarray(qpos, dtype=np.intp)
        self.kp = (
            model.actuator_gainprm[self.actuators, 0].copy() if kp is None
            else np.asarray(kp, dtype=np.float64)
        )
        # A MuJoCo position actuator stores its damping as biasprm[2] = -kv.
        self.kd = (
            -model.actuator_biasprm[self.actuators, 2].copy() if kd is None
            else np.asarray(kd, dtype=np.float64)
        )
        self.torque_limit = (
            np.abs(model.actuator_forcerange[self.actuators]).max(axis=1)
            * float(torque_limit_scale)
        )
        self._engaged = False

    @property
    def engaged(self) -> bool:
        """Whether the position actuators have been neutralised."""
        return self._engaged

    def engage(self) -> None:
        """Zero the position actuators' gains, so torque is the only input.

        Irreversible for this model instance by design: leaving a position servo
        half-active alongside a torque command gives a stiffness that is neither
        the simulated one nor the real one, and the resulting numbers would be
        uninterpretable.
        """
        if self._engaged:
            return
        self.model.actuator_gainprm[self.actuators, 0] = 0.0
        self.model.actuator_biasprm[self.actuators, 1] = 0.0
        self.model.actuator_biasprm[self.actuators, 2] = 0.0
        self._engaged = True

    def torque(self, data: mujoco.MjData, q_desired: np.ndarray) -> np.ndarray:
        """Return the clipped joint torque that tracks ``q_desired``.

        Computed-torque form: a PD term on joint error plus ``qfrc_bias``, which
        carries gravity *and* Coriolis. The wrists run at kp=30 against links
        that sag ~9 mm without this term, which tilts a held tray by 3.4 degrees.
        """
        error = np.asarray(q_desired, dtype=np.float64) - data.qpos[self.qpos]
        command = self.kp * error - self.kd * data.qvel[self.dofs]
        return np.clip(
            command + data.qfrc_bias[self.dofs], -self.torque_limit, self.torque_limit
        )

    def saturation(self, torque: np.ndarray) -> float:
        """Fraction of joints sitting at their torque limit.

        Worth watching: a controller that only works while saturated is relying
        on a stiffness the hardware does not have.
        """
        return float(np.mean(np.abs(np.abs(torque) - self.torque_limit) < 1e-9))


@dataclass
class ActuatorModel:
    """Command latency and transmission backlash on a joint-target stream.

    Both default to zero, which is the identity.

    ``latency_steps`` is in **control steps**: the OpenArm driver talks CAN at
    1 kHz while this controller runs at 50 Hz, so one control step of delay is
    already a pessimistic 20 ms.

    ``backlash`` is radians of slack in the transmission: the joint stops
    following until the command has moved further than the deadband, after which
    it follows offset by it. This is why it is not just noise -- it is a
    *hysteresis*, so it costs phase, and phase is what a derivative term spends.
    """

    latency_steps: int = 0
    backlash: float = 0.0
    _queue: list[np.ndarray] = field(default_factory=list, repr=False)
    _transmitted: np.ndarray | None = field(default=None, repr=False)

    @property
    def active(self) -> bool:
        """Whether this model changes the command at all."""
        return self.latency_steps > 0 or self.backlash > 0.0

    def reset(self, target: np.ndarray | None = None) -> None:
        """Clear the delay line, optionally priming it with a held pose."""
        self._queue = []
        self._transmitted = None if target is None else np.array(target, dtype=np.float64)

    def apply(self, target: np.ndarray) -> np.ndarray:
        """Return the target as the joint actually receives it."""
        target = np.asarray(target, dtype=np.float64)
        if self.latency_steps > 0:
            self._queue.append(target.copy())
            # Prime the line with the first command rather than with zeros: a
            # queue of zeros would command the arm to fold up on the first step.
            while len(self._queue) <= self.latency_steps:
                self._queue.insert(0, self._queue[0].copy())
            target = self._queue.pop(0)
        if self.backlash > 0.0:
            if self._transmitted is None:
                self._transmitted = target.copy()
            else:
                slack = target - self._transmitted
                moved = np.abs(slack) > self.backlash
                self._transmitted[moved] = (
                    target[moved] - np.sign(slack[moved]) * self.backlash
                )
            target = self._transmitted.copy()
        return target


@dataclass(frozen=True)
class DynamicsRanges:
    """Multiplicative ranges for one dynamics randomization draw."""

    #: Body mass, with rotational inertia scaled to match so the body stays
    #: physically consistent.
    mass: tuple[float, float] = (0.8, 1.25)
    #: Sliding, torsional and rolling friction, scaled together.
    friction: tuple[float, float] = (0.7, 1.4)
    #: Actuator position gain, standing in for drive-train and tuning variation.
    gain: tuple[float, float] = (0.9, 1.1)
    #: Joint damping.
    damping: tuple[float, float] = (0.8, 1.3)


class DynamicsRandomizer:
    """Resample masses, frictions, gains and damping from a nominal snapshot.

    Always scales from the **snapshot**, never from the live value, so repeated
    sampling cannot random-walk the model away from the scene as authored -- that
    is a slow drift that looks like an unstable controller.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        bodies: tuple[int, ...] = (),
        geoms: tuple[int, ...] = (),
        actuators: np.ndarray | None = None,
        dofs: np.ndarray | None = None,
        ranges: DynamicsRanges = DynamicsRanges(),
    ) -> None:
        """Snapshot everything this randomizer is allowed to touch."""
        self.model = model
        self.bodies = np.asarray(bodies, dtype=np.intp)
        self.geoms = np.asarray(geoms, dtype=np.intp)
        self.actuators = (
            np.arange(model.nu, dtype=np.intp) if actuators is None
            else np.asarray(actuators, dtype=np.intp)
        )
        self.dofs = (
            np.arange(model.nv, dtype=np.intp) if dofs is None
            else np.asarray(dofs, dtype=np.intp)
        )
        self.ranges = ranges
        self._mass = model.body_mass[self.bodies].copy() if len(self.bodies) else None
        self._inertia = model.body_inertia[self.bodies].copy() if len(self.bodies) else None
        self._friction = model.geom_friction[self.geoms].copy() if len(self.geoms) else None
        self._gain = model.actuator_gainprm[self.actuators, 0].copy()
        self._bias = model.actuator_biasprm[self.actuators, 1].copy()
        self._damping = model.dof_damping[self.dofs].copy()

    def restore(self) -> None:
        """Put every randomized quantity back to the scene's authored value."""
        if self._mass is not None:
            self.model.body_mass[self.bodies] = self._mass
            self.model.body_inertia[self.bodies] = self._inertia
        if self._friction is not None:
            self.model.geom_friction[self.geoms] = self._friction
        self.model.actuator_gainprm[self.actuators, 0] = self._gain
        self.model.actuator_biasprm[self.actuators, 1] = self._bias
        self.model.dof_damping[self.dofs] = self._damping

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw and apply one set of scales, returning what was applied."""
        applied: dict[str, np.ndarray] = {}
        if self._mass is not None:
            scale = rng.uniform(*self.ranges.mass, size=len(self.bodies))
            self.model.body_mass[self.bodies] = self._mass * scale
            self.model.body_inertia[self.bodies] = self._inertia * scale[:, None]
            applied["mass"] = scale
        if self._friction is not None:
            scale = rng.uniform(*self.ranges.friction, size=len(self.geoms))
            self.model.geom_friction[self.geoms] = self._friction * scale[:, None]
            applied["friction"] = scale
        scale = rng.uniform(*self.ranges.gain, size=len(self.actuators))
        # A position actuator's bias term is -kp, so it has to track the gain or
        # the actuator stops being a position servo and becomes an offset one.
        self.model.actuator_gainprm[self.actuators, 0] = self._gain * scale
        self.model.actuator_biasprm[self.actuators, 1] = self._bias * scale
        applied["gain"] = scale
        scale = rng.uniform(*self.ranges.damping, size=len(self.dofs))
        self.model.dof_damping[self.dofs] = self._damping * scale
        applied["damping"] = scale
        return applied
