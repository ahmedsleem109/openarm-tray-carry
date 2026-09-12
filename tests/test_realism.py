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

"""Tests for the sim-to-real layer: torque control, actuator faults, dynamics DR."""

from __future__ import annotations

import numpy as np
import pytest
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry
from openarm_gym.realism import ActuatorModel, DynamicsRanges



def _carry(**kwargs) -> BimanualTrayCarry:
    return BimanualTrayCarry(scene_path(), **kwargs)


# ----------------------------------------------------------- the actuator model


def test_actuator_model_defaults_to_the_identity() -> None:
    """Off by default, so nothing measured before it existed moves."""
    model = ActuatorModel()
    assert not model.active
    target = np.array([0.1, -0.2, 0.3])
    assert np.array_equal(model.apply(target), target)


def test_latency_delays_the_command_by_whole_control_steps() -> None:
    """A one-step delay line returns the previous command, not zeros.

    Priming with the first command matters: a queue of zeros would command the
    arm to fold up on the first step of every episode.
    """
    model = ActuatorModel(latency_steps=1)
    first = model.apply(np.array([1.0, 2.0]))
    assert np.array_equal(first, [1.0, 2.0]), "the delay line must prime, not zero"
    assert np.array_equal(model.apply(np.array([3.0, 4.0])), [1.0, 2.0])
    assert np.array_equal(model.apply(np.array([5.0, 6.0])), [3.0, 4.0])


def test_backlash_holds_until_the_slack_is_taken_up() -> None:
    """Motion inside the deadband is not transmitted; beyond it, it lags by one."""
    model = ActuatorModel(backlash=0.1)
    assert np.allclose(model.apply(np.array([0.0])), [0.0])
    # Inside the deadband: the joint does not move at all.
    assert np.allclose(model.apply(np.array([0.05])), [0.0])
    # Past it: moves, but stays a deadband behind the command.
    assert np.allclose(model.apply(np.array([0.5])), [0.4])
    # Reversing must take up the slack again before anything moves back.
    assert np.allclose(model.apply(np.array([0.35])), [0.4])
    assert np.allclose(model.apply(np.array([0.2])), [0.3])


def test_backlash_is_hysteresis_not_a_constant_offset() -> None:
    """The sign of the lag follows the direction of travel.

    That is why backlash costs phase rather than accuracy, and why it is worth
    modelling separately from noise.
    """
    up = ActuatorModel(backlash=0.1)
    up.apply(np.array([0.0]))
    rising = float(up.apply(np.array([1.0]))[0])
    down = ActuatorModel(backlash=0.1)
    down.apply(np.array([0.0]))
    falling = float(down.apply(np.array([-1.0]))[0])
    assert rising == pytest.approx(0.9)
    assert falling == pytest.approx(-0.9)


def test_actuator_reset_clears_the_delay_line() -> None:
    """Episodes must not inherit the previous episode's queued commands."""
    model = ActuatorModel(latency_steps=2)
    model.apply(np.array([1.0]))
    model.apply(np.array([2.0]))
    model.reset()
    assert np.array_equal(model.apply(np.array([9.0])), [9.0])


# ------------------------------------------------------------ the torque servo


def test_torque_control_neutralises_the_position_actuators() -> None:
    """Engaging must leave torque as the only input to those joints.

    A position servo left half-active alongside a commanded torque gives a
    stiffness that is neither the simulated one nor the real one.
    """
    carry = _carry(torque_control=True)
    assert carry.torque is not None
    for servo in carry.torque.values():
        assert servo.engaged
        assert np.allclose(carry.model.actuator_gainprm[servo.actuators, 0], 0.0)
        assert np.allclose(carry.model.actuator_biasprm[servo.actuators, 1], 0.0)
        assert np.allclose(carry.model.actuator_biasprm[servo.actuators, 2], 0.0)


def test_position_control_leaves_the_scene_actuators_alone() -> None:
    """Without the flag the scene's own servos must be untouched."""
    carry = _carry()
    assert carry.torque is None
    assert carry.model.actuator_gainprm[0, 0] > 0.0


def test_torque_servo_respects_its_limit() -> None:
    """Commanded torque never exceeds the modelled motor's rating."""
    carry = _carry(torque_control=True, torque_limit_scale=0.5)
    carry.reset()
    servo = carry.torque["left"]
    # A wildly wrong target would demand unbounded torque from the PD term.
    torque = servo.torque(carry.data, carry.data.qpos[servo.qpos] + 3.0)
    assert np.all(np.abs(torque) <= servo.torque_limit + 1e-9)
    assert servo.saturation(torque) == pytest.approx(1.0)
    # Half of the model's own force range, which is per DM motor part number.
    expected = np.abs(carry.model.actuator_forcerange[servo.actuators]).max(axis=1) * 0.5
    assert np.allclose(servo.torque_limit, expected)


def test_torque_controlled_grasp_still_carries_the_tray() -> None:
    """The whole point: the carry survives being driven by torque, not position.

    Position servos commanded into a rigid grasped object are the least
    transferable thing here, so this is the test that says the sim-to-real path
    is real rather than aspirational.
    """
    carry = _carry(torque_control=True)
    carry.reset()
    report = carry.grasp()
    total = sum(report[s]["force"] for s in ("left", "right"))
    assert total > 50.0, f"torque-controlled grasp only carries {total:.1f} N"

    start = carry.tray_pose[0].copy()
    goal = start + np.array([0.0, 0.0, 0.10])
    for i in range(150):
        alpha = min(1.0, (i + 1) / 110)
        carry.command_tray(start * (1 - alpha) + goal * alpha, carry.balance_tilt())
    assert carry.tray_pose[0][2] - start[2] > 0.05, "torque control dropped the tray"
    assert carry.ball_on_tray()


# -------------------------------------------------------- dynamics randomization


def test_dynamics_randomization_changes_the_model_and_restores_it() -> None:
    """Sampling must move mass and friction, and ``restore`` must undo it."""
    carry = _carry()
    nominal_mass = carry.model.body_mass.copy()
    nominal_friction = carry.model.geom_friction.copy()
    applied = carry.randomize_dynamics(np.random.default_rng(0))

    assert set(applied) == {"mass", "friction", "gain", "damping"}
    assert not np.allclose(carry.model.body_mass, nominal_mass)
    assert not np.allclose(carry.model.geom_friction, nominal_friction)

    carry._dynamics.restore()
    assert np.allclose(carry.model.body_mass, nominal_mass)
    assert np.allclose(carry.model.geom_friction, nominal_friction)


def test_dynamics_randomization_always_scales_from_the_snapshot() -> None:
    """Repeated draws must not random-walk the model away from the scene.

    Scaling the live value instead of the authored one drifts slowly, which
    presents as a controller that mysteriously degrades over a long run.
    """
    carry = _carry()
    nominal = carry.model.body_mass[carry._tray]
    rng = np.random.default_rng(0)
    for _ in range(20):
        carry.randomize_dynamics(rng)
    ratio = carry.model.body_mass[carry._tray] / nominal
    low, high = DynamicsRanges().mass
    assert low <= ratio <= high, f"mass drifted to {ratio:.3f}x nominal"


def test_dynamics_randomization_scales_inertia_with_mass() -> None:
    """Mass and rotational inertia have to move together to stay consistent."""
    carry = _carry()
    before = (
        carry.model.body_mass[carry._tray],
        carry.model.body_inertia[carry._tray].copy(),
    )
    carry.randomize_dynamics(np.random.default_rng(3))
    after = (carry.model.body_mass[carry._tray], carry.model.body_inertia[carry._tray])
    assert np.allclose(after[1] / before[1], after[0] / before[0])


def test_gain_randomization_reaches_the_torque_law() -> None:
    """Under torque control the servo's own gains must be the ones randomized.

    ``engage`` zeroes the position gains, so a randomizer that only wrote to the
    model would be a silent no-op -- the most expensive kind of bug in a
    domain-randomization experiment, because it looks like DR that does nothing.
    """
    carry = _carry(torque_control=True)
    nominal = {side: servo.kp.copy() for side, servo in carry.torque.items()}
    applied = carry.randomize_dynamics(np.random.default_rng(1))
    for side, servo in carry.torque.items():
        assert not np.allclose(servo.kp, nominal[side])
        assert np.all(servo.kp > 0.0)
    assert len(applied["gain"]) == carry.model.nu
