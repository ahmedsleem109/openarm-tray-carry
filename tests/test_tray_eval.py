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

"""Tests for the named sim-to-real conditions and the evaluation loop.

The risk this file guards against is a quiet one: an evaluation that reports a
condition it did not actually apply, or that applies it to one controller and
not the other. Either produces a plausible table that means nothing.
"""

from __future__ import annotations

import argparse

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.tray_eval import (
    CONDITIONS,
    Condition,
    LoggedVision,
    add_condition_args,
    build_carry,
    condition_from_args,
    episode_rngs,
    evaluate_controller,
)
from openarm_gym.control.tray_task import CarryPlan, Waypoint



def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_condition_args(parser)
    return parser.parse_args(argv)


# ------------------------------------------------------------- the conditions


def test_the_default_condition_turns_nothing_on() -> None:
    """``clean`` has to reproduce every number measured before the layer existed."""
    clean = CONDITIONS["clean"]
    assert clean == Condition("clean")
    assert not clean.noisy_pixels
    assert not clean.torque_control
    assert clean.latency_steps == 0 and clean.backlash == 0.0
    assert not clean.randomize_dynamics


def test_the_all_condition_turns_on_every_mechanism() -> None:
    """It is the row that matters: a real arm hands you its failure modes at once."""
    every = CONDITIONS["all"]
    assert every.noisy_pixels
    assert every.torque_control and every.torque_limit_scale < 1.0
    assert every.latency_steps > 0 and every.backlash > 0.0
    assert every.randomize_dynamics


def test_a_summary_names_only_what_is_enabled() -> None:
    """The printed line is the record of what was run; it must not overstate."""
    assert "baseline" in CONDITIONS["clean"].summary()
    assert CONDITIONS["latency-1"].summary() == "latency 1 step(s)"
    assert "backlash" in CONDITIONS["backlash-0.5"].summary()
    assert "torque" not in CONDITIONS["backlash-0.5"].summary()


def test_an_unset_flag_does_not_override_its_preset() -> None:
    """``None`` means "not given" -- a zero default would silently erase a preset."""
    condition = condition_from_args(_parse(["--condition", "backlash-0.5"]))
    assert condition == CONDITIONS["backlash-0.5"]


def test_an_explicit_flag_overrides_the_preset_it_is_added_to() -> None:
    """Presets are a starting point, not a straitjacket."""
    condition = condition_from_args(
        _parse(["--condition", "backlash-0.5", "--latency-steps", "2"])
    )
    assert condition.backlash == CONDITIONS["backlash-0.5"].backlash
    assert condition.latency_steps == 2
    assert condition.name == "backlash-0.5"


def test_flags_on_their_own_build_a_condition_named_custom() -> None:
    """A hand-built condition must not report itself as ``clean`` in the results."""
    condition = condition_from_args(_parse(["--torque-control", "--read-noise", "2.0"]))
    assert condition.torque_control and condition.read_noise == 2.0
    assert condition.name == "custom"


# --------------------------------------------------------------- the plumbing


def test_a_condition_reaches_the_controller_it_builds() -> None:
    """The knobs that are constructor arguments have to actually arrive there."""
    condition = CONDITIONS["all"]
    carry = build_carry(scene_path(), condition)
    try:
        assert carry.torque is not None
        assert carry.cameras.noisy
        for side in ("left", "right"):
            assert carry.actuators[side].latency_steps == condition.latency_steps
            assert carry.actuators[side].backlash == condition.backlash
    finally:
        carry.close()


def test_a_clean_condition_leaves_the_controller_at_its_defaults() -> None:
    """Otherwise "clean" would quietly be a different physics model."""
    carry = build_carry(scene_path(), CONDITIONS["clean"])
    try:
        assert carry.torque is None
        assert not carry.cameras.noisy
        assert not carry.actuators["left"].active
    finally:
        carry.close()


def test_camera_and_dynamics_draws_come_from_separate_streams() -> None:
    """Sharing one generator would make enabling the sensor model move the tray.

    That is not hypothetical here -- it is the bug class this repository already
    fixed once in the visual randomizer.
    """
    condition = Condition("both", read_noise=2.0, randomize_dynamics=True)
    rng, on_reset = episode_rngs(condition, 5, 0)
    assert rng is not None and on_reset is not None

    plain = Condition("dyn-only", randomize_dynamics=True)
    _, dyn_only = episode_rngs(plain, 5, 0)

    # Same episode seed, one condition with camera noise and one without: the
    # dynamics draw has to come out identical either way.
    carry = build_carry(scene_path(), condition)
    try:
        with_noise = _draw(on_reset, carry)
        without_noise = _draw(dyn_only, carry)
    finally:
        carry.close()
    assert set(with_noise) == {"mass", "friction", "gain", "damping"}
    for key, scales in with_noise.items():
        assert np.allclose(scales, without_noise[key]), key


def _draw(hook, carry) -> dict:
    """Apply one ``on_reset`` hook and return the scales it produced."""
    captured: dict = {}
    original = carry.randomize_dynamics

    def spy(rng, *args, **kwargs):
        captured.update(original(rng, *args, **kwargs))
        return captured

    carry.randomize_dynamics = spy
    try:
        hook(carry)
    finally:
        carry.randomize_dynamics = original
    return captured


def test_a_condition_without_dynamics_asks_for_no_hook() -> None:
    """A hook that fired anyway would randomize a condition that says it does not."""
    rng, on_reset = episode_rngs(CONDITIONS["clean"], 1, 0)
    assert rng is None and on_reset is None


def test_the_logged_wrapper_forwards_its_control_rate() -> None:
    """``run_plan`` checks the rate; a wrapper that hides it makes the check vacuous.

    The bug it guards against has no symptom other than a damping term scaled by
    the ratio of the two rates.
    """

    class Fake:
        control_hz = 33.0

        def reset(self) -> None:
            pass

    wrapped = LoggedVision(Fake(), None)
    assert wrapped.control_hz == 33.0


def test_evaluate_controller_summarises_every_plan_it_was_given() -> None:
    """The classical row is ``policy=None``; it must still report a full summary."""
    carry = build_carry(scene_path(), CONDITIONS["clean"])
    plans = [CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 6),))] * 2
    try:
        row = evaluate_controller(
            carry, plans, None, condition=CONDITIONS["clean"], seed=0, verbose=False
        )
    finally:
        carry.close()
    assert row["episodes"] == 2
    assert 0 <= row["survived"] <= 2
    assert row["ball_sight_rmse_m"] is None
    assert row["mean_grip_n"] > 0.0
