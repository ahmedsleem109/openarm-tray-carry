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

"""Tests for carry plans, ball disturbances and the shared rollout driver."""

from __future__ import annotations

import numpy as np
import pytest
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import (
    BimanualTrayCarry,
    LayoutRanges,
    tilt_quat,
    tray_quat,
)
from openarm_gym.control.tray_task import (
    CarryPlan,
    Waypoint,
    flat_proprio,
    random_carry,
    run_plan,
    straight_carry,
)



def _carry(**kwargs) -> BimanualTrayCarry:
    return BimanualTrayCarry(scene_path(), **kwargs)


# ------------------------------------------------------------------- the plans


def test_random_carry_is_reproducible_from_its_seed() -> None:
    """Two generators on the same seed must produce the same manoeuvre."""
    first = random_carry(np.random.default_rng(7))
    second = random_carry(np.random.default_rng(7))
    assert len(first.waypoints) == len(second.waypoints)
    for a, b in zip(first.waypoints, second.waypoints):
        assert np.array_equal(a.offset, b.offset)
        assert a.steps == b.steps
    assert np.array_equal(first.ball_offset, second.ball_offset)
    assert first.disturbances.keys() == second.disturbances.keys()


def test_random_carry_travels_less_along_the_trays_short_axis() -> None:
    """x is the 7.5 cm half-axis and y the 15 cm one, so x must be limited more.

    A plus/minus 4 cm x leg was enough to lose the ball with no disturbance at
    all, which is what this bound exists to prevent.
    """
    rng = np.random.default_rng(0)
    offsets = np.array(
        [w.offset for _ in range(40) for w in random_carry(rng).waypoints]
    )
    assert np.abs(offsets[:, 0]).max() <= 0.025 + 1e-9
    assert np.abs(offsets[:, 1]).max() <= 0.11 + 1e-9


def test_disturbances_never_land_during_the_lift() -> None:
    """A shove during the lift measures the grasp, not the balancer."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        plan = random_carry(rng)
        lift_steps = plan.waypoints[0].steps
        for step in plan.disturbances:
            assert step > lift_steps, f"shove at {step} lands inside the {lift_steps}-step lift"


def test_every_requested_disturbance_is_scheduled() -> None:
    """Asking for six shoves must give six, not five.

    The schedule is a dict keyed by control step, so two draws landing on the
    same step silently collapse into one. Sampling with replacement lost one
    shove in 14 plans out of 200 -- invisible except as a dataset that is
    slightly gentler than the one that was asked for.
    """
    rng = np.random.default_rng(0)
    counts = [len(random_carry(rng, n_disturbances=6).disturbances) for _ in range(200)]
    assert min(counts) == 6, f"{sum(c < 6 for c in counts)} of 200 plans lost a shove"


def test_plan_step_count_matches_its_waypoints() -> None:
    """``steps`` is the contract the disturbance schedule is indexed against."""
    plan = CarryPlan(waypoints=(Waypoint(np.zeros(3), 30), Waypoint(np.zeros(3), 45)))
    assert plan.steps == 75


# ------------------------------------------------------------- the disturbance


def test_nudge_ball_changes_the_ball_velocity_in_the_tray_frame() -> None:
    """A kick is a velocity change in the tray frame, in m/s."""
    carry = _carry()
    carry.reset()
    carry.grasp()
    carry.place_ball()
    before = carry.ball_in_tray()[1][:2].copy()
    carry.nudge_ball(np.array([0.2, 0.0]))
    after = carry.ball_in_tray()[1][:2]
    assert after[0] - before[0] == pytest.approx(0.2, abs=2e-3)
    assert after[1] - before[1] == pytest.approx(0.0, abs=2e-3)


# ------------------------------------------------------------------ the driver


def test_run_plan_keeps_the_ball_and_records_aligned_traces() -> None:
    """The classical controller must fly the fixed carry, with matching traces.

    Every per-step array has to be the same length: they are zipped into training
    samples, and a silent off-by-one would pair each image with the next step's
    label.
    """
    carry = _carry()
    rollout = run_plan(carry, straight_carry())
    assert rollout.survived
    assert rollout.lost_at is None
    lengths = {
        len(rollout.ball_xy),
        len(rollout.ball_vxy),
        len(rollout.setpoints),
        len(rollout.labels),
        len(rollout.positions),
        len(rollout.tracking),
        len(rollout.grip),
        len(rollout.proprio),
    }
    assert lengths == {rollout.steps}
    # With no policy the commanded setpoint *is* the classical label.
    assert np.array_equal(rollout.setpoints, rollout.labels)
    assert rollout.policy_error == pytest.approx(0.0)
    assert rollout.mean_tracking < 0.01


def test_run_plan_records_pixels_only_when_asked() -> None:
    """Rendering dominates a control step, so it must not happen unbidden.

    Without this the evaluation of a state-only controller would silently pay for
    two camera renders per step.
    """
    carry = _carry(camera_names=("topcam",), image_size=(32, 40))
    renders = 0
    original = carry.cameras.render

    def counting(*args, **kwargs):
        nonlocal renders
        renders += 1
        return original(*args, **kwargs)

    carry.cameras.render = counting
    short = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 5),))
    run_plan(carry, short)
    assert renders == 0, f"{renders} renders with no consumer for the pixels"

    run_plan(carry, short, record_pixels=True)
    assert renders == 5
    assert carry.cameras.camera_names == ("topcam",)
    carry.close()


def test_recorded_ball_target_is_the_state_that_was_observed() -> None:
    """The ball target must describe the instant the image was taken.

    Reading the ball *after* the step instead paired every frame with the next
    step's ball position -- a 20 ms lead that asks the estimator to predict the
    future and biases both the position and the velocity target. The lag-1
    comparison is kept in the assertion message because it is what made the bug
    unmistakable: it matched to 1e-11 while lag 0 was off by 2 mm.
    """
    carry = _carry(camera_names=("topcam",), image_size=(32, 40))
    observed: list[np.ndarray] = []

    def spy(_obs: dict) -> np.ndarray:
        observed.append(carry.ball_in_tray()[0][:2].copy())
        return np.zeros(2)

    rollout = run_plan(
        carry, CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 12),)),
        tilt_policy=spy,
    )
    seen = np.asarray(observed)
    at_step = float(np.abs(seen - rollout.ball_xy).max())
    one_late = float(np.abs(seen[1:] - rollout.ball_xy[:-1]).max())
    assert at_step < 1e-9, (
        f"target differs from the observed state by {at_step * 1000:.2f} mm; "
        f"against the *next* step it differs by {one_late * 1000:.2f} mm, so the "
        "target is being read after the step instead of before it"
    )
    carry.close()


def test_a_driving_policy_without_cameras_is_refused_clearly() -> None:
    """Better a named error than a bare KeyError from inside the policy."""
    carry = _carry()
    with pytest.raises(ValueError, match="camera"):
        run_plan(
            carry,
            CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 2),)),
            tilt_policy=lambda obs: obs["pixels"]["topcam"][0, 0, :2],
        )


def test_a_policy_built_for_another_control_rate_is_refused() -> None:
    """A rate mismatch scales a differentiating policy's damping, silently.

    Nothing else about the rollout would look wrong, which is why it is worth
    an explicit check rather than a comment.
    """
    class Differentiating:
        control_hz = 50.0

        def __call__(self, _obs: dict) -> np.ndarray:
            return np.zeros(2)

    carry = _carry(camera_names=("topcam",), image_size=(32, 40), control_hz=25.0)
    with pytest.raises(ValueError, match="50 Hz"):
        run_plan(
            carry,
            CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 2),)),
            tilt_policy=Differentiating(),
        )
    carry.close()


def test_action_noise_perturbs_what_is_executed_not_what_is_labelled() -> None:
    """The label has to stay the clean expert setpoint, or the policy learns noise."""
    carry = _carry()
    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 10),))
    rollout = run_plan(
        carry, plan, action_noise=0.05, action_rng=np.random.default_rng(0)
    )
    assert not np.allclose(rollout.setpoints, rollout.labels)
    quiet = run_plan(carry, plan)
    assert np.array_equal(quiet.setpoints, quiet.labels)


def test_run_plan_uses_the_policy_it_is_given_and_still_records_the_label() -> None:
    """A driven rollout must keep the classical label for comparison.

    That is what makes a learned controller's error measurable *in closed loop*
    rather than only against a held-out dataset.
    """
    carry = _carry(camera_names=("topcam",), image_size=(32, 40))
    calls = 0

    def flat_policy(obs: dict) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert "pixels" in obs and "topcam" in obs["pixels"]
        return np.zeros(2)

    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 8),))
    rollout = run_plan(carry, plan, tilt_policy=flat_policy)
    assert calls == rollout.steps
    assert np.allclose(rollout.setpoints, 0.0)
    # The expert would not have commanded level throughout, so the two differ.
    assert rollout.policy_error > 0.0
    carry.close()


# ------------------------------------------------------------ the yaw goal


def test_tray_quat_reduces_to_the_tilt_when_yaw_is_zero() -> None:
    """The yaw goal must be a pure extension, not a change to the old behaviour."""
    assert np.allclose(tray_quat(0.05, -0.03, 0.0), tilt_quat(0.05, -0.03))


def test_the_tilt_means_the_same_thing_at_every_yaw() -> None:
    """The balance law's axes are the tray's own, so yaw must not rotate them.

    The ball's offset and velocity are measured in the tray frame, so a commanded
    ``(roll, pitch)`` has to act about the *yawed* axes. The invariant that
    captures it: undo the yaw, and the tray's surface normal is identical
    whatever the yaw was. Composing the other way round would rotate the
    balancer's own feedback by the yaw angle, turning a stable loop into a spiral
    -- and that failure would only show up once a yawed carry was flown.
    """
    import mujoco

    def normal_in_the_yawed_frame(yaw: float) -> np.ndarray:
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, tray_quat(0.08, -0.05, yaw))
        undo = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0.0],
                [np.sin(-yaw), np.cos(-yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return undo @ mat.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    reference = normal_in_the_yawed_frame(0.0)
    for yaw in (np.deg2rad(15), np.deg2rad(90), -np.deg2rad(40)):
        assert np.allclose(normal_in_the_yawed_frame(yaw), reference, atol=1e-9), (
            f"the commanded tilt changes meaning at {np.rad2deg(yaw):.0f} degrees of yaw"
        )
    # And the yaw is really applied: the world-frame orientation does change.
    assert not np.allclose(tray_quat(0.08, -0.05, np.deg2rad(15)), tilt_quat(0.08, -0.05))


def test_yaw_is_off_unless_asked_for() -> None:
    """Randomized carries stay planar by default, so old numbers still apply."""
    rng = np.random.default_rng(0)
    assert all(w.yaw == 0.0 for _ in range(10) for w in random_carry(rng).waypoints)
    turned = random_carry(np.random.default_rng(0), yaw=np.deg2rad(8))
    assert any(w.yaw != 0.0 for w in turned.waypoints)


def test_a_yawed_carry_keeps_the_ball_and_actually_turns_the_tray() -> None:
    """Both arms have to rotate the tray together while the balancer holds the ball.

    A handle sits 0.15 m out, so an 8 degree yaw sweeps it 21 mm along x -- this is
    a coordination test as much as a balance one.
    """
    import mujoco

    carry = BimanualTrayCarry(scene_path())
    goal = np.deg2rad(8.0)
    plan = CarryPlan(
        waypoints=(
            Waypoint(np.array([0.0, 0.0, 0.10]), 120),
            Waypoint(np.array([0.0, 0.05, 0.10]), 220, yaw=goal),
        ),
        ball_offset=np.array([0.01, 0.02]),
    )
    rollout = run_plan(carry, plan)
    assert rollout.survived, "the yawed carry lost the ball"
    assert rollout.yaws[-1] == pytest.approx(goal, abs=1e-6)

    # Read the achieved yaw off the tray's own orientation.
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, carry.tray_pose[1])
    achieved = float(np.arctan2(mat.reshape(3, 3)[1, 0], mat.reshape(3, 3)[0, 0]))
    assert achieved > 0.5 * goal, (
        f"commanded {np.rad2deg(goal):.1f} deg of yaw, achieved "
        f"{np.rad2deg(achieved):.1f} deg"
    )
    carry.close()


def test_flat_proprio_has_a_fixed_width_and_no_ball_state() -> None:
    """Proprioception is 18 joint angles plus the tray's 7-value pose.

    The ball is absent by design: recovering it from pixels is the task.
    """
    carry = _carry()
    carry.reset()
    obs = carry.observe(pixels=False)
    assert "pixels" not in obs
    flat = flat_proprio(obs)
    assert flat.shape == (25,)
    assert flat.dtype == np.float32


def test_on_reset_runs_after_the_reset_and_before_the_grasp() -> None:
    """The hook exists so per-episode dynamics randomization has somewhere to go.

    Its window is the whole point: the draw has to be applied to a settled model
    *before* the jaws close, or the grasp transform is captured under one set of
    masses and frictions and the episode is then flown under another.
    """
    carry = _carry()
    seen = []

    def on_reset(c: BimanualTrayCarry) -> None:
        # After reset(), the jaws are cracked open and no grasp has been
        # captured yet; after grasp() both of those are false.
        seen.append(dict(c._grasp_rel))

    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 5),))
    rollout = run_plan(carry, plan, on_reset=on_reset)
    assert len(seen) == 1
    assert seen[0] == {}
    assert set(carry._grasp_rel) == {"left", "right"}
    assert rollout.steps == 5


def test_run_plan_without_a_hook_is_unchanged() -> None:
    """The hook defaults to off, so every earlier measurement still reproduces."""
    carry = _carry()
    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.02]), 8),))
    first = run_plan(carry, plan)
    second = run_plan(carry, plan, on_reset=None)
    assert np.allclose(first.ball_xy, second.ball_xy)


def test_plan_yaw_is_relative_to_the_heading_the_tray_was_grasped_at() -> None:
    """A plan with no yaw goal must not untwist a tray that started turned.

    Positions in a plan are offsets from the grasp-time pose; the yaw goal has to
    be one too. Commanding an absolute zero instead orders the arms to unturn the
    tray on the first control step, which is a jerk straight through the grasp --
    measured at a layout 1.6 degrees off, it took tray tracking from 4.5 mm to
    10.2 mm and threw the ball well before the episode's disturbance arrived.
    """
    carry = _carry()
    turned = np.radians(3.0)
    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.05]), 60),))
    run_plan(
        carry,
        plan,
        on_reset=lambda c: c.randomize_layout(
            np.random.default_rng(0), LayoutRanges(x=(0, 0), y=(0, 0), yaw=(turned, turned))
        ),
    )
    assert carry.tray_yaw() == pytest.approx(turned, abs=np.radians(0.5))


def test_handle_source_replaces_the_privileged_handle_positions() -> None:
    """The grasp must use what it is told, not what it can read out of the scene.

    Asserted by lying to it: handles reported 6 cm off in y put the jaws
    somewhere the posts are not, and the grasp fails. A grasp that succeeded here
    would mean the argument was being ignored.
    """
    carry = _carry()
    plan = CarryPlan(waypoints=(Waypoint(np.array([0.0, 0.0, 0.05]), 60),))

    def lying(c: BimanualTrayCarry) -> dict:
        offset = np.array([0.0, 0.06, 0.0])
        return {
            "left": c.handle_pos("left") + offset,
            "right": c.handle_pos("right") + offset,
        }

    honest = run_plan(carry, plan)
    misled = run_plan(carry, plan, handle_source=lying)
    assert honest.mean_tracking < 0.01
    assert misled.mean_tracking > honest.mean_tracking


def test_stop_on_loss_false_flies_the_whole_plan() -> None:
    """Both halves of a comparison have to come back the same length.

    ``stop_on_loss=False`` used to break out of the current waypoint anyway, so
    a run that lost the ball still ended early and could not be laid beside one
    that did not.
    """
    # A shove far past the 0.30 m/s the balancer recovers, with the tray held
    # level: the ball goes over the short edge and stays gone.
    plan = CarryPlan(
        waypoints=(Waypoint(np.array([0.0, 0.0, 0.06]), 120),),
        ball_offset=np.array([0.04, 0.0]),
        disturbances={20: np.array([0.9, 0.0])},
    )

    def level(_obs: dict) -> np.ndarray:
        return np.zeros(2)

    carry = _carry(camera_names=("topcam",), image_size=(32, 40))
    rollout = run_plan(carry, plan, tilt_policy=level, stop_on_loss=False)
    assert rollout.lost_at is not None, "this plan is supposed to lose the ball"
    assert rollout.steps == plan.steps
    assert len(rollout.ball_xy) == plan.steps
