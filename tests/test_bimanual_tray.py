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

"""Tests for the two-arm tray grasp and the ball-balancing controller."""

from __future__ import annotations

import numpy as np
import pytest
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry



def _carry() -> BimanualTrayCarry:
    return BimanualTrayCarry(scene_path())


@pytest.fixture(scope="module")
def grasped() -> BimanualTrayCarry:
    """One grasped tray, shared by the tests that only read from it."""
    carry = _carry()
    carry.reset()
    carry.grasp()
    return carry


# ------------------------------------------------------------------- the scene


def test_scene_has_no_welds() -> None:
    """The tray must be held by friction, not by an equality constraint.

    Only the two finger mimic constraints may remain: a weld here would make
    the grasp a fiction.
    """
    carry = _carry()
    names = [
        __import__("mujoco").mj_id2name(
            carry.model, __import__("mujoco").mjtObj.mjOBJ_EQUALITY, e
        )
        for e in range(carry.model.neq)
    ]
    assert carry.model.neq == 2, f"expected only finger mimics, got {names}"
    assert all("mimic" in (n or "") for n in names), names


def test_arm_collision_geometry_is_enabled() -> None:
    """Collisions stay on: the grasp depends on finger/handle contact."""
    carry = _carry()
    collidable = sum(
        1
        for g in range(carry.model.ngeom)
        if carry.model.geom_contype[g] or carry.model.geom_conaffinity[g]
    )
    assert collidable > 30, f"only {collidable} collidable geoms -- collisions off?"


def test_reset_starts_clear_of_the_tray() -> None:
    """Reset must not leave the jaws inside the handle posts.

    The scene's home keyframe does exactly that, and the contact impulse throws
    the tray half a metre before anything can be grasped.
    """
    import mujoco

    carry = _carry()
    carry.reset()
    touching = [
        c
        for c in range(carry.data.ncon)
        if "handle"
        in (
            (mujoco.mj_id2name(carry.model, mujoco.mjtObj.mjOBJ_GEOM, carry.data.contact[c].geom1) or "")
            + (mujoco.mj_id2name(carry.model, mujoco.mjtObj.mjOBJ_GEOM, carry.data.contact[c].geom2) or "")
        )
    ]
    assert not touching, f"{len(touching)} arm/handle contacts at reset"


# ------------------------------------------------------------------- the grasp


def test_grasp_closes_on_both_handles(grasped: BimanualTrayCarry) -> None:
    """Both hands must end up in contact with their post, carrying real force."""
    report = grasped.grip_report()
    for side in ("left", "right"):
        assert report[side]["contacts"] > 0, f"{side} hand made no contact: {report}"
        assert report[side]["force"] > 5.0, f"{side} grip too weak: {report}"


def test_grasp_lifts_the_tray(grasped: BimanualTrayCarry) -> None:
    """The friction grasp must carry the tray's weight clear of the table."""
    carry = _carry()
    carry.reset()
    carry.grasp()
    start_z = carry.tray_pose[0][2]
    goal = carry.tray_pose[0] + np.array([0.0, 0.0, 0.10])
    for _ in range(150):
        carry.command_tray(goal, np.array([1.0, 0.0, 0.0, 0.0]))
    lifted = carry.tray_pose[0][2] - start_z
    assert lifted > 0.05, f"tray only rose {lifted * 1000:.1f} mm"


# -------------------------------------------------------------- the controller


def test_ball_frame_matches_a_seated_ball(grasped: BimanualTrayCarry) -> None:
    """A ball seated on the tray reads as centred, one radius above the face."""
    grasped.place_ball()
    rel, vel = grasped.ball_in_tray()
    assert np.allclose(rel[:2], 0.0, atol=1e-6), rel
    assert rel[2] == pytest.approx(0.025, abs=1e-6)
    # Velocity is relative to the tray, and the tray is still being actively
    # held, so it drifts slightly -- this is near zero, not exactly zero.
    assert np.allclose(vel, 0.0, atol=1e-3), vel


def test_balancing_keeps_the_ball_and_not_balancing_loses_it() -> None:
    """The controller earns its place: the same carry succeeds only with it.

    This is the baseline any learned policy has to beat, so it is asserted as a
    comparison rather than as an absolute success rate.
    """
    level = np.array([1.0, 0.0, 0.0, 0.0])
    outcome = {}
    for balancing in (True, False):
        carry = _carry()
        carry.reset()
        carry.grasp()
        carry.place_ball(np.array([0.02, 0.04]))
        start = carry.tray_pose[0].copy()
        lift = start + np.array([0.0, 0.0, 0.10])
        goal = lift + np.array([0.0, 0.10, 0.0])
        for target, steps in ((lift, 120), (goal, 200)):
            src = carry.tray_pose[0].copy()
            for i in range(steps):
                a = min(1.0, (i + 1) / max(1, steps - 40))
                carry.command_tray(
                    src * (1 - a) + target * a,
                    carry.balance_tilt() if balancing else level,
                )
        outcome[balancing] = carry.ball_on_tray()
    assert outcome[True], "the balancing controller dropped the ball"
    assert not outcome[False], (
        "the ball survived without balancing -- the test no longer distinguishes "
        "the controller, so make the manoeuvre harder"
    )


def test_balancing_recentres_an_offset_ball() -> None:
    """Seeded off-centre, the PD law must drive the ball back toward the middle."""
    carry = _carry()
    carry.reset()
    carry.grasp()
    carry.place_ball(np.array([0.02, 0.04]))
    before = np.linalg.norm(carry.ball_in_tray()[0][:2])
    # Hold station rather than commanding a step: a 10 cm jump in one control
    # step jerks the tray hard enough to throw the ball off, which would test
    # the trajectory, not the PD law.
    hold = carry.tray_pose[0].copy()
    for _ in range(120):
        carry.command_tray(hold, carry.balance_tilt())
    after = np.linalg.norm(carry.ball_in_tray()[0][:2])
    assert after < before / 2.0, f"ball offset {before:.4f} -> {after:.4f} m"


def test_balance_tilt_is_level_when_the_ball_is_gone(grasped: BimanualTrayCarry) -> None:
    """A lost ball must not saturate the tilt and tear the grasp off the posts."""
    import mujoco

    grasped.data.qpos[25:28] = np.array([1.0, 1.0, 0.1])
    grasped.data.qvel[24:30] = 0.0
    mujoco.mj_forward(grasped.model, grasped.data)
    assert not grasped.ball_on_tray()
    assert np.allclose(grasped.balance_tilt(), [1.0, 0.0, 0.0, 0.0])


def test_full_carry_needs_no_ball_placement_workaround() -> None:
    """reset -> grasp -> lift -> carry must work on the scene's own initial state.

    The ball starts seated on the tray in the keyframe. With the original
    cylindrical handles the approach tilted the tray and rolled the ball off
    before the jaws closed, so tests had to re-seat it; box handles give form
    closure, the tray slides instead of tilting, and the ball rides along.
    """
    carry = _carry()
    carry.reset()
    before = carry.tray_pose[0].copy()
    carry.grasp()
    assert carry.ball_on_tray(), "the approach lost the ball before grasping"

    start = carry.tray_pose[0].copy()
    lift = start + np.array([0.0, 0.0, 0.10])
    goal = lift + np.array([0.0, 0.10, 0.0])
    for target, steps in ((lift, 120), (goal, 200)):
        src = carry.tray_pose[0].copy()
        for i in range(steps):
            a = min(1.0, (i + 1) / max(1, steps - 40))
            carry.command_tray(src * (1 - a) + target * a, carry.balance_tilt())

    assert carry.ball_on_tray(), "ball lost during the carry"
    assert carry.tray_pose[0][2] - before[2] > 0.05, "tray never lifted"
    tracked = carry.tray_pose[0][1] - start[1]
    assert tracked > 0.07, f"tray only tracked {tracked * 100:.1f} cm of 10 cm"


def test_the_approach_does_not_bulldoze_the_tray() -> None:
    """Reaching in must not shove the tray across the table.

    The original approach drove in at the handle centre, which put the gripper's
    ``ee_base_link`` -- the palm, not the fingers -- against the tray's top face
    and pushed the whole tray 30 mm forward before the jaws reached the posts.
    Reaching in above that height and settling down cuts it to about 5 mm.
    """
    carry = _carry()
    carry.reset()
    before = carry.tray_pose[0].copy()
    carry.grasp()
    moved = float(np.linalg.norm((carry.tray_pose[0] - before)[:2]))
    assert moved < 0.012, f"the approach displaced the tray {moved * 1000:.1f} mm"


def test_the_old_flat_approach_is_what_displaced_the_tray() -> None:
    """Pin the cause, so the fix cannot be silently reverted.

    Asserted as a comparison rather than an absolute: what matters is that the
    reach height is the mechanism, not the specific millimetres.
    """
    flat = _carry()
    flat.reset()
    before = flat.tray_pose[0].copy()
    flat.grasp(approach_lift=0.0, close_lift=0.0)
    displaced = float(np.linalg.norm((flat.tray_pose[0] - before)[:2]))
    assert displaced > 0.020, (
        f"the flat approach only displaced the tray {displaced * 1000:.1f} mm, so "
        "this no longer demonstrates why grasp() reaches in high"
    )


def test_box_handles_give_a_firm_grasp() -> None:
    """The handles must present flat faces to the jaws, not a smooth cylinder.

    Clamping a cylinder leaves the tray free to rotate inside the grasp on
    friction alone, and the grip thinned to a single finger contact under load.
    Flat faces perpendicular to the closing axis hold it: measured nine
    worst-case contacts and roughly double the normal force.
    """
    import mujoco

    carry = _carry()
    for name in ("handle_left", "handle_right"):
        gid = mujoco.mj_name2id(carry.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert carry.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX, (
            f"{name} is not a box; a cylinder cannot resist rotation in the grasp"
        )
    carry.reset()
    report = carry.grasp()
    total = sum(report[s]["force"] for s in ("left", "right"))
    assert total > 50.0, f"grasp only carries {total:.1f} N total"


# ------------------------------------------------------------ layout jitter


def test_randomize_layout_moves_the_tray_and_keeps_the_ball_on_it() -> None:
    """The tray starts somewhere else, and the ball is re-seated on top of it.

    Re-seating is not cosmetic: the ball was resting on the tray's *old* pose, so
    leaving it there would have the grasp lift the tray out from under it.
    """
    carry = _carry()
    carry.reset()
    home = carry.tray_pose[0].copy()
    applied = carry.randomize_layout(np.random.default_rng(0))
    moved = carry.tray_pose[0]
    assert not np.allclose(moved[:2], home[:2])
    assert moved[2] == pytest.approx(home[2], abs=1e-9)
    assert carry.ball_on_tray()
    rel, _ = carry.ball_in_tray()
    assert np.linalg.norm(rel[:2]) < 1e-6
    assert set(applied) == {"x", "y", "yaw"}


def test_randomize_layout_draws_around_the_keyframe_not_the_live_pose() -> None:
    """Sampling around wherever the tray is would random-walk it out of reach.

    Two draws from the same seed have to land in the same place however many
    episodes have run in between.
    """
    carry = _carry()
    poses = []
    for _ in range(3):
        carry.reset()
        carry.randomize_layout(np.random.default_rng(11))
        poses.append(carry.tray_pose[0].copy())
    for pose in poses[1:]:
        assert np.allclose(pose, poses[0])


def test_randomize_layout_stays_inside_the_carry_envelope() -> None:
    """Bounded on the near side, where the carry was measured to fail.

    Pulling the tray closer than about 0.33 makes the arms fold up: tray tracking
    degrades from 3 mm to 7-16 mm and randomized carries start throwing the ball.
    Reaching out is free to 0.40, which is why the envelope leans outward.

    The yaw term is what makes this worth asserting rather than reading off the
    ranges: a handle sits 0.15 m out, so yaw moves it along x as well, and the
    two bounds interact.
    """
    carry = _carry()
    rng = np.random.default_rng(3)
    for _ in range(25):
        carry.reset()
        carry.randomize_layout(rng)
        for side in ("left", "right"):
            handle = carry.handle_pos(side)
            assert 0.33 < handle[0] < 0.40
            assert 0.12 < abs(handle[1]) < 0.18


def test_the_grasp_still_holds_from_a_jittered_layout() -> None:
    """The grasp reads the handles out of the scene, so it has to follow them.

    Measured over the default envelope: the tray is shoved 5.7-5.9 mm, against
    5.8 mm from the nominal layout, and the grip stays at about 42 N per hand.
    """
    carry = _carry()
    carry.reset()
    carry.randomize_layout(np.random.default_rng(4242))
    before = carry.tray_pose[0].copy()
    report = carry.grasp()
    assert np.linalg.norm(carry.tray_pose[0][:2] - before[:2]) < 0.010
    for side in ("left", "right"):
        assert report[side]["contacts"] >= 2
        assert report[side]["force"] > 35.0
