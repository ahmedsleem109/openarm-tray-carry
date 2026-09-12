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

"""Watch the tray carry: render it to an MP4, or drive it in the live viewer.

A controller nobody can watch is hard to trust, and the failure modes here are
visual -- the tray slipping in the grasp, the ball spiralling instead of
settling, a shove arriving mid-carry. Both paths run the *same*
:func:`~openarm_gym.control.tray_task.run_plan` the tests and the evaluation use,
so what you see is the manoeuvre that was measured.

``--viewer`` opens ``mujoco.viewer`` and steps the plan in real time, which needs
a display; the default offscreen path does not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry, tray_quat
from openarm_gym.control.tray_task import random_carry, run_plan, straight_carry
from openarm_gym.vision import CameraRig, write_mp4



def build_plan(args) -> object:
    """Return the plan named on the command line."""
    if args.plan == "straight":
        return straight_carry()
    return random_carry(np.random.default_rng(args.seed))


def run_viewer(carry: BimanualTrayCarry, plan, policy) -> None:
    """Step the plan in the interactive viewer, paced to wall-clock time.

    Deliberately not routed through :func:`run_plan`: that function owns the
    stepping loop, and the viewer needs to own it instead so it can sync and
    sleep. The control law is the same, which is what matters.
    """
    import mujoco.viewer

    carry.reset()
    carry.grasp()
    carry.place_ball(np.asarray(plan.ball_offset, dtype=np.float64))
    origin = carry.tray_pose[0].copy()
    period = 1.0 / carry.control_hz

    with mujoco.viewer.launch_passive(carry.model, carry.data) as viewer:
        step, yaw_from = 0, 0.0
        for waypoint in plan.waypoints:
            target = origin + waypoint.offset
            src = carry.tray_pose[0].copy()
            for i in range(waypoint.steps):
                if not viewer.is_running():
                    return
                tick = time.perf_counter()
                if step in plan.disturbances:
                    carry.nudge_ball(np.asarray(plan.disturbances[step], dtype=np.float64))
                    print(f"  step {step}: shoved the ball {plan.disturbances[step]}")
                alpha = min(1.0, (i + 1) / max(1, waypoint.steps - waypoint.ramp))
                commanded = src * (1.0 - alpha) + target * alpha
                yaw = yaw_from * (1.0 - alpha) + waypoint.yaw * alpha
                setpoint = (
                    carry.balance_setpoint()
                    if policy is None
                    else np.asarray(policy(carry.observe()), dtype=np.float64)
                )
                carry.command_tray(commanded, tray_quat(*setpoint, yaw))
                viewer.sync()
                step += 1
                if not carry.ball_on_tray():
                    print(f"  ball lost at step {step}")
                    return
                slack = period - (time.perf_counter() - tick)
                if slack > 0:
                    time.sleep(slack)
            yaw_from = waypoint.yaw
        print(f"  finished {step} steps with the ball still on the tray")


def main(argv: list[str] | None = None) -> int:
    """Render or view one tray-carry episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/tray_carry.mp4"))
    parser.add_argument("--camera", default="balancecam")
    parser.add_argument("--plan", choices=("straight", "random"), default="random")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument(
        "--size", type=int, nargs=2, default=(480, 640), metavar=("H", "W")
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="checkpoint to drive the tilt with; omit for the classical balancer",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action="store_true", help="live viewer instead of MP4")
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        help="start the tray somewhere different, inside the reach envelope",
    )
    parser.add_argument(
        "--handle-detector",
        type=Path,
        default=None,
        help="grasp from this checkpoint's estimate instead of reading the "
        "handle positions out of the simulator",
    )
    args = parser.parse_args(argv)

    plan = build_plan(args)
    # The policy's cameras are small and fixed by its checkpoint; the video wants
    # a presentable resolution, so they are separate rigs. Without a policy the
    # scene needs no small cameras at all.
    policy = None
    policy_cameras: tuple[str, ...] = ()
    image_size = (84, 112)
    if args.policy is not None:
        from openarm_gym.policies.tray_vision import load_controller

        # 50 Hz is BimanualTrayCarry's own default, which this script keeps.
        policy = load_controller(args.policy, device=args.device, control_hz=50.0)
        policy_cameras = tuple(policy.policy.cameras)
        image_size = tuple(policy.policy.image_size)
        policy.reset()

    handles = None
    if args.handle_detector is not None:
        from openarm_gym.policies.handle_vision import load_handle_detector

        handles = load_handle_detector(args.handle_detector, device=args.device)
        # The grasp needs its cameras rendered too, so they join the rig the
        # tilt policy asked for rather than replacing it.
        policy_cameras = tuple(
            dict.fromkeys(policy_cameras + tuple(handles.detector.cameras))
        )
        image_size = tuple(handles.detector.image_size)

    carry = BimanualTrayCarry(
        scene_path(), camera_names=policy_cameras, image_size=image_size
    )

    on_reset = None
    if args.randomize_layout:
        layout_rng = np.random.default_rng(args.seed).spawn(1)[0]

        def on_reset(c: BimanualTrayCarry) -> None:
            """Jitter the tray's starting pose, before the jaws close on it."""
            c.randomize_layout(layout_rng)

    handle_source = None
    if handles is not None:

        def handle_source(c: BimanualTrayCarry) -> dict:
            """Locate both posts from the standoff view."""
            return handles(c.observe(pixels=True))

    if args.viewer:
        if on_reset is not None or handle_source is not None:
            parser.error("--viewer does not take --randomize-layout or --handle-detector")
        run_viewer(carry, plan, policy)
        carry.close()
        return 0

    video = CameraRig(carry.model, (args.camera,), image_size=tuple(args.size))
    started = time.perf_counter()
    rollout = run_plan(
        carry,
        plan,
        tilt_policy=policy,
        video=(video, args.camera),
        on_reset=on_reset,
        handle_source=handle_source,
    )
    elapsed = time.perf_counter() - started

    path = write_mp4(args.out, rollout.frames, fps=args.fps)
    video.close()
    carry.close()
    grasped_by = "camera estimate" if handles is not None else "true handle positions"
    layout = "randomized layout" if args.randomize_layout else "authored layout"
    print(f"grasped from the {grasped_by}, {layout}")
    print(
        f"{'vision policy' if policy else 'classical balancer'} on the {args.plan} plan: "
        f"{rollout.steps} steps, ball "
        f"{'kept' if rollout.survived else f'lost at {rollout.lost_at}'}, "
        f"peak offset {np.abs(rollout.ball_xy).max() * 100:.1f} cm"
    )
    print(
        f"{len(rollout.frames)} frames in {elapsed:.0f}s -> {path} "
        f"({path.stat().st_size / 1e6:.1f} MB, {len(rollout.frames) / args.fps:.1f}s of video)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
