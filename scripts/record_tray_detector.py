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

"""Record ball-detection samples with the ball placed *uniformly* on the tray.

The vision policy's job, in ``estimator`` mode, is to answer "where is the ball
on the tray and how fast is it moving". That is static perception -- the control
is already done by ``balance_law``. Collecting it from rollouts of the classical
controller is what made it fail, and the reason is structural rather than a
tuning problem:

**A stabilising controller keeps the ball centred.** Over 28 demonstration
episodes the ball's RMS offset was 10.5 mm against a 75 mm half-tray, so almost
every frame is the same picture. The resulting estimator was accurate to 4.6 mm
on the expert's own distribution and 72 mm once it drove itself, and at the first
step of an episode -- ball seeded 29 mm out -- it simply predicted the centre.
Widening the rollouts helped but only halved the gap, because the expert keeps
pulling the distribution back to the middle.

So this script does not use a policy at all. It flies the tray along ordinary
carry paths, and **teleports the ball to a uniformly sampled position and
velocity** every few control steps. Coverage of the tray is then uniform by
construction, and there is no policy generating the state distribution for
covariate shift to act on.

Two details make the samples valid:

* The ball is resampled, then allowed to run for exactly the frame stack's span
  before the sample is taken, so the two stacked frames show *consistent* motion
  and the velocity label describes what actually happened between them.
* The tray's tilt is commanded from a small random walk rather than from the
  balancer. A balancer would react to each teleport with a large corrective
  tilt, which would correlate tray orientation with ball position and hand the
  network a shortcut that does not exist at run time.

Samples are independent, so each one's frame stack is stored explicitly as
``(N, S, H, W, 3)``; :class:`~openarm_gym.policies.tray_vision.TrayVisionDataset`
detects that layout. Mix these files with rollout episodes freely -- the trainer
globs both.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import (
    BimanualTrayCarry,
    balance_law,
    tray_quat,
)
from openarm_gym.control.tray_task import flat_proprio, random_carry
from openarm_gym.policies.tray_vision import FRAME_STACK, FRAME_STRIDE


#: Sampling envelope for the ball, in the tray frame. The tray's top face is
#: 0.075 x 0.15 m half-extents; sampling right to the edge would include states
#: the ball cannot physically hold, so it stops just inside.
BALL_HALF_X = 0.068
BALL_HALF_Y = 0.140
#: Speed envelope, m/s. The classical balancer recovers 0.30 m/s along the short
#: axis, so sampling a little past that covers everything it will ever see.
MAX_SPEED = 0.35


def sample_ball_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw a uniform in-plane position and velocity, in the tray frame."""
    position = np.array(
        [rng.uniform(-BALL_HALF_X, BALL_HALF_X), rng.uniform(-BALL_HALF_Y, BALL_HALF_Y)]
    )
    # Uniform over the disc, not over (speed, angle): the latter crowds the
    # samples toward zero speed, which is the case already over-represented.
    angle = rng.uniform(0.0, 2.0 * np.pi)
    speed = MAX_SPEED * np.sqrt(rng.uniform(0.0, 1.0))
    return position, speed * np.array([np.cos(angle), np.sin(angle)])


def place_ball_moving(
    carry: BimanualTrayCarry, position: np.ndarray, velocity: np.ndarray
) -> None:
    """Teleport the ball onto the tray face with a given in-plane velocity."""
    import mujoco

    tray_pos, tray_quat_now = carry.tray_pose
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, tray_quat_now)
    rot = rot.reshape(3, 3)

    local = np.array([position[0], position[1], 0.025])
    carry.data.qpos[25:28] = tray_pos + rot @ local
    carry.data.qpos[28:32] = np.array([1.0, 0.0, 0.0, 0.0])
    carry.data.qvel[24:30] = 0.0
    # The tray is moving, so the ball's world velocity is the tray's plus the
    # in-plane velocity we are asking for relative to it.
    carry.data.qvel[24:27] = carry.data.qvel[18:21] + rot @ np.array(
        [velocity[0], velocity[1], 0.0]
    )
    mujoco.mj_forward(carry.model, carry.data)


def collect(
    carry: BimanualTrayCarry,
    rng: np.random.Generator,
    *,
    samples: int,
    tilt_sigma: float,
    layouts: int = 1,
    randomize_layout: bool = False,
) -> dict:
    """Fly carry paths, resampling the ball, and return one file's arrays.

    ``layouts`` splits the file into that many reset-and-regrasp cycles. With
    ``randomize_layout`` each cycle draws a new starting pose for the tray, which
    is the only way this recorder gets layout variety: a grasp captures a
    transform and the tray is then carried by it, so the layout cannot change
    without letting go. One grasp costs about two seconds against the hundred a
    file of samples takes, so splitting finely is nearly free.
    """
    span = 1 + FRAME_STRIDE * (FRAME_STACK - 1)
    pixels: dict[str, list[np.ndarray]] = {n: [] for n in carry.cameras.camera_names}
    ball_xy, ball_vxy, labels, proprio, positions = [], [], [], [], []
    tilt = np.zeros(2)

    for cycle in range(max(1, layouts)):
        carry.reset()
        if randomize_layout:
            carry.randomize_layout(rng)
        carry.grasp()
        origin = carry.tray_pose[0].copy()
        # Each cycle owns a slice of the file, so a run that ends early still
        # spreads its samples over every layout rather than over the first few.
        quota = min(samples, round(samples * (cycle + 1) / max(1, layouts)))
        while len(ball_xy) < quota:
            plan = random_carry(rng, n_disturbances=0)
            for waypoint in plan.waypoints:
                target = origin + waypoint.offset
                src = carry.tray_pose[0].copy()
                step = 0
                while step < waypoint.steps and len(ball_xy) < quota:
                    # A fresh uniform ball state, then exactly `span` control steps
                    # so the stacked frames bracket real, consistent motion.
                    position, velocity = sample_ball_state(rng)
                    place_ball_moving(carry, position, velocity)

                    window: dict[str, list[np.ndarray]] = {
                        n: [] for n in carry.cameras.camera_names
                    }
                    # Render only the offsets the stack actually uses. Rendering is
                    # the most expensive part of a control step, and for a 2-frame
                    # stack at stride 2 the middle frame is never looked at.
                    wanted = {k * FRAME_STRIDE for k in range(FRAME_STACK)}
                    for offset in range(span):
                        alpha = min(
                            1.0, (step + 1) / max(1, waypoint.steps - waypoint.ramp)
                        )
                        commanded = src * (1.0 - alpha) + target * alpha
                        # A small random walk on the tilt, independent of the ball.
                        # Driving it from the balancer would make the tray's
                        # orientation a giveaway for the ball's position.
                        tilt = np.clip(
                            0.9 * tilt + rng.normal(0.0, tilt_sigma, size=2), -0.12, 0.12
                        )
                        if offset in wanted:
                            for name, image in carry.observe(pixels=True)["pixels"].items():
                                window[name].append(image)
                        carry.command_tray(commanded, tray_quat(*tilt))
                        step += 1

                    relative, velocity_now = carry.ball_in_tray()
                    if not carry.ball_on_tray():
                        continue
                    for name in window:
                        pixels[name].append(np.stack(window[name]))
                    ball_xy.append(relative[:2].copy())
                    ball_vxy.append(velocity_now[:2].copy())
                    labels.append(balance_law(relative[:2], velocity_now[:2]))
                    proprio.append(flat_proprio(carry.observe(pixels=False)))
                    positions.append((commanded - origin).copy())

    arrays = {
        "labels": np.asarray(labels, dtype=np.float32),
        "ball_xy": np.asarray(ball_xy, dtype=np.float32),
        "ball_vxy": np.asarray(ball_vxy, dtype=np.float32),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "positions": np.asarray(positions, dtype=np.float32),
    }
    for name, frames in pixels.items():
        arrays[f"pixels_{name}"] = np.asarray(frames, dtype=np.uint8)
    return arrays


def main(argv: list[str] | None = None) -> int:
    """Record uniformly-sampled ball-detection data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("D:/openarm_data/tray_vision"))
    parser.add_argument("--files", type=int, default=8)
    parser.add_argument(
        "--samples", type=int, default=900, help="samples per file"
    )
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--image-size", type=int, nargs=2, default=(84, 112))
    parser.add_argument(
        "--cameras", nargs="+", default=list(BimanualTrayCarry.VISION_CAMERAS)
    )
    parser.add_argument(
        "--tilt-sigma",
        type=float,
        default=0.02,
        help="random-walk step on the commanded tray tilt, radians",
    )
    parser.add_argument(
        "--layouts",
        type=int,
        default=1,
        help="reset-and-regrasp cycles per file; with --randomize-layout, how "
        "many different tray poses the file covers",
    )
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        help="draw a new tray pose for each cycle",
    )
    parser.add_argument("--prefix", default="detector")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    carry = BimanualTrayCarry(
        scene_path(),
        camera_names=tuple(args.cameras),
        image_size=tuple(args.image_size),
    )

    started = time.perf_counter()
    total = 0
    try:
        for i in range(args.files):
            seed = args.seed + i
            began = time.perf_counter()
            arrays = collect(
                carry,
                np.random.default_rng(seed),
                samples=args.samples,
                tilt_sigma=args.tilt_sigma,
                layouts=args.layouts,
                randomize_layout=args.randomize_layout,
            )
            path = args.out / f"{args.prefix}_{seed:05d}.npz"
            np.savez_compressed(path, **arrays)
            total += len(arrays["ball_xy"])
            spread = np.sqrt((arrays["ball_xy"] ** 2).mean())
            print(
                f"  file {i:3d}  {len(arrays['ball_xy']):5d} samples  "
                f"{path.stat().st_size / 1e6:5.1f} MB  "
                f"{time.perf_counter() - began:5.1f}s  "
                f"RMS ball offset {spread * 1000:5.1f} mm"
            )
    finally:
        carry.close()

    (args.out / f"meta_{args.prefix}.json").write_text(
        json.dumps(
            {
                "scene": scene_path(),
                "cameras": list(args.cameras),
                "image_size": list(args.image_size),
                "samples": total,
                "sampling": "ball position and velocity uniform over the tray face",
                "ball_half_extents_m": [BALL_HALF_X, BALL_HALF_Y],
                "max_speed_ms": MAX_SPEED,
                "frame_stack": FRAME_STACK,
                "frame_stride": FRAME_STRIDE,
                "prestacked": True,
                "layouts_per_file": args.layouts,
                "randomized_layout": bool(args.randomize_layout),
            },
            indent=2,
        )
    )
    print(
        f"\n{total} samples in {time.perf_counter() - started:.0f}s -> {args.out}\n"
        "For comparison, rollout demonstrations carry an RMS ball offset of about "
        "10.5 mm; uniform sampling over a 75 mm half-tray gives roughly 50 mm."
    )
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
