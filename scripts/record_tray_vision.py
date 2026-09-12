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

"""Record tray-carry demonstrations as (images -> tilt setpoint) pairs.

The classical balancer drives; every control step contributes one training
sample: the camera images and proprioception a policy is allowed to see, the
``(roll, pitch)`` setpoint the controller produced, and -- as an auxiliary
target -- the ball's true offset in the tray frame, which is exactly the
quantity vision has to recover.

One compressed ``.npz`` per episode, written as it completes, because a run that
dies of memory pressure two thirds of the way through should still leave two
thirds of a dataset. Episodes where the expert lost the ball are **dropped**:
behaviour cloning cannot learn from a demonstration of failing.

Pixels are recorded **clean**. Sensor noise is applied as augmentation at
training time instead, so it can be ablated without re-recording 800 MB.

**The recording distribution is deliberately harsher than the task.** A plain
demonstration of the carry is dominated by the state the expert has already
recentred: over 28 such episodes the ball's RMS offset was 10.5 mm against a
75 mm half-tray, so an estimator trained on them is at its worst exactly where a
closed loop spends its first seconds. Measured consequence: the estimate was
accurate to 4.2 mm on held-out expert episodes and 72 mm driving itself, and at
step 0 -- ball seeded 29 mm off centre -- it simply predicted the centre.

Three knobs widen it, and they are not equally useful. Seeding the ball further
out (``--ball-offset``) fixes only the first frames. Shoving it
(``--disturbances``) barely helps, because the expert recentres within a few
tenths of a second -- six shoves an episode moved the RMS offset from 10.5 mm to
only 13 mm. What works is ``--action-noise``: the executed tilt is perturbed
while the *recorded label stays the clean expert setpoint*, so the ball is held
off centre continuously and every frame is labelled with the correction from
where it actually is. This is the same recipe the repository's scripted experts
already use, for the same reason.

The default of 0.02 rad is where it was measured to sit best, and the sweep is
worth keeping because it is not monotonic:

=============  ===============  ==================
action noise   expert survival  RMS ball offset
=============  ===============  ==================
0.000 rad      7/8              13.2 mm
0.020 rad      7/8              **23.7 mm**
0.030 rad      4/8              15.8 mm
0.045 rad      4/8              22.2 mm
=============  ===============  ==================

Past 0.02 the expert itself starts dropping the ball, and coverage *falls* --
survivorship, since only the calm episodes live long enough to be written. More
noise is not more recovery data; it is fewer episodes of it.

The evaluation task keeps ``random_carry``'s own defaults, so the training
distribution is a superset of it rather than a redefinition of it.

**``--policy`` is the stronger version of the same idea (DAgger).** Perturbing the
expert's command widens the distribution blindly; driving with the *policy* and
labelling with the expert widens it exactly where the policy actually goes wrong.
Episodes collected this way are **kept even when the ball is lost**, which is the
opposite of the rule for demonstrations: a failed demonstration teaches nothing,
but a failure the expert can label is the most informative frame there is. Every
recorded label is still the expert's clean setpoint at the state the policy
reached, and the run stops at the moment the ball leaves the tray, so no label is
taken from a state where the expert has given up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry
from openarm_gym.control.tray_task import random_carry, run_plan



def record_episode(
    carry: BimanualTrayCarry,
    seed: int,
    *,
    disturbances: int,
    ball_offset: float,
    action_noise: float,
    policy=None,
    randomize_layout: bool = False,
) -> tuple[dict | None, dict]:
    """Fly one randomized carry and return ``(sample arrays, stats)``.

    With no ``policy`` the expert drives and a lost ball means the episode is
    discarded -- the arrays come back ``None``. With a ``policy`` driving, a lost
    ball is the point of the exercise and the episode is kept.

    ``randomize_layout`` jitters where the tray starts. Its generator is spawned
    from the episode seed rather than shared with the plan's or the action
    noise's, so turning it on leaves both of those drawing exactly what they drew
    before.
    """
    plan = random_carry(
        np.random.default_rng(seed),
        n_disturbances=disturbances,
        ball_offset=ball_offset,
    )
    started = time.perf_counter()
    if policy is not None:
        policy.reset()
    on_reset = None
    if randomize_layout:
        layout_rng = np.random.default_rng(200_000 + seed).spawn(1)[0]

        def on_reset(carry_to_place: BimanualTrayCarry) -> None:
            """Jitter the tray's starting pose, before the jaws close on it."""
            carry_to_place.randomize_layout(layout_rng)
    rollout = run_plan(
        carry,
        plan,
        tilt_policy=policy,
        record_pixels=True,
        action_noise=action_noise,
        action_rng=np.random.default_rng(100_000 + seed),
        on_reset=on_reset,
    )
    stats = {
        "seed": seed,
        "survived": bool(rollout.survived),
        "steps": int(rollout.steps),
        "planned_steps": int(plan.steps),
        "mean_tracking": rollout.mean_tracking,
        "seconds": time.perf_counter() - started,
        "driver": "policy" if policy is not None else "expert",
    }
    # An episode with no steps has no traces to summarise, and reducing over an
    # empty axis raises rather than returning something harmless.
    if not rollout.steps:
        return None, stats
    stats["peak_ball_offset"] = [float(v) for v in np.abs(rollout.ball_xy).max(axis=0)]
    stats["rms_ball_offset"] = float(np.sqrt((rollout.ball_xy**2).mean()))
    if not rollout.survived and policy is None:
        return None, stats

    arrays = {
        # The CLEAN expert setpoint, not the perturbed one that was executed:
        # the label has to be the correction, or the policy learns the noise.
        "labels": rollout.labels,
        "ball_xy": rollout.ball_xy,
        "ball_vxy": rollout.ball_vxy,
        "proprio": rollout.proprio,
        "positions": rollout.positions,
    }
    for name, frames in rollout.pixels.items():
        arrays[f"pixels_{name}"] = frames
    return arrays, stats


def main(argv: list[str] | None = None) -> int:
    """Record a dataset of tray-carry demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("D:/openarm_data/tray_vision"),
        help="directory to write one .npz per surviving episode into",
    )
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disturbances",
        type=int,
        default=6,
        help="shoves per episode; more than the task uses, on purpose (see above)",
    )
    parser.add_argument(
        "--ball-offset",
        type=float,
        default=0.05,
        help="half-width of the seeded starting offset, metres",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.02,
        help="radians of tilt perturbation executed but not recorded as the label",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="drive with this checkpoint and label with the expert (DAgger)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        help="jitter where the tray starts, so the estimator sees more than one",
    )
    parser.add_argument(
        "--prefix",
        default="episode",
        help="filename prefix, so a DAgger round lands beside the demonstrations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=(84, 112),
        metavar=("H", "W"),
        help="small on purpose: an episode is ~480 steps over two cameras",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(BimanualTrayCarry.VISION_CAMERAS),
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    carry_hz = 50.0
    policy = None
    if args.policy is not None:
        from openarm_gym.policies.tray_vision import load_controller

        policy = load_controller(
            args.policy, device=args.device, control_hz=carry_hz
        )
        args.cameras = list(policy.policy.cameras)
        args.image_size = list(policy.policy.image_size)
        print(f"DAgger round: driving with {args.policy}")
    carry = BimanualTrayCarry(
        scene_path(),
        camera_names=tuple(args.cameras),
        image_size=tuple(args.image_size),
    )

    kept, dropped, all_stats = 0, 0, []
    started = time.perf_counter()
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            arrays, stats = record_episode(
                carry, seed,
                disturbances=args.disturbances,
                ball_offset=args.ball_offset,
                action_noise=args.action_noise,
                policy=policy,
                randomize_layout=args.randomize_layout,
            )
            all_stats.append(stats)
            if arrays is None:
                dropped += 1
                print(f"  ep {i:3d} seed {seed:5d}  DROPPED (ball lost at {stats['steps']})")
                continue
            path = args.out / f"{args.prefix}_{seed:05d}.npz"
            np.savez_compressed(path, **arrays)
            kept += 1
            print(
                f"  ep {i:3d} seed {seed:5d}  {stats['steps']:4d} steps  "
                f"{path.stat().st_size / 1e6:5.1f} MB  {stats['seconds']:4.1f}s  "
                f"peak ball ({stats['peak_ball_offset'][0]:.3f}, "
                f"{stats['peak_ball_offset'][1]:.3f}) m  "
                f"rms {stats['rms_ball_offset'] * 1000:4.1f} mm"
            )
    finally:
        carry.close()

    meta = {
        "scene": scene_path(),
        "cameras": list(args.cameras),
        "image_size": list(args.image_size),
        "episodes_kept": kept,
        "episodes_dropped": dropped,
        "expert": "BimanualTrayCarry.balance_setpoint (classical PD)",
        "disturbances_per_episode": args.disturbances,
        "ball_offset_half_width_m": args.ball_offset,
        "action_noise_rad": args.action_noise,
        "driver": "policy (DAgger)" if policy is not None else "expert",
        "label": "(roll, pitch) tilt setpoint, radians",
        "pixels": "clean; sensor noise is a training-time augmentation",
        "stats": all_stats,
    }
    # A DAgger round must not overwrite the demonstrations' metadata.
    meta_name = "meta.json" if policy is None else f"meta_{args.prefix}.json"
    (args.out / meta_name).write_text(json.dumps(meta, indent=2))
    elapsed = time.perf_counter() - started
    total_mb = sum(p.stat().st_size for p in args.out.glob("*.npz")) / 1e6
    print(
        f"\nkept {kept}/{args.episodes} episodes ({dropped} dropped), "
        f"{total_mb:.0f} MB in {elapsed:.0f}s -> {args.out}"
    )
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
