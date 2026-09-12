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

"""Fly the vision policy and the classical balancer under every sim-to-real condition.

This is an evaluation, not a build. Camera noise, torque actuation, torque
limits, command latency, transmission backlash and dynamics randomization are
all already implemented and tested; they default to off, which is why the
project's headline "vision 12/12" was measured on clean pixels and position
control. This script turns each of them on in turn and reports what the drop
actually is.

Two controllers per condition, on identical plans and identical seeds:

* ``classical``, reading the ball's state out of the simulator -- the ceiling,
  and the row that says whether a condition hurt the *carry* or hurt *seeing*.
* ``vision-estimator``, seeing only two 84x112 camera images and joint angles.

Reporting both is the point. A condition that costs the vision policy exactly
what it costs the classical controller has not found a perception weakness; it
has found a mechanics weakness that the estimator was never going to fix.
Backlash is the one to watch, since it is what hurt the classical controller most
(it unloads the grip: 41.7 N -> 17.8 N at 0.5 deg).

Every number this prints is a simulation number.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.tray_eval import (
    CONDITIONS,
    LoggedVision,
    Stopwatch,
    build_carry,
    evaluate_controller,
)
from openarm_gym.control.tray_task import random_carry
from openarm_gym.policies.tray_vision import load_controller



def main(argv: list[str] | None = None) -> int:
    """Run the sweep and print one row per condition per controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("D:/openarm_data/tray_vision/policy.pt")
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument(
        "--seed",
        type=int,
        default=9000,
        help="the same seeds the clean evaluation uses, so the rows line up",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(CONDITIONS),
        choices=sorted(CONDITIONS),
        help="which named conditions to fly; default is all of them, in order",
    )
    parser.add_argument(
        "--randomize-layout",
        action="store_true",
        help="jitter the tray's starting pose on top of every condition",
    )
    parser.add_argument("--json", type=Path, default=Path("../results/tray_realism.json"))
    args = parser.parse_args(argv)

    # Loaded once first only to learn which cameras the checkpoint expects. The
    # plans are built once and shared by every condition: the whole comparison
    # rests on each controller flying the same manoeuvres.
    spec = load_controller(args.checkpoint, device=args.device)
    plans = [random_carry(np.random.default_rng(args.seed + i)) for i in range(args.episodes)]
    order = [name for name in CONDITIONS if name in set(args.conditions)]

    clock = Stopwatch()
    results: dict[str, dict] = {}
    for name in order:
        condition = CONDITIONS[name]
        if args.randomize_layout:
            # Layout is scene variation rather than a sim-to-real knob, so it is
            # not in the presets; it composes with all of them.
            condition = replace(condition, randomize_layout=True)
        print(f"\n=== {name}: {condition.summary()}", flush=True)
        # One controller per condition, because torque control, latency and
        # backlash are all fixed at construction.
        carry = build_carry(
            scene_path(),
            condition,
            camera_names=tuple(spec.policy.cameras),
            image_size=tuple(spec.policy.image_size),
        )
        vision = LoggedVision(
            load_controller(
                args.checkpoint,
                device=args.device,
                control_hz=carry.control_hz,
                mode="estimator",
            ),
            carry,
        )
        row = {}
        for label, policy in (("classical", None), ("vision-estimator", vision)):
            row[label] = evaluate_controller(
                carry, plans, policy, condition=condition, seed=args.seed, label=label
            )
        carry.close()
        results[name] = {"summary": condition.summary(), **row}
        if args.json:
            # Written after every condition: the sweep takes tens of minutes, and
            # a partial table is worth more than a lost one.
            args.json.write_text(json.dumps(results, indent=2))

    header = (
        f"{'condition':<16}{'classical':>12}{'vision':>10}"
        f"{'cls peak':>11}{'vis peak':>11}{'sight err':>12}{'grip':>10}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for name, row in results.items():
        cls, vis = row["classical"], row["vision-estimator"]
        print(
            f"{name:<16}"
            f"{cls['survived']:>7}/{cls['episodes']:<4}"
            f"{vis['survived']:>5}/{vis['episodes']:<4}"
            f"{cls['mean_peak_ball_m'] * 100:>8.1f} cm"
            f"{vis['mean_peak_ball_m'] * 100:>8.1f} cm"
            f"{vis['ball_sight_rmse_m'] * 1000:>9.1f} mm"
            f"{vis['mean_grip_n']:>8.1f} N"
        )
    print(
        f"\n{len(order)} conditions x 2 controllers x {args.episodes} plans "
        f"in {clock.elapsed / 60:.1f} min"
    )
    if args.json:
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
