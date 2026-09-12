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

"""Compare the vision policy against the classical balancer, closed loop.

Five controllers on **the same plans and the same seeds**, because a survival
rate quoted on different manoeuvres is not a comparison:

* ``level`` -- never tilts. The floor: what the carry achieves with no balancing
  at all, and the same comparison the controller tests assert.
* ``classical`` -- :meth:`BimanualTrayCarry.balance_setpoint`, reading the ball's
  state out of the simulator. The ceiling, and the source of the training labels.
* ``vision-estimator`` -- the learned policy estimating the ball's position from
  pixels and driving the *classical* balance law, damping on the differenced
  position estimate.
* ``vision-head-velocity`` -- the same, but damping on the network's regressed
  velocity output instead of the differenced one.
* ``vision-tilt`` -- the same weights, regressing the tilt end to end.

The last three share a checkpoint and differ only in how the loop is closed,
which is what makes the comparison worth running: open-loop regression error does
not tell you whether a balancer balances. The headline number is survival, with
the tilt tracking error and the ball-position error reported alongside to say
*why* a controller failed -- whether it could not see the ball, or could see it
and still did the wrong thing.

The run happens under one sim-to-real :class:`Condition`, ``clean`` by default so
the numbers stay comparable with everything measured before. Pass ``--condition``
for a named preset, or the individual flags to build one. For the whole table at
once -- every condition, classical against vision -- use
``evaluate_tray_realism.py``, which drives this same machinery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.tray_eval import (
    LoggedVision,
    Stopwatch,
    add_condition_args,
    build_carry,
    condition_from_args,
    evaluate_controller,
    level_policy,
)
from openarm_gym.control.tray_task import random_carry
from openarm_gym.policies.tray_vision import load_controller



def main(argv: list[str] | None = None) -> int:
    """Evaluate every controller on the same plans and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("D:/openarm_data/tray_vision/policy.pt")
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument(
        "--seed",
        type=int,
        default=9000,
        help="held well clear of the recording seeds, so these plans are unseen",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None)
    add_condition_args(parser)
    args = parser.parse_args(argv)
    condition = condition_from_args(args)

    # Loaded once first only to learn which cameras the checkpoint expects; the
    # scene has to exist before a controller can be told the rate it runs at.
    spec = load_controller(args.checkpoint, device=args.device)
    carry = build_carry(
        scene_path(),
        condition,
        camera_names=tuple(spec.policy.cameras),
        image_size=tuple(spec.policy.image_size),
    )

    def controller(**kwargs):
        """Load the checkpoint at the rate the carry actually runs at."""
        return load_controller(
            args.checkpoint, device=args.device, control_hz=carry.control_hz, **kwargs
        )

    policies = {
        "level": level_policy,
        "classical": None,
        # The same weights three times over, differing only in how the loop closes.
        "vision-estimator": LoggedVision(controller(mode="estimator"), carry),
        "vision-head-velocity": LoggedVision(
            controller(mode="estimator", velocity_source="head"), carry
        ),
        "vision-tilt": LoggedVision(controller(mode="tilt"), carry),
    }

    plans = [random_carry(np.random.default_rng(args.seed + i)) for i in range(args.episodes)]
    print(f"condition {condition.name}: {condition.summary()}")
    clock = Stopwatch()
    results = {
        name: evaluate_controller(
            carry, plans, policy, condition=condition, seed=args.seed, label=name
        )
        for name, policy in policies.items()
    }
    carry.close()

    print(f"\n{'controller':<22}{'survived':>10}{'peak ball':>12}{'tilt err':>12}{'sight err':>12}")
    for name, row in results.items():
        sight = row["ball_sight_rmse_m"]
        print(
            f"{name:<22}{row['survived']:>5}/{row['episodes']:<4}"
            f"{row['mean_peak_ball_m'] * 100:>9.1f} cm"
            f"{row['tilt_rmse_vs_classical_rad'] * 1000:>9.1f} mrad"
            + (f"{sight * 1000:>9.1f} mm" if sight is not None else f"{'-':>12}")
        )
    print(f"\n{args.episodes} plans x {len(results)} controllers in {clock.elapsed:.0f}s")
    if args.json:
        args.json.write_text(
            json.dumps({"condition": condition.name, "results": results}, indent=2)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
