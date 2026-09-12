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

"""Grasp the tray from pixels, over randomized layouts, against the privileged grasp.

Both arms approach and close on the handle posts, with exactly one thing
differing between the two rows:

* ``privileged`` reads the posts' positions out of the simulator. This is what
  every earlier number in this project was measured with.
* ``vision`` sees two camera images from the arms' retracted pose and estimates
  the same two positions. The approach, the lift height and the close are
  unchanged, so any difference between the rows is the estimate and nothing else.

Each layout is flown **twice per row**, because the two things worth knowing are
measured at different moments:

1. *The grasp itself.* Reset, jitter the layout, estimate, approach, close --
   then read the tray's displacement, the contact count and the grip force. The
   failure this project already fixed once was the gripper's palm bulldozing the
   tray 32.5 mm forward before the jaws arrived, and a bad estimate reintroduces
   precisely that, so it is measured rather than assumed. The gate is under
   10 mm; the privileged approach achieves 5.8 mm.
2. *Whether it holds.* The same layout, flown through a full randomized carry
   with the classical balancer driving. A grasp that is merely marginal passes
   the first test and slips during the second.

The balancer is the classical one on purpose: this script measures the grasp, and
putting the learned ball policy in the loop as well would confound the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry
from openarm_gym.control.tray_eval import Stopwatch
from openarm_gym.control.tray_task import random_carry, run_plan
from openarm_gym.policies.handle_vision import SIDES, load_handle_detector


#: Tray displacement during the approach that counts as a clean grasp, metres.
SHOVE_GATE = 0.010


def true_handles(carry: BimanualTrayCarry) -> dict[str, np.ndarray]:
    """The posts' actual positions, used only to score the estimate."""
    return {side: carry.handle_pos(side).copy() for side in SIDES}


def main(argv: list[str] | None = None) -> int:
    """Fly both grasps over the same randomized layouts and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("D:/openarm_data/tray_handles/detector.pt"),
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--fixed-layout",
        action="store_true",
        help="keep the authored layout, which separates the estimate's error "
        "from the jitter it has to cope with",
    )
    parser.add_argument("--json", type=Path, default=Path("../results/visual_grasp.json"))
    args = parser.parse_args(argv)

    estimator = load_handle_detector(args.checkpoint, device=args.device)
    carry = BimanualTrayCarry(
        scene_path(),
        camera_names=tuple(estimator.detector.cameras),
        image_size=tuple(estimator.detector.image_size),
    )
    plans = [
        random_carry(np.random.default_rng(args.seed + i)) for i in range(args.episodes)
    ]

    def layout(index: int):
        """This episode's layout draw, or ``None`` when it is held fixed.

        Its own spawned stream, and seeded from the episode index alone, so the
        privileged and the vision row meet identical layouts.
        """
        if args.fixed_layout:
            return None
        rng = np.random.default_rng(args.seed + index).spawn(1)[0]
        return lambda c: c.randomize_layout(rng)

    def estimate(c: BimanualTrayCarry) -> dict[str, np.ndarray]:
        """Where the detector thinks the posts are, from the standoff view."""
        return estimator(c.observe(pixels=True))

    results: dict[str, dict] = {}
    clock = Stopwatch()
    for name, use_vision in (("privileged", False), ("vision", True)):
        shoves, grips, contacts, errors = [], [], [], []
        clean, held, kept = 0, 0, 0
        for i, plan in enumerate(plans):
            # 1. the grasp, measured where it happens.
            carry.reset()
            draw = layout(i)
            if draw is not None:
                draw(carry)
            before = carry.tray_pose[0].copy()
            handles = None
            if use_vision:
                handles = estimate(carry)
                errors.append(estimator.error(true_handles(carry)))
            report = carry.grasp(handles=handles)
            shoves.append(float(np.linalg.norm(carry.tray_pose[0][:2] - before[:2])))
            grips.append(min(report[s]["force"] for s in SIDES))
            contacts.append(min(report[s]["contacts"] for s in SIDES))
            clean += shoves[-1] < SHOVE_GATE and contacts[-1] >= 1

            # 2. whether it survives a carry. run_plan resets and re-grasps, so
            # the layout draw is handed to it rather than reused from above --
            # the generator was already consumed, so it is rebuilt from the same
            # seed and lands on the same layout.
            rollout = run_plan(
                carry,
                plan,
                on_reset=layout(i),
                handle_source=estimate if use_vision else None,
            )
            # Tracking is what exposes a slipping grasp: a tray that is no longer
            # held stops following the commanded pose long before it is dropped.
            holding = rollout.mean_tracking < 0.02
            held += holding
            kept += rollout.survived
            print(
                f"  {name:<11} ep {i:2d}  shove {shoves[-1] * 1000:5.1f} mm  "
                f"grip {grips[-1]:5.1f} N  contacts {contacts[-1]}  "
                f"held={str(bool(holding)):<5} ball={str(rollout.survived):<5}"
                + (f"  handle err {errors[-1] * 1000:5.1f} mm" if use_vision else ""),
                flush=True,
            )
        results[name] = {
            "episodes": len(plans),
            "clean_grasps": clean,
            "grasp_held": held,
            "ball_kept": kept,
            "mean_shove_m": float(np.mean(shoves)),
            "max_shove_m": float(np.max(shoves)),
            "mean_grip_n": float(np.mean(grips)),
            "min_contacts": int(np.min(contacts)),
            "handle_error_m": float(np.mean(errors)) if use_vision else None,
            "max_handle_error_m": float(np.max(errors)) if use_vision else None,
        }

    carry.close()
    header = (
        f"{'grasp':<12}{'clean':>9}{'held':>9}{'ball kept':>11}"
        f"{'shove':>11}{'worst':>10}{'grip':>9}{'handle err':>13}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for name, row in results.items():
        error = row["handle_error_m"]
        print(
            f"{name:<12}{row['clean_grasps']:>5}/{row['episodes']:<4}"
            f"{row['grasp_held']:>5}/{row['episodes']:<4}"
            f"{row['ball_kept']:>6}/{row['episodes']:<4}"
            f"{row['mean_shove_m'] * 1000:>8.1f} mm"
            f"{row['max_shove_m'] * 1000:>7.1f} mm"
            f"{row['mean_grip_n']:>7.1f} N"
            + (f"{error * 1000:>10.1f} mm" if error is not None else f"{'-':>13}")
        )
    print(f"\n{args.episodes} layouts x 2 grasps in {clock.elapsed / 60:.1f} min")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
