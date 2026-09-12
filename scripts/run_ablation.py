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

"""The force-conditioning ablation: does joint torque help, and where?

For each task, trains one policy with the torque channel and one without, over
several seeds, then evaluates both in closed loop on held-out layouts.

The design that makes the result interpretable:

*A negative control.* ``move_puck`` is pushing a light puck across a table. No
force feedback is needed to do it. If the torque channel "helps" there as much
as on the contact-rich tasks, the effect is a capacity or leakage artefact, not
contact sensing, and the whole result should be discarded.

*Held-out layouts.* Evaluation seeds are disjoint from the seeds the
demonstrations were recorded on, so the comparison measures generalisation.

*Multiple training seeds and intervals.* One run per arm cannot separate a real
effect from initialisation noise, and a rate without an interval invites
over-reading. Both arms get the same seeds, the same data and the same budget.

This script only orchestrates: the honest reading of the result is that a null
outcome is a result, and that anything it does show is evidence about simulated
contact, not about hardware.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openarm_gym.policies.runner import PolicyRunner, evaluate, wilson_interval
from openarm_gym.registry import CONTACT_RICH, CONTROL_TASKS, TASKS

#: Evaluation layouts start here, well clear of the recording seeds.
EVAL_SEED_BASE = 900_000


def train_one(
    dataset: Path, out: Path, *, use_torque: bool, steps: int, seed: int
) -> None:
    """Shell out to the training script for one ablation arm."""
    if out.exists():
        print(f"  {out.name} exists, skipping")
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_policy.py")),
        str(dataset),
        "--out",
        str(out),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--use-torque" if use_torque else "--no-torque",
    ]
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    """Train and evaluate both ablation arms across tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        default=list(CONTACT_RICH) + list(CONTROL_TASKS),
        choices=sorted(TASKS),
    )
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results", type=Path, default=Path("results/ablation.json"))
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--train-seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args(argv)

    eval_seeds = [EVAL_SEED_BASE + i for i in range(args.eval_episodes)]
    results: dict = {}

    for task in args.tasks:
        dataset = args.datasets / f"openarm_{task}"
        if not dataset.exists():
            print(f"! no dataset at {dataset}, skipping {task}")
            continue

        env_cls, _expert_cls = TASKS[task]
        results[task] = {
            "is_control": task in CONTROL_TASKS,
            "arms": {},
        }

        for arm, use_torque in (("with_torque", True), ("without_torque", False)):
            per_seed = []
            for seed in args.train_seeds:
                checkpoint = args.checkpoints / f"{task}_{arm}_s{seed}.pt"
                print(f"\n=== {task} / {arm} / seed {seed} ===")
                train_one(
                    dataset,
                    checkpoint,
                    use_torque=use_torque,
                    steps=args.steps,
                    seed=seed,
                )

                runner = PolicyRunner(checkpoint)
                env = env_cls(
                    camera_names=tuple(runner.camera_names),
                    domain_randomize=True,
                )
                outcome = evaluate(env, runner, eval_seeds)
                env.close()

                low, high = wilson_interval(outcome["successes"], outcome["episodes"])
                outcome["ci95"] = [low, high]
                per_seed.append(outcome)
                print(
                    f"  -> {outcome['successes']}/{outcome['episodes']} "
                    f"({100 * outcome['rate']:.0f}%, 95% CI "
                    f"{100 * low:.0f}-{100 * high:.0f}%)"
                )

            pooled_successes = sum(o["successes"] for o in per_seed)
            pooled_total = sum(o["episodes"] for o in per_seed)
            low, high = wilson_interval(pooled_successes, pooled_total)
            results[task]["arms"][arm] = {
                "per_seed": per_seed,
                "pooled": {
                    "successes": pooled_successes,
                    "episodes": pooled_total,
                    "rate": pooled_successes / max(pooled_total, 1),
                    "ci95": [low, high],
                },
            }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2))

    print(f"\n{'task':<14}{'control':>9}{'with torque':>16}{'without':>16}{'delta':>9}")
    for task, entry in results.items():
        arms = entry["arms"]
        if len(arms) < 2:
            continue
        with_rate = arms["with_torque"]["pooled"]["rate"]
        without_rate = arms["without_torque"]["pooled"]["rate"]
        print(
            f"{task:<14}{'yes' if entry['is_control'] else 'no':>9}"
            f"{100 * with_rate:>15.0f}%{100 * without_rate:>15.0f}%"
            f"{100 * (with_rate - without_rate):>+8.0f}%"
        )
    print(f"\nwrote {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
