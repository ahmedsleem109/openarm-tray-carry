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

"""Evaluate a trained policy in closed loop and append the result to a file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from openarm_gym.policies.runner import PolicyRunner, evaluate, wilson_interval
from openarm_gym.registry import TASKS

#: Evaluation layouts start well clear of the seeds used for recording.
EVAL_SEED_BASE = 900_000


def main(argv: list[str] | None = None) -> int:
    """Run one checkpoint over held-out layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("-n", "--episodes", type=int, default=30)
    parser.add_argument("--arm", default=None, help="label for the results file")
    parser.add_argument("--results", type=Path, default=Path("results/ablation.json"))
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)

    runner = PolicyRunner(args.checkpoint)
    env_cls, _expert_cls = TASKS[args.task]
    env = env_cls(camera_names=tuple(runner.camera_names), domain_randomize=True)
    if args.max_steps:
        env.MAX_STEPS = args.max_steps

    seeds = [EVAL_SEED_BASE + i for i in range(args.episodes)]
    started = time.perf_counter()
    outcome = evaluate(env, runner, seeds)
    elapsed = time.perf_counter() - started
    env.close()

    low, high = wilson_interval(outcome["successes"], outcome["episodes"])
    outcome["ci95"] = [low, high]
    arm = args.arm or ("with_torque" if runner.use_torque else "without_torque")

    print(
        f"{args.task} / {arm}: {outcome['successes']}/{outcome['episodes']} "
        f"({100 * outcome['rate']:.0f}%, 95% CI {100 * low:.0f}-{100 * high:.0f}%) "
        f"mean length {outcome['mean_length']:.0f} steps, {elapsed:.0f}s"
    )

    # Merge into the results file rather than overwriting, so arms can be run
    # one at a time.
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.results.exists():
        results = json.loads(args.results.read_text())
    entry = results.setdefault(
        args.task, {"is_control": args.task == "move_puck", "arms": {}}
    )
    entry["arms"][arm] = {"per_seed": [outcome], "pooled": outcome}
    args.results.write_text(json.dumps(results, indent=2))
    print(f"  merged into {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
