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

"""Run the scripted experts and report their success rates."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from openarm_gym.registry import TASKS


def run_episode(env, expert, seed: int, action_noise: float) -> tuple[bool, float, int]:
    """Run one episode and return ``(success, return, steps)``."""
    env.reset(seed=seed)
    expert.reset(seed=seed)

    total = 0.0
    steps = 0
    while True:
        action = expert.act()
        _obs, reward, terminated, truncated, info = env.step(
            expert.perturb(action, action_noise)
        )
        total += reward
        steps += 1
        if terminated or truncated:
            return bool(info["is_success"]), total, steps


def evaluate(name: str, args) -> tuple[int, int, list[float], list[int]]:
    """Roll one task out over randomized resets."""
    env_cls, expert_cls = TASKS[name]
    env = env_cls(domain_randomize=args.domain_randomize)
    expert = expert_cls(env, waypoint_noise=args.noise)

    successes = 0
    returns: list[float] = []
    lengths: list[int] = []
    for i in range(args.episodes):
        ok, total, steps = run_episode(env, expert, args.seed + i, args.action_noise)
        successes += ok
        returns.append(total)
        lengths.append(steps)
        if args.verbose:
            print(
                f"  {name:11s} ep {i:3d} seed={args.seed + i:<5d} "
                f"{'SUCCESS' if ok else 'fail   '} "
                f"return={total:8.1f} steps={steps:4d}"
            )
    env.close()
    return successes, args.episodes, returns, lengths


def main(argv: list[str] | None = None) -> int:
    """Roll the experts out and print a success table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks", nargs="*", default=list(TASKS), help="tasks to run (default: all)"
    )
    parser.add_argument("-n", "--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise", type=float, default=0.0, help="waypoint perception bias (m)"
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.0,
        help="per-step joint command noise (rad), for DART-style collection",
    )
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows = []
    for name in args.tasks:
        if name not in TASKS:
            parser.error(f"unknown task {name!r}; known: {sorted(TASKS)}")
        successes, total, returns, lengths = evaluate(name, args)
        rows.append((name, successes, total, float(np.mean(returns)), float(np.mean(lengths))))
    elapsed = time.perf_counter() - started

    print(f"\n{'task':<14}{'success':>12}{'rate':>8}{'return':>10}{'length':>9}")
    for name, successes, total, mean_return, mean_length in rows:
        print(
            f"{name:<14}{successes:>6}/{total:<5}{100.0 * successes / total:>6.0f}%"
            f"{mean_return:>10.1f}{mean_length:>9.0f}"
        )
    overall = sum(r[1] for r in rows), sum(r[2] for r in rows)
    print(
        f"\noverall {overall[0]}/{overall[1]} "
        f"({100.0 * overall[0] / max(overall[1], 1):.0f}%) in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
