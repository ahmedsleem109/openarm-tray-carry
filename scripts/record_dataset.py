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

"""Record scripted-expert demonstrations as a LeRobotDataset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from openarm_gym.recording import DatasetRecorder
from openarm_gym.registry import TASKS

#: Cameras recorded by default: one fixed overhead view and one wrist view per
#: arm. The head cameras are redundant with the ceiling for these tasks and
#: double the storage.
DEFAULT_CAMERAS = ("camera_ceiling", "camera_wrist_right", "camera_wrist_left")


def main(argv: list[str] | None = None) -> int:
    """Collect a demonstration dataset for one task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("-n", "--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument(
        "--cameras", nargs="*", default=list(DEFAULT_CAMERAS), help="camera names"
    )
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.015,
        help="joint command noise (rad) executed but not recorded, so the "
        "recorded label is the expert's correction from the perturbed state",
    )
    parser.add_argument(
        "--waypoint-noise",
        type=float,
        default=0.002,
        help="per-episode perception bias on the expert's targets (m)",
    )
    parser.add_argument("--domain-randomize", action="store_true", default=True)
    parser.add_argument("--no-domain-randomize", dest="domain_randomize",
                        action="store_false")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="cap on rollouts; 0 means 4x the episode target")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="cap each rollout below the task's own limit; a rollout that has "
        "not succeeded by then almost never does, and cutting it short is most "
        "of the collection budget",
    )
    parser.add_argument(
        "--images",
        action="store_true",
        default=True,
        help="store frames as images rather than video (default). Datasets "
        "this small are cheaper as images: video encoding dominates collection "
        "and decoding dominates the training dataloader",
    )
    parser.add_argument("--videos", dest="images", action="store_false")
    parser.add_argument("--writer-threads", type=int, default=8)
    args = parser.parse_args(argv)

    cameras = tuple(args.cameras)
    out = args.out or Path("datasets") / f"openarm_{args.task}"
    repo_id = args.repo_id or f"openarm/{args.task}"

    env_cls, expert_cls = TASKS[args.task]
    env = env_cls(
        camera_names=cameras,
        image_size=(args.height, args.width),
        domain_randomize=args.domain_randomize,
    )
    expert = expert_cls(env, waypoint_noise=args.waypoint_noise)

    recorder = DatasetRecorder(
        repo_id=repo_id,
        root=out,
        fps=int(env.control_hz),
        camera_names=cameras,
        image_size=(args.height, args.width),
        use_videos=not args.images,
        image_writer_threads=args.writer_threads,
    )

    max_attempts = args.max_attempts or args.episodes * 4
    already = recorder.resumed_from
    if already:
        print(f"  resuming: {already} episodes already recorded")
    kept = 0
    attempts = 0
    started = time.perf_counter()

    # Seeds continue past what is already stored, so a resumed run collects
    # new layouts rather than repeating the ones already on disk.
    while already + kept < args.episodes and attempts < max_attempts:
        stats = recorder.record(
            env,
            expert,
            args.seed + already * 3 + attempts,
            action_noise=args.action_noise,
            max_steps=args.max_steps or None,
        )
        attempts += 1
        kept += stats.success
        if attempts % 5 == 0 or already + kept >= args.episodes:
            rate = 100.0 * kept / attempts
            elapsed = time.perf_counter() - started
            print(
                f"  kept {already + kept:4d}/{args.episodes}  "
                f"attempts {attempts:4d}  ({rate:.0f}% success)  {elapsed:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    lengths = [e.steps for e in recorder.episodes]
    print(
        f"\n{args.task}: wrote {kept} episodes from {attempts} rollouts "
        f"({100.0 * kept / max(attempts, 1):.0f}% success) in {elapsed:.0f}s"
    )
    if lengths:
        print(
            f"  {sum(lengths)} frames, mean episode {np.mean(lengths):.0f} steps "
            f"({sum(lengths) / int(env.control_hz):.0f}s of robot time)"
        )
    print(f"  dataset at {out}")
    env.close()
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
