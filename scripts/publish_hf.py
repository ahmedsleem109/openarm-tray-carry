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

"""Publish the two checkpoints to the HuggingFace Hub, with a model card.

Both networks together are under a megabyte, so this is about findability rather
than size: a checkpoint nobody can load is a checkpoint nobody will check.

The card is generated rather than hand-written, from the same numbers the
evaluation scripts wrote to ``results/``. A model card that drifts from the
measurements it cites is worse than no card, and hand-copied numbers drift.

Requires a login first, once:

    huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CARD = """---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - bimanual
  - mujoco
  - visuomotor
  - sim2real
library_name: pytorch
pipeline_tag: robotics
---

# Tray carry — vision policies for two OpenArm v2 arms

Two 7-DOF arms grasp a tray by its handle posts and carry it while keeping a
loose ball from rolling off, in MuJoCo, with collisions enabled and nothing
welded. Both halves of the task are driven by camera images.

Code, measurements and the full write-up:
**https://github.com/ahmedsleem109/openarm-tray-carry**

## What is here

| file | what it is | parameters |
|---|---|---|
| `policy.pt` | ball state estimator — two 84×112 cameras, two stacked frames | {ball_params:.2f} M |
| `detector.pt` | handle post detector — one frame per camera, run once per episode | {handle_params:.2f} M |
| `tray_ball_estimator.onnx` | the same estimator, ONNX opset 17 | — |
| `tray_handle_detector.onnx` | the same detector, ONNX opset 17 | — |
| `manifest.json` | input layout, normalisation statistics, output scales | — |

`manifest.json` matters: proprioception is standardised with the training set's
mean and standard deviation, and the networks emit O(1) numbers that become
metres and radians only after multiplying by the scales recorded there.

## How they are used

The estimator does **not** output an action. It estimates the ball's position in
the tray frame, and a classical PD law turns that into a tray tilt:

```
accel = -kp * ball_xy - kd * ball_vxy      kp = 12.0, kd = 4.5
pitch = accel_x / g                        roll = -accel_y / g
```

That split is deliberate and was measured. Regressing the tilt end-to-end from
the same weights produces outputs at roughly **half** the required magnitude —
MSE regression toward the mean — and half the loop gain is a different
controller. Estimating state keeps the gains exact by construction.

The velocity channel the network outputs is **not** used by the controller: at
84×112 the ball moves about a tenth of a pixel between stacked frames, and the
regressed velocity was measured at 11.7 mm/s RMSE against 12.7 mm/s for a
predictor that ignores the images entirely. The controller differences the
position estimate instead.

## Measured

12 unseen randomized carries, randomized tray layouts, identical plans and seeds
across rows. `classical` reads the ball's true state out of the simulator and is
the ceiling, not a competitor. The floor — carrying the tray level with no
balancing — is 1/12.

{table}

Grasping from `detector.pt` instead of from simulator state: **12/12** clean
grasps, tray displaced {shove:.1f} mm during the approach against {privileged_shove:.1f} mm
for the privileged path, handle estimate error {handle_error:.1f} mm.

## Training, in one paragraph

The lesson that made it work: in estimator mode the network is a **detector**,
so it has to be trained on the state distribution you want it accurate over —
which is *uniform*, not whatever a stabilising controller visits. Behaviour
cloning on the expert's own rollouts scored 0/12 while being accurate to 4.6 mm
on the expert's own states, because a good balancer keeps the ball centred and
the resulting data is one picture over and over. Training data here is
uniformly-sampled ball positions over the tray face, plus DAgger episodes, plus
demonstrations — all over randomized tray layouts.

## Limits

Every number above is a **simulation** number. There is no real arm. The cameras
are fixed to the world, so the handle detector predicts world coordinates and
extrinsic calibration is assumed away — on hardware that is a real step and it
is where the transfer gap lives. Rendering is MuJoCo's, with a gain-and-read
noise sensor model and no photorealism.

## Credit

The OpenArm v2 robot model is [Enactic's](https://github.com/enactic/openarm_mujoco),
used unmodified. Apache-2.0.
"""


def build_table(realism: dict) -> str:
    """Render the sim-to-real sweep as a markdown table from its own JSON."""
    rows = ["| condition | classical | vision | ball offset | sight error | grip |",
            "|---|---|---|---|---|---|"]
    for name, entry in realism.items():
        classical, vision = entry["classical"], entry["vision-estimator"]
        rows.append(
            f"| {name} | {classical['survived']}/{classical['episodes']} "
            f"| **{vision['survived']}/{vision['episodes']}** "
            f"| {vision['mean_peak_ball_m'] * 100:.1f} cm "
            f"| {vision['ball_sight_rmse_m'] * 1000:.1f} mm "
            f"| {vision['mean_grip_n']:.1f} N |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    """Build the card and upload every artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ahmedsleem109/openarm-tray-carry")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("D:/openarm_data/tray_vision_layout/policy.pt"),
    )
    parser.add_argument(
        "--handle-detector",
        type=Path,
        default=Path("D:/openarm_data/tray_handles/detector.pt"),
    )
    parser.add_argument("--export", type=Path, default=Path("export"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the card next to the exports and upload nothing",
    )
    args = parser.parse_args(argv)

    manifest = json.loads((args.export / "manifest.json").read_text())
    realism = json.loads((args.results / "tray_realism_layout.json").read_text())
    grasp = json.loads((args.results / "visual_grasp.json").read_text())

    card = CARD.format(
        ball_params=manifest["ball_estimator"]["parameters"] / 1e6,
        handle_params=manifest["handle_detector"]["parameters"] / 1e6,
        table=build_table(realism),
        shove=grasp["vision"]["mean_shove_m"] * 1000,
        privileged_shove=grasp["privileged"]["mean_shove_m"] * 1000,
        handle_error=grasp["vision"]["handle_error_m"] * 1000,
    )
    card_path = args.export / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"wrote {card_path} ({len(card.splitlines())} lines)")

    uploads = [
        (args.policy, "policy.pt"),
        (args.handle_detector, "detector.pt"),
        (args.export / "tray_ball_estimator.onnx", "tray_ball_estimator.onnx"),
        (args.export / "tray_handle_detector.onnx", "tray_handle_detector.onnx"),
        (args.export / "manifest.json", "manifest.json"),
        (card_path, "README.md"),
    ]
    missing = [str(src) for src, _ in uploads if not src.exists()]
    if missing:
        print("missing artifacts:\n  " + "\n  ".join(missing))
        return 1
    if args.dry_run:
        print("\ndry run — would upload:")
        for src, dest in uploads:
            print(f"  {src}  ->  {dest}  ({src.stat().st_size / 1e6:.2f} MB)")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    for src, dest in uploads:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"  uploaded {dest}")
    print(f"\nhttps://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
