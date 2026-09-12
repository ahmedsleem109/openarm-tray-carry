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

"""Record what the cameras see just before a grasp, labelled with the handles.

One sample is one *layout*: reset the arms to their retracted pose, jitter where
the tray is, render both cameras, and record the two handle posts' world
positions. There is no rollout and no policy involved, which is the same lesson
the ball detector taught -- "where is the handle" is static perception, so it
should be trained on the state distribution it has to be accurate over rather
than on whatever trajectory a controller happens to generate.

Two choices worth stating:

* **The layout envelope here is deliberately wider than the evaluation's.** The
  detector is trained over a larger range of tray poses than it is later asked
  about, so evaluation layouts sit in the interior of the training distribution
  rather than on its edge.
* **The ball is placed somewhere random on the tray.** It is the brightest thing
  in the scene and it has nothing to do with where the handles are. Left at the
  centre of every sample it would be a fixed landmark the network could key the
  tray's position to, and that shortcut would evaporate the moment a real
  episode started with the ball off centre.

Camera noise is not baked in; the trainer applies it, so one dataset serves both
the clean and the noisy case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path

from openarm_gym.control.bimanual_tray import BimanualTrayCarry, LayoutRanges
from openarm_gym.policies.handle_vision import SIDES, encode_handles


#: Training envelope, wider than :class:`LayoutRanges`'s evaluation default.
TRAIN_LAYOUT = LayoutRanges(x=(-0.03, 0.055), y=(-0.03, 0.03), yaw=(-0.07, 0.07))

#: Ball placement envelope on the tray face, metres in the tray frame.
BALL_HALF_X = 0.055
BALL_HALF_Y = 0.120


def collect(
    carry: BimanualTrayCarry,
    rng: np.random.Generator,
    *,
    samples: int,
    ranges: LayoutRanges,
) -> dict[str, np.ndarray]:
    """Render ``samples`` random layouts from the arms' retracted pose."""
    pixels: dict[str, list[np.ndarray]] = {n: [] for n in carry.cameras.camera_names}
    labels, layouts = [], []
    for _ in range(samples):
        # reset() is what puts the arms at the standoff pose the detector will be
        # called from at run time, so the training view and the deployed view are
        # the same view by construction.
        carry.reset()
        applied = carry.randomize_layout(rng, ranges)
        carry.place_ball(
            np.array(
                [
                    rng.uniform(-BALL_HALF_X, BALL_HALF_X),
                    rng.uniform(-BALL_HALF_Y, BALL_HALF_Y),
                ]
            )
        )
        observation = carry.observe(pixels=True)
        for name, image in observation["pixels"].items():
            pixels[name].append(image)
        labels.append(encode_handles({s: carry.handle_pos(s) for s in SIDES}))
        layouts.append([applied["x"], applied["y"], applied["yaw"]])

    arrays: dict[str, np.ndarray] = {
        "handles": np.asarray(labels, dtype=np.float32),
        "layouts": np.asarray(layouts, dtype=np.float32),
    }
    for name, frames in pixels.items():
        arrays[f"pixels_{name}"] = np.asarray(frames, dtype=np.uint8)
    return arrays


def main(argv: list[str] | None = None) -> int:
    """Record handle-detection samples over randomized tray layouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("D:/openarm_data/tray_handles"))
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--samples", type=int, default=600, help="samples per file")
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--image-size", type=int, nargs=2, default=(84, 112))
    parser.add_argument(
        "--cameras", nargs="+", default=list(BimanualTrayCarry.VISION_CAMERAS)
    )
    parser.add_argument("--prefix", default="handles")
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
                ranges=TRAIN_LAYOUT,
            )
            path = args.out / f"{args.prefix}_{seed:05d}.npz"
            np.savez_compressed(path, **arrays)
            total += len(arrays["handles"])
            spread = np.abs(arrays["layouts"]).max(axis=0)
            print(
                f"  file {i:3d}  {len(arrays['handles']):5d} samples  "
                f"{path.stat().st_size / 1e6:5.1f} MB  "
                f"{time.perf_counter() - began:5.1f}s  "
                f"max |dx| {spread[0] * 1000:4.0f} mm  |dy| {spread[1] * 1000:4.0f} mm  "
                f"|yaw| {np.degrees(spread[2]):4.1f} deg",
                flush=True,
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
                "sampling": "tray pose uniform over the training layout envelope",
                "layout_ranges": {
                    "x": list(TRAIN_LAYOUT.x),
                    "y": list(TRAIN_LAYOUT.y),
                    "yaw": list(TRAIN_LAYOUT.yaw),
                },
                "view": "arms at the retracted reset pose, before any approach",
            },
            indent=2,
        )
    )
    print(f"\n{total} samples in {time.perf_counter() - started:.0f}s -> {args.out}")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
