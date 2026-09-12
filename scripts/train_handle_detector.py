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

"""Train the handle detector: one pre-grasp frame per camera to both posts.

Validation is split by **file**, not by frame. Every sample in a file is an
independent layout here, so a frame-level split would not leak the way it does
for rollout data -- but files are the unit the recorder varies its seed over, and
holding whole ones out keeps the split honest if the recorder ever gains
correlated samples.

The reported number is the mean distance between the predicted handle position
and the true one, in millimetres. It has a directly meaningful threshold, which
is rare in this project: the jaws open to 85 mm around a 24 mm post, so roughly
30 mm of slack per side, and the approach has to put the pads around the post
without the palm catching the tray. An error much past a centimetre will start
to cost grasps -- but the number that settles it is grasp success under the
detector, not this one. Validation error is not the metric; see STATUS.md on
exactly this trap for the ball estimator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from openarm_gym.policies.handle_vision import (
    HANDLE_SCALE,
    HandleDataset,
    HandleDetector,
    collate,
)
from openarm_gym.policies.tray_vision import augment


@torch.no_grad()
def evaluate(detector: HandleDetector, loader: DataLoader, device: str) -> float:
    """Return the mean handle position error in metres."""
    detector.eval()
    total, count = 0.0, 0
    for images, labels in loader:
        images = {k: v.to(device) for k, v in images.items()}
        predicted = detector(images).cpu().numpy().reshape(-1, 2, 3)
        truth = labels.numpy().reshape(-1, 2, 3)
        total += float(
            np.linalg.norm((predicted - truth) * HANDLE_SCALE, axis=-1).sum()
        )
        count += predicted.shape[0] * 2
    detector.train()
    return total / count


def main(argv: list[str] | None = None) -> int:
    """Train a handle detector and write a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", type=Path, nargs="?", default=Path("D:/openarm_data/tray_handles")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("D:/openarm_data/tray_handles/detector.pt")
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--val-files", type=int, default=1)
    parser.add_argument("--gain-noise", type=float, default=0.04)
    parser.add_argument("--read-noise", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    meta = json.loads((args.dataset / "meta_handles.json").read_text())
    cameras = tuple(meta["cameras"])
    image_size = tuple(meta["image_size"])

    paths = sorted(args.dataset.glob("handles_*.npz"))
    if len(paths) <= args.val_files:
        parser.error(f"only {len(paths)} files; need more than --val-files")
    val_paths = paths[-args.val_files :]
    train_paths = paths[: -args.val_files]

    started = time.perf_counter()
    train_set = HandleDataset(train_paths, cameras)
    val_set = HandleDataset(val_paths, cameras)
    print(
        f"loaded {len(train_set)} train / {len(val_set)} val samples "
        f"in {time.perf_counter() - started:.1f}s"
    )

    # num_workers=0 for the same reason as the ball trainer: Windows spawns
    # rather than forks, so a worker would copy the whole in-RAM dataset.
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=0,
    )

    detector = HandleDetector(cameras, image_size, width=args.width).to(args.device)
    params = sum(p.numel() for p in detector.parameters())
    print(f"detector: {params / 1e6:.2f}M parameters on {args.device}")

    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )

    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        total, batches = 0.0, 0
        for images, labels in train_loader:
            images = {k: v.to(args.device, non_blocking=True) for k, v in images.items()}
            augment(images, args.gain_noise, args.read_noise)
            loss = torch.nn.functional.mse_loss(
                detector(images), labels.to(args.device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss.detach())
            batches += 1

        error = evaluate(detector, val_loader, args.device)
        history.append(
            {"epoch": epoch, "train_mse": total / batches, "val_error_m": error}
        )
        marker = ""
        if error < best:
            best = error
            marker = " *"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": detector.state_dict(),
                    "cameras": list(cameras),
                    "image_size": list(image_size),
                    "width": args.width,
                    "val_error_m": error,
                },
                args.out,
            )
        print(
            f"epoch {epoch:3d}  train mse {total / batches:.5f}  "
            f"val handle err {error * 1000:6.2f} mm  "
            f"{time.perf_counter() - epoch_started:5.1f}s{marker}",
            flush=True,
        )

    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest validation handle error {best * 1000:.2f} mm -> {args.out}")
    print(
        "Validation error is not the metric. Run evaluate_visual_grasp.py: "
        "grasp success and tray displacement over randomized layouts are."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
