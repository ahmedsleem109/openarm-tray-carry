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

"""Behaviour-clone the classical tray balancer from camera images.

Trains both of the policy's heads at once -- the ball's tray-frame state, which is
what closes the loop in the default ``estimator`` control mode, and the expert's
tilt setpoint, kept as the end-to-end baseline it is compared against. The split
is by **episode**, not by frame:
consecutive frames of one episode are nearly identical, so a frame-level split
would put near-duplicates of every validation sample in the training set and
report a validation error that means nothing.

Camera noise is applied here rather than baked into the recording, so the same
dataset trains with and without the sensor model.
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

from openarm_gym.policies.tray_vision import (
    BALL_SCALE,
    FRAME_STACK,
    FRAME_STRIDE,
    TILT_SCALE,
    VELOCITY_SCALE,
    TrayVisionDataset,
    TrayVisionPolicy,
    augment,
    collate,
)


@torch.no_grad()
def evaluate(
    policy: TrayVisionPolicy, loader: DataLoader, device: str
) -> tuple[float, float, float]:
    """Return ``(tilt RMSE rad, ball position RMSE m, ball velocity RMSE m/s)``.

    Position and velocity are reported apart because they fail for different
    reasons and cost different things: a position error biases the P term, while a
    velocity error -- which comes from differencing two frames 40 ms apart -- goes
    straight into the damping.
    """
    policy.eval()
    tilt_sq, pos_sq, vel_sq, count = 0.0, 0.0, 0.0, 0
    for images, proprio, tilt, ball in loader:
        images = {k: v.to(device) for k, v in images.items()}
        pred_tilt, pred_state = policy(images, proprio.to(device))
        error = (pred_state.cpu() - ball) ** 2
        tilt_sq += float(((pred_tilt.cpu() - tilt) ** 2).sum())
        pos_sq += float(error[:, :2].sum())
        vel_sq += float(error[:, 2:].sum())
        count += len(tilt)
    policy.train()
    return (
        float(np.sqrt(tilt_sq / (2 * count)) * TILT_SCALE),
        float(np.sqrt(pos_sq / (2 * count)) * BALL_SCALE),
        float(np.sqrt(vel_sq / (2 * count)) * VELOCITY_SCALE),
    )


def main(argv: list[str] | None = None) -> int:
    """Train a tray-tilt vision policy and write a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("D:/openarm_data/tray_vision"))
    parser.add_argument("--out", type=Path, default=Path("D:/openarm_data/tray_vision/policy.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=1.0,
        help="weight on the ball-position head; 0 disables it",
    )
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--val-episodes", type=int, default=4)
    parser.add_argument("--gain-noise", type=float, default=0.04)
    parser.add_argument("--read-noise", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    meta = json.loads((args.dataset / "meta.json").read_text())
    cameras = tuple(meta["cameras"])
    image_size = tuple(meta["image_size"])

    # Both the demonstrations and any DAgger rounds beside them.
    paths = sorted(args.dataset.glob("*.npz"))
    if len(paths) <= args.val_episodes:
        parser.error(f"only {len(paths)} episodes; need more than --val-episodes")
    # Hold out whole episodes, spread across the prefixes rather than taken from
    # one end: with a DAgger round present, the tail of the sorted list is all
    # DAgger, and validating only on that measures a different distribution from
    # the one being trained on.
    val_paths = paths[:: max(1, len(paths) // args.val_episodes)][: args.val_episodes]
    train_paths = [p for p in paths if p not in set(val_paths)]

    started = time.perf_counter()
    train_set = TrayVisionDataset(args.dataset, cameras, episodes=train_paths)
    val_set = TrayVisionDataset(args.dataset, cameras, episodes=val_paths)
    # Validation must be normalised with the training statistics, or the two
    # halves are measured in different units.
    val_set.proprio_mean = train_set.proprio_mean
    val_set.proprio_std = train_set.proprio_std
    print(
        f"loaded {len(train_paths)} train / {len(val_paths)} val episodes "
        f"({len(train_set)} / {len(val_set)} samples) in {time.perf_counter() - started:.1f}s"
    )

    # num_workers=0 on purpose: Windows spawns rather than forks, so every worker
    # would pickle a copy of the in-RAM uint8 dataset, and RAM is the binding
    # constraint on this machine.
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=0,
    )

    proprio_dim = train_set.proprio[0].shape[1]
    policy = TrayVisionPolicy(
        cameras, image_size, proprio_dim, width=args.width
    ).to(args.device)
    params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {params / 1e6:.2f}M parameters on {args.device}")

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader)
    )

    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        totals = np.zeros(2)
        batches = 0
        for images, proprio, tilt, ball in train_loader:
            images = {k: v.to(args.device, non_blocking=True) for k, v in images.items()}
            augment(images, args.gain_noise, args.read_noise)
            pred_tilt, pred_ball = policy(images, proprio.to(args.device))
            tilt_loss = torch.nn.functional.mse_loss(pred_tilt, tilt.to(args.device))
            ball_loss = torch.nn.functional.mse_loss(pred_ball, ball.to(args.device))
            loss = tilt_loss + args.aux_weight * ball_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            totals += [float(tilt_loss.detach()), float(ball_loss.detach())]
            batches += 1

        tilt_rmse, pos_rmse, vel_rmse = evaluate(policy, val_loader, args.device)
        history.append(
            {
                "epoch": epoch,
                "train_tilt_mse": totals[0] / batches,
                "train_ball_mse": totals[1] / batches,
                "val_tilt_rmse_rad": tilt_rmse,
                "val_ball_rmse_m": pos_rmse,
                "val_velocity_rmse_ms": vel_rmse,
            }
        )
        # Selected on the ball position error, not the tilt error: the estimator
        # mode is what closes the loop, and its accuracy is what survival depends
        # on. Selecting on tilt would pick the checkpoint best at the head that
        # was measured to fail closed loop.
        marker = ""
        if pos_rmse < best:
            best = pos_rmse
            marker = " *"
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": policy.state_dict(),
                    "cameras": list(cameras),
                    "image_size": list(image_size),
                    "proprio_dim": proprio_dim,
                    "width": args.width,
                    "frame_stack": FRAME_STACK,
                    "frame_stride": FRAME_STRIDE,
                    "proprio_mean": train_set.proprio_mean,
                    "proprio_std": train_set.proprio_std,
                    "val_tilt_rmse_rad": tilt_rmse,
                    "val_ball_rmse_m": pos_rmse,
                    "val_velocity_rmse_ms": vel_rmse,
                    "args": vars(args) | {"dataset": str(args.dataset), "out": str(args.out)},
                },
                args.out,
            )
        print(
            f"  epoch {epoch:3d}  train tilt {totals[0] / batches:.4f}  "
            f"state {totals[1] / batches:.4f}  |  val tilt {tilt_rmse * 1000:6.2f} mrad  "
            f"ball {pos_rmse * 1000:5.2f} mm  vel {vel_rmse * 1000:6.1f} mm/s  "
            f"({time.perf_counter() - epoch_started:.0f}s){marker}"
        )

    (args.out.with_suffix(".history.json")).write_text(json.dumps(history, indent=2))
    print(f"\nbest val ball-position RMSE {best * 1000:.2f} mm -> {args.out}")
    print(
        f"For scale: the balance law turns a ball-position error of e into a tilt "
        f"error of kp*e/g, so 1 mm of sight error is about "
        f"{12.0 / 9.81:.1f} mrad of tilt. Closed-loop survival is the real test "
        "-- run scripts/evaluate_tray_vision.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
