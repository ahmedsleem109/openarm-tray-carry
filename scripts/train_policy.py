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

"""Train an action-chunking policy on recorded OpenArm demonstrations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from openarm_gym.policies.act import ACTPolicy, act_loss
from openarm_gym.policies.cache import CachedChunkDataset
from openarm_gym.policies.data import Normalizer, normalize_images

NORMALIZED_KEYS = ("observation.state", "observation.torque", "action")


def main(argv: list[str] | None = None) -> int:
    """Train a policy and write a checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--repo-id", default="openarm/local")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    # Throughput is launch-overhead bound at small batches: 39 samples/s at
    # batch 8 against 334 at batch 64, for 1.06 GB of VRAM.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-torque",
        dest="use_torque",
        action="store_true",
        default=True,
        help="feed joint torque alongside joint position (default)",
    )
    parser.add_argument(
        "--no-torque",
        dest="use_torque",
        action="store_false",
        help="the ablation arm: vision and joint position only",
    )
    # The cache serves tensors straight from RAM, so workers only add
    # copies and memory pressure.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CachedChunkDataset(args.dataset, args.repo_id, args.chunk_size)
    cameras = dataset.camera_keys
    print(
        f"dataset: {dataset.num_episodes} episodes, {len(dataset)} frames, "
        f"cameras {[c.rsplit('.', 1)[-1] for c in cameras]}"
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    normalizer = Normalizer(dataset.stats, NORMALIZED_KEYS, device)
    state_dim = len(dataset.stats["observation.state"]["mean"])

    policy = ACTPolicy(
        state_dim=state_dim,
        action_dim=state_dim,
        n_cameras=len(cameras),
        chunk_size=args.chunk_size,
        use_torque=args.use_torque,
    ).to(device)
    parameters = sum(p.numel() for p in policy.parameters())
    print(
        f"policy: {parameters / 1e6:.1f}M parameters, "
        f"torque {'ON' if args.use_torque else 'OFF'}, device {device}"
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    policy.train()
    step = 0
    running: dict[str, float] = {}
    started = time.perf_counter()

    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break

            images = normalize_images(
                batch["images"].to(device, non_blocking=True).float().div_(255.0)
            )
            state = normalizer.normalize(
                "observation.state", batch["observation.state"].to(device)
            )
            torque = normalizer.normalize(
                "observation.torque", batch["observation.torque"].to(device)
            )
            actions = normalizer.normalize(
                "action", batch["action"].to(device)
            )

            with torch.amp.autocast(device, enabled=device == "cuda"):
                predicted, mu, logvar = policy(images, state, torque, actions)
                loss, parts = act_loss(
                    predicted, actions, mu, logvar, kl_weight=args.kl_weight
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()

            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            step += 1

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - started
                means = {k: v / args.log_every for k, v in running.items()}
                print(
                    f"  step {step:6d}/{args.steps}  "
                    f"l1 {means['l1']:.4f}  kl {means['kl']:.4f}  "
                    f"{step / elapsed:.1f} steps/s",
                    flush=True,
                )
                running = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "config": {
                "state_dim": state_dim,
                "action_dim": state_dim,
                "n_cameras": len(cameras),
                "chunk_size": args.chunk_size,
                "use_torque": args.use_torque,
            },
            "cameras": cameras,
            "stats": {
                key: {
                    "mean": normalizer.mean[key].cpu().tolist(),
                    "std": normalizer.std[key].cpu().tolist(),
                }
                for key in NORMALIZED_KEYS
            },
        },
        args.out,
    )
    elapsed = time.perf_counter() - started
    print(f"\nsaved {args.out}  ({elapsed / 60:.1f} min)")
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"steps": args.steps, "use_torque": args.use_torque}, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
