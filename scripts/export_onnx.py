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

"""Export a trained policy to ONNX, for running in the browser.

Only the inference path is exported: the CVAE encoder is training-only, and at
inference the latent is fixed at the prior mean, so the exported graph takes
images and proprioception and returns an action chunk.

The normalisation statistics travel alongside as JSON rather than being baked
in, so the page can show them and so a mismatch is visible rather than silent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from openarm_gym.policies.act import ACTPolicy


class _InferenceWrapper(torch.nn.Module):
    """Expose the policy's inference path as a plain two-input module."""

    def __init__(self, policy: ACTPolicy) -> None:
        """Wrap a trained policy."""
        super().__init__()
        self.policy = policy

    def forward(self, images: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Return the predicted action chunk.

        ``proprio`` is the already-concatenated state (and torque, when the
        policy uses it), matching the width the policy was trained with.
        """
        batch = proprio.shape[0]
        latent = torch.zeros(
            batch, self.policy.latent_dim, device=proprio.device, dtype=proprio.dtype
        )
        memory = torch.cat(
            [
                self.policy.vision(images),
                self.policy.proprio_embed(proprio).unsqueeze(1),
                self.policy.latent_embed(latent).unsqueeze(1),
            ],
            dim=1,
        )
        memory = self.policy.encoder(memory)
        queries = self.policy.query_embed.expand(batch, -1, -1)
        return self.policy.head(self.policy.decoder(queries, memory))


def main(argv: list[str] | None = None) -> int:
    """Export a checkpoint to ONNX plus a sidecar metadata file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    policy = ACTPolicy(**config, pretrained_vision=False)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()

    wrapper = _InferenceWrapper(policy).eval()
    images = torch.zeros(1, config["n_cameras"], 3, args.height, args.width)
    proprio = torch.zeros(1, policy.proprio_dim)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (images, proprio),
        str(args.out),
        input_names=["images", "proprio"],
        output_names=["action_chunk"],
        dynamic_axes={"images": {0: "batch"}, "proprio": {0: "batch"}},
        opset_version=args.opset,
    )

    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "cameras": payload["cameras"],
                "chunk_size": config["chunk_size"],
                "use_torque": config["use_torque"],
                "image_size": [args.height, args.width],
                "stats": payload["stats"],
                "imagenet_mean": [0.485, 0.456, 0.406],
                "imagenet_std": [0.229, 0.224, 0.225],
            },
            indent=2,
        )
    )

    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB) and {sidecar.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
