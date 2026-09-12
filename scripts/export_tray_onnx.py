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

"""Export the ball estimator and the handle detector to ONNX.

Both networks are tiny -- 0.16 M and 0.08 M parameters -- so the point of this
is not speed. It is that the checkpoints stop being readable only by this
repository's own Python: ONNX runs them from a browser, from C++, or from
anything else that has to drive the same arm later.

Two things travel beside the graph in a JSON sidecar rather than being baked
into it:

* **The normalisation statistics.** Proprioception is standardised with the
  training set's mean and standard deviation, and a consumer that guesses them
  produces plausible-looking output that is quietly wrong. In a file they can be
  checked.
* **The output scales.** The networks emit O(1) numbers; metres and radians come
  from multiplying by ``BALL_SCALE``, ``VELOCITY_SCALE``, ``TILT_SCALE`` and
  ``HANDLE_SCALE``. Same argument.

The export is verified against PyTorch on random input after it is written, at
a tolerance tight enough that a transposed axis or a dropped normalisation shows
up. A silently-wrong export is worse than none: it fails as a control problem,
somewhere else, much later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from openarm_gym.policies.handle_vision import (
    HANDLE_NOMINAL,
    HANDLE_SCALE,
    HandleDetector,
)
from openarm_gym.policies.tray_vision import (
    BALL_SCALE,
    TILT_SCALE,
    VELOCITY_SCALE,
    TrayVisionPolicy,
)

#: Agreement required between PyTorch and onnxruntime, in the networks' own
#: scaled units. The ball head's outputs are O(1) and BALL_SCALE is 0.1 m, so
#: 1e-4 here is 10 micrometres of ball position -- far below the 4.7 mm the
#: closed loop actually achieves, and far above float32 round-off.
TOLERANCE = 1e-4


class _BallInference(torch.nn.Module):
    """The ball policy with positional inputs, since ONNX has no dicts.

    Cameras are passed in the order the checkpoint lists them, and that order is
    written into the sidecar -- a consumer feeding the two cameras the other way
    round gets a confident, wrong answer otherwise.
    """

    def __init__(self, policy: TrayVisionPolicy) -> None:
        """Wrap ``policy`` so ``forward`` takes one tensor per camera."""
        super().__init__()
        self.policy = policy
        self.cameras = tuple(policy.cameras)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tilt, ball state)`` from images then proprioception."""
        images = dict(zip(self.cameras, args[: len(self.cameras)]))
        return self.policy(images, args[-1])


class _HandleInference(torch.nn.Module):
    """The handle detector with positional inputs, one tensor per camera."""

    def __init__(self, detector: HandleDetector) -> None:
        """Wrap ``detector`` so ``forward`` takes one tensor per camera."""
        super().__init__()
        self.detector = detector
        self.cameras = tuple(detector.cameras)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Return the six scaled handle offsets."""
        return self.detector(dict(zip(self.cameras, args)))


def _check(module: torch.nn.Module, path: Path, inputs: list[torch.Tensor]) -> float:
    """Run the exported graph against PyTorch and return the worst difference.

    On **random** input, not on the zeros used to trace the graph. A network fed
    zeros is dominated by its biases, so a dropped normalisation or a transposed
    spatial axis can pass a zero-input comparison and fail on a real image.
    """
    import onnxruntime

    generator = torch.Generator().manual_seed(0)
    probe = [
        torch.rand(t.shape, generator=generator) - 0.5 for t in inputs
    ]
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    fed = {i.name: t.numpy() for i, t in zip(session.get_inputs(), probe)}
    produced = session.run(None, fed)
    with torch.no_grad():
        expected = module(*probe)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    return max(
        float(np.abs(a - b.numpy()).max()) for a, b in zip(produced, expected)
    )


def export_ball(checkpoint: Path, out_dir: Path) -> dict:
    """Export the ball estimator, returning its sidecar metadata."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = TrayVisionPolicy(
        tuple(blob["cameras"]),
        tuple(blob["image_size"]),
        int(blob["proprio_dim"]),
        width=int(blob["width"]),
        frame_stack=int(blob["frame_stack"]),
    )
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    module = _BallInference(policy)

    height, width = blob["image_size"]
    channels = 3 * int(blob["frame_stack"])
    inputs = [torch.zeros(1, channels, height, width) for _ in blob["cameras"]]
    inputs.append(torch.zeros(1, int(blob["proprio_dim"])))
    names = [f"image_{name}" for name in blob["cameras"]] + ["proprio"]

    path = out_dir / "tray_ball_estimator.onnx"
    torch.onnx.export(
        module,
        tuple(inputs),
        str(path),
        input_names=names,
        output_names=["tilt", "ball_state"],
        dynamic_axes={name: {0: "batch"} for name in [*names, "tilt", "ball_state"]},
        opset_version=17,
        # The dynamo exporter cannot translate dynamic_axes for a module whose
        # forward takes *args, which this one does because ONNX has no dicts and
        # the camera count is not fixed. The legacy tracer handles it.
        dynamo=False,
    )
    worst = _check(module, path, inputs)
    return {
        "file": path.name,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "cameras": list(blob["cameras"]),
        "image_size": list(blob["image_size"]),
        "frame_stack": int(blob["frame_stack"]),
        "frame_stride": int(blob["frame_stride"]),
        "inputs": {
            "images": "one per camera, (B, 3*frame_stack, H, W), float32 in [-0.5, 0.5], "
            "oldest frame first, RGB",
            "proprio": "(B, proprio_dim), standardised with proprio_mean/proprio_std",
        },
        "outputs": {
            "tilt": "(B, 2) roll, pitch — multiply by tilt_scale for radians",
            "ball_state": "(B, 4) x, y, vx, vy in the tray frame — multiply the "
            "first two by ball_scale for metres and the last two by velocity_scale "
            "for m/s",
        },
        "proprio_mean": [float(v) for v in np.asarray(blob["proprio_mean"]).ravel()],
        "proprio_std": [float(v) for v in np.asarray(blob["proprio_std"]).ravel()],
        "tilt_scale": TILT_SCALE,
        "ball_scale": BALL_SCALE,
        "velocity_scale": VELOCITY_SCALE,
        "onnx_vs_torch_max_abs": worst,
    }


def export_handles(checkpoint: Path, out_dir: Path) -> dict:
    """Export the handle detector, returning its sidecar metadata."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    detector = HandleDetector(
        tuple(blob["cameras"]), tuple(blob["image_size"]), width=int(blob["width"])
    )
    detector.load_state_dict(blob["state_dict"])
    detector.eval()
    module = _HandleInference(detector)

    height, width = blob["image_size"]
    inputs = [torch.zeros(1, 3, height, width) for _ in blob["cameras"]]
    names = [f"image_{name}" for name in blob["cameras"]]

    path = out_dir / "tray_handle_detector.onnx"
    torch.onnx.export(
        module,
        tuple(inputs),
        str(path),
        input_names=names,
        output_names=["handle_offsets"],
        dynamic_axes={name: {0: "batch"} for name in [*names, "handle_offsets"]},
        opset_version=17,
        dynamo=False,
    )
    worst = _check(module, path, inputs)
    return {
        "file": path.name,
        "parameters": sum(p.numel() for p in detector.parameters()),
        "cameras": list(blob["cameras"]),
        "image_size": list(blob["image_size"]),
        "inputs": {
            "images": "one per camera, (B, 3, H, W), float32 in [-0.5, 0.5], RGB, "
            "taken with the arms at the retracted pose before any approach"
        },
        "outputs": {
            "handle_offsets": "(B, 6) left xyz then right xyz — multiply by "
            "handle_scale and add handle_nominal for world metres"
        },
        "handle_scale": HANDLE_SCALE,
        "handle_nominal": HANDLE_NOMINAL.tolist(),
        "validation_error_m": float(blob.get("val_error_m", float("nan"))),
        "onnx_vs_torch_max_abs": worst,
    }


def main(argv: list[str] | None = None) -> int:
    """Export both networks and write the sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--out", type=Path, default=Path("export"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ball_estimator": export_ball(args.policy, args.out),
        "handle_detector": export_handles(args.handle_detector, args.out),
        "note": "Images are RGB uint8 divided by 255 and shifted by -0.5. "
        "The ball estimator's velocity output was measured to be uninformative "
        "at this resolution; the controller differences the position estimate "
        "instead. See STATUS.md.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for name, entry in manifest.items():
        if not isinstance(entry, dict):
            continue
        worst = entry["onnx_vs_torch_max_abs"]
        verdict = "ok" if worst < TOLERANCE else "MISMATCH"
        print(
            f"{name:<16} {entry['parameters'] / 1e6:.2f}M params  "
            f"-> {entry['file']}  onnx vs torch {worst:.2e}  {verdict}"
        )
        if worst >= TOLERANCE:
            return 1
    print(f"\nwrote {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
