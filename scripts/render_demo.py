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

"""Render the demo video: the same carry with the controller off and on.

`render_tray.py` renders one rollout, which is the right tool for looking at a
change. It is the wrong tool for showing someone what this project does: a
single clip of two arms holding a tray shows neither what the control is for nor
what the policy can actually see, and a viewer has no way to tell a working
system from a lucky one.

So this renders **two runs of the same plan side by side** -- identical layout,
identical grasp, identical disturbance, differing only in whether the balancer is
running -- and draws the numbers over them:

* the ball's offset from the tray centre, which is the quantity being controlled;
* the tilt being commanded, which is the control input;
* the grip force at each hand, because the tray is held by friction and form
  closure rather than welded, and that is the failure the whole task rests on;
* the policy's own camera images, at the 84x112 they are actually given, with its
  ball *estimate* drawn against the truth -- the point being that the right-hand
  arm is flying on those two small pictures and nothing else.

The grasp is included rather than skipped, because "the arms found the handles
from the cameras and closed on them" is half of what is worth showing, and it
happens inside `grasp()` where the rollout driver cannot see it. That is what
`BimanualTrayCarry.on_step` is for.

Everything is run under a named sim-to-real condition, and the condition is
printed on the video, so nobody has to take on trust which of the realism knobs
were turned on.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openarm_gym.assets import scene_path
from PIL import Image, ImageDraw, ImageFont

from openarm_gym.control.bimanual_tray import BimanualTrayCarry
from openarm_gym.control.tray_eval import CONDITIONS, build_carry
from openarm_gym.control.tray_task import random_carry, run_plan
from openarm_gym.policies.handle_vision import SIDES, load_handle_detector
from openarm_gym.policies.tray_vision import load_controller
from openarm_gym.vision import CameraRig, write_mp4


#: Composition geometry. One panel per run, side by side, with a readout strip
#: under each and a wider strip along the bottom for what the policy sees.
CANVAS = (1280, 920)
PANEL = (600, 450)
#: Top of the video panels. The header above them carries three lines -- what
#: this is, what is being compared, and which realism knobs are on -- and the
#: bottom band has to fit two 84x112 thumbnails at 2x plus their captions.
PANEL_Y = 104
GAP = 16

#: Tray half-extents, metres, for the schematic and for the offset gauge.
TRAY_HALF = (0.075, 0.15)

WHITE = (235, 238, 242)
DIM = (150, 158, 168)
GOOD = (90, 210, 140)
WARN = (250, 190, 90)
BAD = (245, 110, 110)
BACKDROP = (18, 20, 26)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a UI font, falling back to PIL's bitmap font if none is installed."""
    for name in (("seguisb.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Take:
    """One run's frames and the per-step numbers drawn over them."""

    label: str
    sublabel: str
    frames: list[np.ndarray] = field(default_factory=list)
    #: Ball offset in the tray frame, metres, per recorded frame.
    ball: list[np.ndarray] = field(default_factory=list)
    #: Commanded ``(roll, pitch)`` in radians, per recorded frame.
    tilt: list[np.ndarray] = field(default_factory=list)
    #: Minimum of the two hands' normal force, newtons.
    grip: list[float] = field(default_factory=list)
    #: The policy's ball estimate, or ``None`` when nothing is estimating.
    estimate: list[np.ndarray | None] = field(default_factory=list)
    #: The images the policy was given, per camera, or ``None``.
    pixels: list[dict[str, np.ndarray] | None] = field(default_factory=list)
    #: True once the jaws have closed and the carry has started.
    carrying: list[bool] = field(default_factory=list)
    #: Carry step index, so disturbances can be flashed at the right moment.
    step: list[int] = field(default_factory=list)
    on_tray: list[bool] = field(default_factory=list)
    #: How far the handle detector's estimate was from the posts' true
    #: positions, metres. Recorded once, before the approach starts.
    handle_error: float | None = None


class Driver:
    """A tilt policy that remembers what it last did, for the overlay.

    Wrapping rather than reaching into the controller keeps the recorded numbers
    the ones that were actually commanded -- including for the ``level`` run,
    where the commanded tilt is zero by definition and saying so on screen is the
    entire point of the comparison.
    """

    def __init__(self, controller=None, control_hz: float = 50.0) -> None:
        """Wrap ``controller``; ``None`` means never tilt."""
        self.controller = controller
        self.control_hz = control_hz
        self.setpoint = np.zeros(2)
        self.estimate: np.ndarray | None = None
        self.pixels: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        """Clear the wrapped controller's frame history."""
        if self.controller is not None:
            self.controller.reset()
        self.setpoint = np.zeros(2)
        self.estimate = None

    def __call__(self, obs: dict) -> np.ndarray:
        """Return the setpoint, keeping a copy of it and of what produced it."""
        self.pixels = obs.get("pixels")
        if self.controller is None:
            self.setpoint = np.zeros(2)
        else:
            self.setpoint = np.asarray(self.controller(obs), dtype=np.float64)
            self.estimate = np.asarray(self.controller.ball_estimate, dtype=np.float64)
        return self.setpoint


class Recorder:
    """Capture a frame and the current numbers on every control step.

    The grasp runs at the same control rate as the carry but takes several
    seconds of simulated time, so it is subsampled -- otherwise two thirds of the
    video is the arms reaching in.
    """

    def __init__(self, take: Take, rig: CameraRig, camera: str, driver: Driver) -> None:
        """Record into ``take``, rendering ``camera`` from ``rig``."""
        self.take = take
        self.rig = rig
        self.camera = camera
        self.driver = driver
        self.grasp_stride = 6
        self.calls = 0
        self.carry_step = 0

    def __call__(self, carry: BimanualTrayCarry) -> None:
        """Append one frame, or skip it while subsampling the grasp."""
        self.calls += 1
        carrying = carry.grasped
        if carrying:
            self.carry_step += 1
        elif self.calls % self.grasp_stride:
            return
        rel, _ = carry.ball_in_tray()
        report = carry.grip_report()
        if not carrying:
            # The policy has not been called yet, so there is nothing to show in
            # the thumbnails. Render them here instead -- one extra render on one
            # frame in six, which is what the grasp is subsampled to anyway.
            self.driver.pixels = carry.observe(pixels=True)["pixels"]
        self.take.frames.append(self.rig.render(carry.data, self.camera))
        self.take.ball.append(rel[:2].copy())
        self.take.tilt.append(self.driver.setpoint.copy())
        self.take.grip.append(min(report[s]["force"] for s in SIDES))
        self.take.estimate.append(
            None if self.driver.estimate is None else self.driver.estimate.copy()
        )
        self.take.pixels.append(self.driver.pixels)
        self.take.carrying.append(carrying)
        self.take.step.append(self.carry_step - 1 if carrying else -1)
        self.take.on_tray.append(carry.ball_on_tray())


def run_take(
    take: Take,
    *,
    condition,
    plan,
    checkpoint: Path,
    detector: Path | None,
    balancing: bool,
    seed: int,
    camera: str,
    device: str,
) -> None:
    """Fly one run of ``plan``, recording frames and numbers into ``take``."""
    policy = load_controller(checkpoint, device=device, control_hz=50.0)
    carry = build_carry(
        scene_path(),
        condition,
        camera_names=tuple(policy.policy.cameras),
        image_size=tuple(policy.policy.image_size),
    )
    driver = Driver(policy if balancing else None, control_hz=carry.control_hz)
    rig = CameraRig(carry.model, (camera,), image_size=(PANEL[1], PANEL[0]))
    carry.on_step = Recorder(take, rig, camera, driver)

    handles = None
    if detector is not None:
        estimator = load_handle_detector(detector, device=device)

        def handles(c: BimanualTrayCarry) -> dict:
            """Locate both posts from the standoff view, and score the estimate."""
            found = estimator(c.observe(pixels=True))
            take.handle_error = estimator.error({s: c.handle_pos(s) for s in SIDES})
            return found

    layout_rng = np.random.default_rng(seed).spawn(1)[0]

    def on_reset(c: BimanualTrayCarry) -> None:
        """Jitter the tray's starting pose, before the jaws close on it."""
        c.randomize_layout(layout_rng)

    driver.reset()
    run_plan(
        carry,
        plan,
        tilt_policy=driver,
        rng=np.random.default_rng(seed) if condition.noisy_pixels else None,
        on_reset=on_reset,
        handle_source=handles,
        # The failure is the thing worth showing, so the run does not stop when
        # the ball goes over the edge -- the arms carry on with an empty tray,
        # which is exactly what it looks like when the control is not there.
        stop_on_loss=False,
    )
    rig.close()
    carry.close()


def _panel(draw: ImageDraw.ImageDraw, take: Take, i: int, x: int, disturbances: set[int]) -> None:
    """Draw one run's label, status line and numbers beside its video."""
    big, small, mono = _font(23, bold=True), _font(16), _font(19)
    # The label sits *inside* the panel, over the empty sky at the top of the
    # render: a header tall enough to hold it above the video would push the
    # thumbnails off the bottom of a 16:9 canvas.
    draw.rectangle([x, PANEL_Y, x + PANEL[0], PANEL_Y + 52], fill=(12, 14, 18))
    draw.text((x + 12, PANEL_Y + 4), take.label, font=big, fill=WHITE)
    draw.text((x + 12, PANEL_Y + 30), take.sublabel, font=small, fill=DIM)
    # A border that goes red the moment the ball is off, so the outcome reads at
    # a glance and from a thumbnail.
    edge = GOOD if take.on_tray[i] else BAD
    draw.rectangle(
        [x - 2, PANEL_Y - 2, x + PANEL[0] + 2, PANEL_Y + PANEL[1] + 2], outline=edge, width=3
    )

    offset = float(np.linalg.norm(take.ball[i])) if take.on_tray[i] else float("nan")
    y = PANEL_Y + PANEL[1] + 14
    if not take.carrying[i]:
        found = (
            ""
            if take.handle_error is None
            else f"   ({take.handle_error * 1000:.1f} mm from the true posts)"
        )
        draw.text(
            (x, y),
            f"grasping — posts located from the cameras{found}",
            font=mono,
            fill=WARN,
        )
    elif not take.on_tray[i]:
        draw.text((x, y), "BALL LOST", font=_font(22, bold=True), fill=BAD)
    else:
        colour = GOOD if offset < 0.05 else WARN
        draw.text((x, y), f"ball off centre   {offset * 1000:5.0f} mm", font=mono, fill=colour)
        # A bar is easier to read at a glance than the number it repeats: full
        # width is the edge of the tray, which is where the ball is lost.
        width = int(min(1.0, offset / TRAY_HALF[0]) * 260)
        draw.rectangle([x + 250, y + 4, x + 250 + 260, y + 18], outline=(60, 66, 76))
        draw.rectangle([x + 250, y + 4, x + 250 + max(2, width), y + 18], fill=colour)

    roll, pitch = np.degrees(take.tilt[i])
    draw.text(
        (x, y + 28),
        f"commanded tilt    roll {roll:+5.1f}°   pitch {pitch:+5.1f}°",
        font=mono,
        fill=WHITE if take.carrying[i] else DIM,
    )
    draw.text(
        (x, y + 52),
        f"grip, weaker hand {take.grip[i]:5.1f} N   (no weld — friction only)",
        font=mono,
        fill=WHITE,
    )
    if take.carrying[i] and take.step[i] in disturbances:
        draw.text((x + 250, PANEL_Y + 8), "◀ BALL SHOVED", font=_font(24, bold=True), fill=BAD)


def _schematic(image: Image.Image, take: Take, i: int, box: tuple[int, int, int, int]) -> None:
    """Draw the tray from above, with the true ball and the policy's estimate."""
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(70, 78, 90), width=2)

    def place(point: np.ndarray) -> tuple[float, float]:
        """Tray-frame metres to pixels inside the box. y is the long axis."""
        u = (point[1] / TRAY_HALF[1] + 1.0) / 2.0
        v = (point[0] / TRAY_HALF[0] + 1.0) / 2.0
        return x0 + u * (x1 - x0), y0 + v * (y1 - y0)

    cx, cy = place(np.zeros(2))
    draw.line([x0, cy, x1, cy], fill=(50, 56, 66))
    draw.line([cx, y0, cx, y1], fill=(50, 56, 66))
    if take.on_tray[i]:
        bx, by = place(take.ball[i])
        draw.ellipse([bx - 7, by - 7, bx + 7, by + 7], fill=(245, 130, 40))
    estimate = take.estimate[i]
    if estimate is not None:
        ex, ey = place(estimate)
        draw.line([ex - 9, ey, ex + 9, ey], fill=(120, 200, 255), width=2)
        draw.line([ex, ey - 9, ex, ey + 9], fill=(120, 200, 255), width=2)


def compose(takes: tuple[Take, Take], plan, condition, fps: int) -> list[np.ndarray]:
    """Lay the two runs out side by side with their readouts and thumbnails."""
    left, right = takes
    count = min(len(left.frames), len(right.frames))
    disturbances = set(plan.disturbances)
    title, small, tiny = _font(27, bold=True), _font(17), _font(15)
    frames = []
    for i in range(count):
        canvas = Image.new("RGB", CANVAS, BACKDROP)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (GAP + 8, 12),
            "Two 7-DOF arms carry a tray by its handles and keep a ball on it",
            font=title,
            fill=WHITE,
        )
        draw.text(
            (GAP + 8, 44),
            "same plan, same tray layout, same shove — the only difference is the control",
            font=small,
            fill=DIM,
        )
        draw.text(
            (GAP + 8, 66),
            f"simulated hardware: {condition.summary()}",
            font=tiny,
            fill=(120, 150, 190),
        )
        draw.line([GAP, 92, CANVAS[0] - GAP, 92], fill=(44, 50, 60))
        for take, x in ((left, GAP), (right, GAP + PANEL[0] + 2 * GAP)):
            canvas.paste(Image.fromarray(take.frames[i]), (x, PANEL_Y))
            _panel(draw, take, i, x, disturbances)

        # What the right-hand run is actually flying on.
        base = PANEL_Y + PANEL[1] + 112
        draw.line([GAP, base - 20, CANVAS[0] - GAP, base - 20], fill=(44, 50, 60))
        draw.text(
            (GAP + 8, base),
            "the vision run, on the right:  everything it is given, and what it infers",
            font=small,
            fill=(120, 200, 255),
        )
        pixels = right.pixels[i] or {}
        for j, (name, image) in enumerate(sorted(pixels.items())):
            thumb = Image.fromarray(image).resize((224, 168), Image.NEAREST)
            canvas.paste(thumb, (GAP + 8 + j * 236, base + 24))
            draw.text((GAP + 8 + j * 236, base + 194), name, font=tiny, fill=DIM)
        draw.text(
            (GAP + 8, base + 212),
            "two 84x112 images + joint angles — the ball's position is never given to it",
            font=tiny,
            fill=DIM,
        )

        panel_x = GAP + PANEL[0] + 2 * GAP
        draw.text((panel_x, base), "the tray, seen from above", font=small, fill=DIM)
        _schematic(canvas, right, i, (panel_x, base + 24, panel_x + 300, base + 174))
        # Drawn rather than written: the cross glyph is not in every UI font,
        # and a missing-glyph box in the legend for the key symbol is worse than
        # no legend at all.
        draw.ellipse(
            [panel_x + 318, base + 44, panel_x + 328, base + 54], fill=(245, 130, 40)
        )
        draw.text((panel_x + 338, base + 40), "where the ball is", font=tiny, fill=DIM)
        draw.line(
            [panel_x + 316, base + 71, panel_x + 330, base + 71], fill=(120, 200, 255), width=2
        )
        draw.line(
            [panel_x + 323, base + 64, panel_x + 323, base + 78], fill=(120, 200, 255), width=2
        )
        draw.text((panel_x + 338, base + 62), "where vision thinks", font=tiny, fill=DIM)
        estimate = right.estimate[i]
        if estimate is not None and right.on_tray[i]:
            error = float(np.linalg.norm(estimate - right.ball[i]))
            draw.text(
                (panel_x + 316, base + 92),
                f"sight error {error * 1000:4.1f} mm",
                font=small,
                fill=WHITE,
            )
        draw.text(
            (panel_x + 316, base + 126),
            f"{i / fps:4.1f} s",
            font=small,
            fill=DIM,
        )
        frames.append(np.asarray(canvas))
    return frames


def main(argv: list[str] | None = None) -> int:
    """Render the side-by-side demo to an MP4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("../results/tray_demo.mp4"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("D:/openarm_data/tray_vision_layout/policy.pt"),
    )
    parser.add_argument(
        "--handle-detector",
        type=Path,
        default=Path("D:/openarm_data/tray_handles/detector.pt"),
    )
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--camera", default="balancecam")
    parser.add_argument("--condition", default="all", choices=sorted(CONDITIONS))
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    condition = CONDITIONS[args.condition]
    plan = random_carry(np.random.default_rng(args.seed))
    takes = (
        Take("CONTROL OFF", "the tray is carried level, nothing corrects the ball"),
        Take("VISION CONTROL ON", "the tray tilts to hold the ball, from camera images"),
    )
    for take, balancing in zip(takes, (False, True)):
        run_take(
            take,
            condition=condition,
            plan=plan,
            checkpoint=args.checkpoint,
            detector=args.handle_detector,
            balancing=balancing,
            seed=args.seed,
            camera=args.camera,
            device=args.device,
        )
        kept = take.on_tray[-1]
        print(
            f"{take.label:<20} {len(take.frames):4d} frames, "
            f"ball {'kept' if kept else 'lost'}",
            flush=True,
        )

    frames = compose(takes, plan, condition, args.fps)
    path = write_mp4(args.out, frames, fps=args.fps)
    print(
        f"{len(frames)} frames -> {path} "
        f"({path.stat().st_size / 1e6:.1f} MB, {len(frames) / args.fps:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
