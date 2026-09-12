// The JS port has to reproduce the Python controller, not merely run. These are
// the numbers openarm_gym measured: the grasp shoves the tray ~6 mm and holds it
// at ~42 N per hand, and the balancer keeps the ball through a carry that the
// level tray loses it on.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { DOMParser } from "@xmldom/xmldom";
import load_mujoco from "@mujoco/mujoco";
import { buildVFS } from "./model-vfs.js";
import { PoseController } from "./ik.js";
import { BimanualTrayCarry, trayQuat } from "./tray.js";

const ROOT = path.resolve(import.meta.dirname, "..");
const SCENE = "scenes/tray_grasp_scene.xml";

async function build() {
  const mujoco = await load_mujoco();
  const vfs = await buildVFS(
    mujoco,
    SCENE,
    async (p) => new Uint8Array(fs.readFileSync(path.join(ROOT, p))),
    DOMParser,
  );
  let model;
  try {
    model = mujoco.MjModel.from_xml_path(SCENE, vfs);
  } finally {
    vfs.delete();
  }
  const data = new mujoco.MjData(model);
  const ik = new PoseController(mujoco, model);
  return { mujoco, carry: new BimanualTrayCarry(mujoco, model, data, ik) };
}

// One carry: lift, then travel sideways, with the tilt from `policy`.
function fly(carry, policy, steps = 260) {
  const [origin] = carry.trayPose();
  const target = [origin[0], origin[1] + 0.08, origin[2] + 0.10];
  let peak = 0;
  for (let i = 0; i < steps; i++) {
    const a = Math.min(1, (i + 1) / (steps - 40));
    const commanded = [
      origin[0] * (1 - a) + target[0] * a,
      origin[1] * (1 - a) + target[1] * a,
      origin[2] * (1 - a) + target[2] * a,
    ];
    const tilt = policy(carry);
    carry.commandTray(commanded, trayQuat(tilt[0], tilt[1]));
    const [rel] = carry.ballInTray();
    peak = Math.max(peak, Math.abs(rel[0]), Math.abs(rel[1]));
    if (!carry.ballOnTray()) break;
  }
  return { kept: carry.ballOnTray(), peak };
}

test("the grasp holds the tray without bulldozing it", async () => {
  const { carry } = await build();
  carry.reset();
  const [before] = carry.trayPose();
  const grip = carry.grasp();
  const [after] = carry.trayPose();
  const shove = Math.hypot(after[0] - before[0], after[1] - before[1]);
  // Python measures 5.8 mm here; allow the port a wide margin but not the
  // 32 mm the bulldozing approach produced.
  assert.ok(shove < 0.012, `tray shoved ${(shove * 1000).toFixed(1)} mm`);
  for (const side of ["left", "right"]) {
    assert.ok(grip[side] > 25, `${side} grip only ${grip[side].toFixed(1)} N`);
  }
});

test("balancing keeps the ball where carrying level loses it", async () => {
  const { carry } = await build();

  carry.reset();
  carry.grasp();
  carry.placeBall([0.03, 0.0]);
  const balanced = fly(carry, (c) => c.balanceSetpoint());

  carry.reset();
  carry.grasp();
  carry.placeBall([0.03, 0.0]);
  const level = fly(carry, () => [0, 0]);

  assert.ok(balanced.kept, "the balancer lost the ball");
  assert.ok(
    balanced.peak < level.peak,
    `balanced peak ${(balanced.peak * 1000).toFixed(0)} mm ` +
      `vs level ${(level.peak * 1000).toFixed(0)} mm`,
  );
  console.log(
    `  balanced: kept=${balanced.kept} peak=${(balanced.peak * 100).toFixed(1)} cm | ` +
      `level: kept=${level.kept} peak=${(level.peak * 100).toFixed(1)} cm`,
  );
});
