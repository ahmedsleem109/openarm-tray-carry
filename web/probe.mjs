// Feasibility probe: does the tray scene load and step under MuJoCo WASM?
import fs from "node:fs";
import path from "node:path";
import { DOMParser } from "@xmldom/xmldom";
import load_mujoco from "@mujoco/mujoco";
import { buildVFS } from "./model-vfs.js";

const ROOT = path.resolve(import.meta.dirname, "..");
const SCENE = "scenes/tray_grasp_scene.xml";

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
console.log(`loaded: ${model.ngeom} geoms, ${model.nq} qpos, ${model.nu} actuators`);

mujoco.mj_resetDataKeyframe(model, data, 0);
mujoco.mj_forward(model, data);
const started = performance.now();
const steps = 2000;
for (let i = 0; i < steps; i++) mujoco.mj_step(model, data);
const elapsed = (performance.now() - started) / 1000;
console.log(
  `${steps} steps in ${elapsed.toFixed(2)}s = ${(steps / elapsed).toFixed(0)} steps/s ` +
    `(realtime needs ${1 / model.opt_timestep ?? 0.001} steps/s)`,
);
