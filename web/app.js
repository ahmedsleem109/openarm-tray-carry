// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The page: load the scene into MuJoCo WASM, grasp the tray, then carry it on a
// slow patrol while the classical balancer keeps the ball on board. The grasp is
// shown rather than skipped -- watching the arms find the posts and close on
// them is half of what makes this a manipulation demo instead of a physics toy.

import load_mujoco from "@mujoco/mujoco";
import { buildVFS } from "./model-vfs.js";
import { PoseController } from "./ik.js";
import { BimanualTrayCarry, trayQuat } from "./tray.js";
import { SceneView } from "./render.js";

const SCENE = "scenes/tray_grasp_scene.xml";
// Files are fetched relative to the repository root, one level above web/.
const ROOT = "../";

const CONTROL_HZ = 50;
//: Seconds of simulation to advance per rendered frame, capped so a slow
//: machine falls behind in time rather than locking up the tab.
const MAX_STEP_SECONDS = 0.05;

const ui = {
  phase: document.getElementById("phase"),
  status: document.getElementById("status"),
  offset: document.getElementById("offset"),
  offsetBar: document.getElementById("offsetBar"),
  tilt: document.getElementById("tilt"),
  grip: document.getElementById("grip"),
  rate: document.getElementById("rate"),
  fps: document.getElementById("fps"),
  shove: document.getElementById("shove"),
  restart: document.getElementById("restart"),
  balancing: document.getElementById("balancing"),
  canvas: document.getElementById("view"),
};

function setPhase(text, tone = "") {
  ui.phase.innerHTML = tone ? `<b style="color:${tone}">${text}</b>` : `<b>${text}</b>`;
}

async function boot() {
  setPhase("loading physics…");
  const mujoco = await load_mujoco();

  setPhase("loading the scene…");
  const vfs = await buildVFS(
    mujoco,
    SCENE,
    async (p) => {
      const response = await fetch(ROOT + p);
      if (!response.ok) throw new Error(`${response.status} fetching ${p}`);
      return new Uint8Array(await response.arrayBuffer());
    },
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
  const carry = new BimanualTrayCarry(mujoco, model, data, ik, { controlHz: CONTROL_HZ });
  const view = new SceneView(ui.canvas, model);

  const fit = () => {
    const width = ui.canvas.clientWidth;
    view.resize(width, Math.round((width * 9) / 16));
  };
  window.addEventListener("resize", fit);
  fit();

  // Drag to orbit. Nothing about the simulation changes; only the camera.
  let azimuth = -0.09;
  let dragging = false;
  let lastX = 0;
  ui.canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    ui.canvas.setPointerCapture(e.pointerId);
  });
  ui.canvas.addEventListener("pointerup", () => (dragging = false));
  ui.canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    azimuth += (e.clientX - lastX) / 600;
    lastX = e.clientX;
    view.setView(azimuth);
  });
  view.setView(azimuth);

  const state = {
    running: false,
    step: 0,
    origin: null,
    lostAt: null,
    frames: 0,
    lastFpsAt: performance.now(),
    stepsThisSecond: 0,
  };

  // The grasp takes a few seconds of simulated time. It is run inside the
  // animation loop rather than blocking, so the page stays responsive and the
  // approach is visible.
  let pending = null;

  function startEpisode(randomize) {
    state.running = false;
    setPhase("resetting…", "#fabe5a");
    carry.reset();
    if (randomize) carry.randomizeLayout();
    // Hand the grasp to the loop as a generator so it can be stepped a slice at
    // a time; the alternative is a frozen tab for several seconds.
    pending = graspSlices();
  }

  // Control steps of the approach to run per rendered frame. The grasp is about
  // 840 control steps, so this shows it in roughly three seconds -- fast enough
  // not to bore, slow enough to see the jaws close.
  const GRASP_SLICE = 14;

  function* graspSlices() {
    setPhase("grasping — approaching the handle posts", "#fabe5a");
    const steps = carry.graspSteps();
    for (;;) {
      let done = false;
      for (let i = 0; i < GRASP_SLICE && !done; i++) done = steps.next().done;
      if (done) break;
      yield;
    }
    const grip = carry.gripForce();
    state.origin = carry.trayPose()[0];
    state.step = 0;
    state.lostAt = null;
    state.running = true;
    ui.status.textContent =
      `grasped: ${grip.left.toFixed(0)} N left, ${grip.right.toFixed(0)} N right — ` +
      `friction and form closure only, no weld`;
    setPhase("carrying", "#5ad28c");
  }

  // A slow patrol, so there is always motion to disturb. Positions are offsets
  // from wherever the tray was grasped, which is what makes the same path valid
  // after the layout is randomized.
  function commandedPose(step) {
    const t = step / CONTROL_HZ;
    const lift = 0.10 * Math.min(1, t / 2.0);
    return [
      state.origin[0] + 0.015 * Math.sin(t * 0.55),
      state.origin[1] + 0.085 * Math.sin(t * 0.32),
      state.origin[2] + lift,
    ];
  }

  // Physics only. The readout is drawn once per *frame*, not once per control
  // step: a control step is 20 physics steps and there are up to 50 of them per
  // frame, so updating the DOM here meant fifty layout invalidations per paint
  // and cost more than the simulation did.
  function controlStep() {
    const tilt = ui.balancing.checked ? carry.balanceSetpoint() : [0, 0];
    carry.commandTray(commandedPose(state.step), trayQuat(tilt[0], tilt[1]));
    state.tilt = tilt;
    state.step += 1;
    state.stepsThisSecond += carry.nSubsteps;
    if (!carry.ballOnTray() && state.lostAt === null) {
      state.lostAt = state.step;
      setPhase("ball lost — turn balancing back on", "#f56e6e");
      ui.status.textContent =
        "the tray is still being carried exactly as before; nothing else changed";
    }
  }

  function drawReadout() {
    const [rel] = carry.ballInTray();
    const offset = Math.hypot(rel[0], rel[1]);
    const on = carry.ballOnTray();
    ui.offset.textContent = on ? `${(offset * 1000).toFixed(0)} mm` : "—";
    const fraction = Math.min(1, offset / 0.075);
    ui.offsetBar.style.width = `${fraction * 100}%`;
    ui.offsetBar.style.background =
      fraction > 0.75 ? "#f56e6e" : fraction > 0.45 ? "#fabe5a" : "#5ad28c";
    const tilt = state.tilt ?? [0, 0];
    ui.tilt.textContent =
      `${((tilt[0] * 180) / Math.PI).toFixed(1)}° / ${((tilt[1] * 180) / Math.PI).toFixed(1)}°`;
    // The contact scan walks every contact in the model, so it is a per-frame
    // cost rather than a per-control-step one.
    const grip = carry.gripForce();
    ui.grip.textContent = `${Math.min(grip.left, grip.right).toFixed(0)} N`;
  }

  let lastFrameAt = performance.now();
  function frame() {
    const now = performance.now();
    const elapsed = Math.min((now - lastFrameAt) / 1000, MAX_STEP_SECONDS);
    lastFrameAt = now;

    if (pending) {
      const { done } = pending.next();
      if (done) pending = null;
    } else if (state.running) {
      const steps = Math.max(1, Math.round(elapsed * CONTROL_HZ));
      for (let i = 0; i < steps; i++) controlStep();
    }

    view.sync(data);
    view.draw();
    if (state.running) drawReadout();

    state.frames += 1;
    if (now - state.lastFpsAt > 1000) {
      const seconds = (now - state.lastFpsAt) / 1000;
      ui.fps.textContent = `${(state.frames / seconds).toFixed(0)} fps`;
      ui.rate.textContent = `${(state.stepsThisSecond / seconds / 1000).toFixed(1)}k steps/s`;
      state.frames = 0;
      state.stepsThisSecond = 0;
      state.lastFpsAt = now;
    }
    requestAnimationFrame(frame);
  }

  ui.shove.addEventListener("click", () => {
    if (!state.running) return;
    const angle = Math.random() * Math.PI * 2;
    // 0.25 m/s: inside what the balancer was measured to recover along the
    // short axis (0.30 m/s) and well past what it survives without balancing.
    const speed = 0.25;
    carry.nudgeBall([speed * Math.cos(angle), speed * Math.sin(angle)]);
    ui.status.textContent = `shoved the ball at ${speed} m/s`;
  });
  ui.restart.addEventListener("click", () => startEpisode(true));
  ui.balancing.addEventListener("change", () => {
    ui.status.textContent = ui.balancing.checked
      ? "balancing on — the tray tilts to bring the ball back"
      : "balancing off — the tray is carried level, and the ball goes where physics takes it";
  });

  startEpisode(false);
  requestAnimationFrame(frame);
}

boot().catch((error) => {
  setPhase("failed to start", "#f56e6e");
  ui.status.textContent = String(error);
  console.error(error);
});
