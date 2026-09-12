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

// The bimanual tray carry, in the browser. A port of
// openarm_gym/control/bimanual_tray.py, and deliberately a close one: the same
// constants, the same sequence, the same balance law, so that what a visitor
// watches in a tab is the controller the measurements in STATUS.md were taken
// from rather than something that merely looks like it.
//
// What is NOT ported is the vision policy. Its training images come from
// MuJoCo's own renderer and this page draws with three.js, so feeding it
// browser pixels would be a different visual distribution -- it would fail, and
// worse, it would look like the policy failing rather than the setup being
// wrong. The page therefore runs the CLASSICAL controller, which reads the
// ball's state directly, and says so on screen. The vision results are in the
// repository, measured properly.
//
// IK comes from Enactic's ik.js unchanged: it drives the same
// left/right_ee_control_point sites by the same damped-least-squares solve as
// the Python PoseController.

const PAD_OFFSET = 0.125; // finger pads ahead of the wrist site, metres
const TOOL_QUAT = [0.7071, 0.0, -0.7071, 0.0];
const TOOL_DIR = [1.0, 0.0, 0.0];

const QPOS = { left: 0, right: 9 }; // first of seven arm joints
const CTRL = { left: 0, right: 8 }; // first of seven arm actuators
const FINGER_QPOS = { left: [7, 8], right: [16, 17] };
const FINGER_CTRL = { left: 7, right: 15 };
// The right gripper opens toward negative values, the left toward positive.
const OPEN = { left: 0.7854, right: -0.7854 };
const SHUT = { left: 0.0, right: 0.0 };

const APPROACH_LIFT = 0.02; // reach in above the handle, then settle down
const CLOSE_LIFT = 0.0;
const BEHIND_X = 0.23;

const GRAVITY = 9.81;
export const BALANCE_KP = 12.0;
export const BALANCE_KD = 4.5;
export const BALANCE_MAX_TILT = 0.2;

const ARM_DOFS = 18; // gravity compensation applies to these and no further
const TRAY_QVEL = 18;
const BALL_QVEL = 24;

const SIDES = ["left", "right"];

// -- small vector and quaternion helpers -------------------------------------

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

export function quatMul(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export const quatConj = (q) => [q[0], -q[1], -q[2], -q[3]];

// Row-major 3x3, matching MuJoCo's own layout.
export function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

const matVec = (m, v) => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];

// The transpose, which is the inverse for a rotation: world into the tray frame.
const matTVec = (m, v) => [
  m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
  m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
  m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
];

export function matToQuat(m) {
  const trace = m[0] + m[4] + m[8];
  if (trace > 0) {
    const s = Math.sqrt(trace + 1.0) * 2;
    return [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s];
  }
  if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2;
    return [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s];
  }
  if (m[4] > m[8]) {
    const s = Math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2;
    return [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s];
  }
  const s = Math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2;
  return [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s];
}

// A small roll about x then pitch about y, inside a yaw about world z. The
// order matters: the balance law works in the tray's own frame, so composing
// yaw * tilt applies the tilt in the yawed frame. The other order silently
// rotates the feedback by the yaw angle.
export function tiltQuat(roll, pitch) {
  const qr = [Math.cos(roll / 2), Math.sin(roll / 2), 0, 0];
  const qp = [Math.cos(pitch / 2), 0, Math.sin(pitch / 2), 0];
  return quatMul(qp, qr);
}

export function trayQuat(roll, pitch, yaw = 0) {
  const qy = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
  return quatMul(qy, tiltQuat(roll, pitch));
}

// The PD law itself. On a surface tilted by a small angle the in-plane
// acceleration is g*sin(t)*cos(t) ~= g*t, so a PD on the ball's tray-frame
// state maps straight to a commanded tilt.
export function balanceLaw(ballXY, ballVXY, kp = BALANCE_KP, kd = BALANCE_KD, maxTilt = BALANCE_MAX_TILT) {
  const ax = -kp * ballXY[0] - kd * ballVXY[0];
  const ay = -kp * ballXY[1] - kd * ballVXY[1];
  const clamp = (v) => Math.max(-maxTilt, Math.min(maxTilt, v));
  // pitch about y from the x acceleration, roll about x from the y one.
  return [clamp(-ay / GRAVITY), clamp(ax / GRAVITY)];
}

// -- the controller ----------------------------------------------------------

export class BimanualTrayCarry {
  // `mujoco` is the WASM module, `ik` an Enactic PoseController over the same
  // model. controlHz must match whatever a differencing policy assumes.
  constructor(mujoco, model, data, ik, { controlHz = 50 } = {}) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.ik = ik;
    this.controlHz = controlHz;
    this.timestep = model.opt_timestep ?? 0.001;
    this.nSubsteps = Math.max(1, Math.round(1 / (controlHz * this.timestep)));

    this.trayBody = this.#body("tray");
    this.ballBody = this.#body("ball");
    this.handleGeom = {
      left: this.#geom("handle_left"),
      right: this.#geom("handle_right"),
    };
    this.grasped = false;
    this.grasp_rel = {};
  }

  #body(name) {
    return this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_BODY.value, name);
  }

  #geom(name) {
    return this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_GEOM.value, name);
  }

  trayPose() {
    const p = this.data.xpos.subarray(this.trayBody * 3, this.trayBody * 3 + 3);
    const q = this.data.xquat.subarray(this.trayBody * 4, this.trayBody * 4 + 4);
    return [Array.from(p), Array.from(q)];
  }

  handlePos(side) {
    const id = this.handleGeom[side];
    return Array.from(this.data.geom_xpos.subarray(id * 3, id * 3 + 3));
  }

  // Ball position and velocity in the tray frame -- the quantity the balance
  // law consumes, and the one a vision policy has to recover from pixels.
  ballInTray() {
    const [trayPos, trayQ] = this.trayPose();
    const rot = quatToMat(trayQ);
    const ballPos = Array.from(
      this.data.xpos.subarray(this.ballBody * 3, this.ballBody * 3 + 3),
    );
    const rel = matTVec(rot, sub(ballPos, trayPos));
    const bv = this.data.qvel.subarray(BALL_QVEL, BALL_QVEL + 3);
    const tv = this.data.qvel.subarray(TRAY_QVEL, TRAY_QVEL + 3);
    const vel = matTVec(rot, [bv[0] - tv[0], bv[1] - tv[1], bv[2] - tv[2]]);
    return [rel, vel];
  }

  ballOnTray(halfX = 0.075, halfY = 0.15) {
    const [rel] = this.ballInTray();
    return Math.abs(rel[0]) <= halfX && Math.abs(rel[1]) <= halfY && rel[2] > -0.02;
  }

  // Total normal force at each hand's fingers. The tray is held by friction and
  // form closure, so this is the number that says whether the grasp is real.
  gripForce() {
    const out = { left: 0, right: 0 };
    // Allocated once and kept: this runs every frame, and the buffer lives in
    // the WASM heap where churn is not free.
    this.forceBuffer ??= new this.mujoco.DoubleBuffer(6);
    for (let i = 0; i < this.data.ncon; i++) {
      const contact = this.data.contact.get(i);
      for (const side of SIDES) {
        const handle = this.handleGeom[side];
        if (contact.geom1 !== handle && contact.geom2 !== handle) continue;
        this.mujoco.mj_contactForce(this.model, this.data, i, this.forceBuffer);
        // The first component is the contact-frame normal.
        out[side] += Math.abs(this.forceBuffer.GetView()[0]);
      }
    }
    return out;
  }

  #siteTarget(padTarget) {
    return sub(padTarget, scale(TOOL_DIR, PAD_OFFSET));
  }

  // The finger pads, from the wrist site they hang off. The fingers extend
  // along the control-point site's local -Z, which is the same convention the
  // Python PoseController uses.
  padPos(side) {
    const { pos, quat } = this.ik.arms[side].pose(this.data);
    const rot = quatToMat(quat);
    return add(pos, matVec(rot, [0, 0, -PAD_OFFSET]));
  }

  wristQuat(side) {
    return this.ik.arms[side].pose(this.data).quat;
  }

  // One control step: IK both arms, hold the grip, advance the physics.
  #servo(padTargets, quats, grip) {
    for (const side of SIDES) {
      this.ik.syncFrom(this.data);
      // The solver's own defaults differ between the two ports; these are the
      // Python ones, so the JS arm follows the same path as the measured arm.
      const q = this.ik.solve(side, this.#siteTarget(padTargets[side]), quats[side], {
        iters: 50,
        tol: 1e-4,
        lambda2: 1e-4,
      });
      for (let j = 0; j < 7; j++) this.data.ctrl[CTRL[side] + j] = q[j];
      this.data.ctrl[FINGER_CTRL[side]] = grip[side];
    }
    for (let s = 0; s < this.nSubsteps; s++) {
      // Gravity compensation on the arm DOFs only, as the real driver applies.
      // Including the tray and the ball here would make the ball weightless,
      // which is a very convincing way to solve the wrong problem.
      for (let d = 0; d < ARM_DOFS; d++) {
        this.data.qfrc_applied[d] = this.data.qfrc_bias[d];
      }
      this.mujoco.mj_step(this.model, this.data);
    }
  }

  // A generator, because the browser has to paint between control steps: the
  // approach takes several seconds of simulated time, and running it as one
  // blocking call freezes the tab through the most interesting part of the
  // demo. Callers that do not care -- the tests, and anything headless --
  // drain it.
  *#sweep(goal, grip, steps, quats) {
    const start = {};
    for (const side of SIDES) start[side] = this.padPos(side);
    for (let i = 0; i < steps; i++) {
      const a = (i + 1) / steps;
      const targets = {};
      for (const side of SIDES) {
        targets[side] = add(scale(start[side], 1 - a), scale(goal[side], a));
      }
      this.#servo(targets, quats, grip);
      yield;
    }
  }

  // Arms retracted and clear of the tray, jaws open. The scene's home keyframe
  // puts the closed jaws inside the handle posts; starting there makes the
  // contact impulse throw the tray across the room.
  reset() {
    this.mujoco.mj_resetDataKeyframe(this.model, this.data, 0);
    this.mujoco.mj_forward(this.model, this.data);
    this.grasped = false;
    this.grasp_rel = {};
    for (const side of SIDES) {
      const sign = side === "left" ? 1 : -1;
      const target = this.#siteTarget([0.24, sign * 0.22, 0.58]);
      this.ik.syncFrom(this.data);
      const q = this.ik.solve(side, target, TOOL_QUAT, {
        iters: 200,
        tol: 1e-4,
        lambda2: 1e-4,
      });
      for (let j = 0; j < 7; j++) {
        this.data.qpos[QPOS[side] + j] = q[j];
        this.data.ctrl[CTRL[side] + j] = q[j];
      }
      for (const qi of FINGER_QPOS[side]) this.data.qpos[qi] = OPEN[side];
      this.data.ctrl[FINGER_CTRL[side]] = OPEN[side];
    }
    this.data.qvel.fill(0);
    this.data.qfrc_applied.fill(0);
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Approach the posts from behind, close, and freeze the grasp transform.
  //
  // The detour above the handle centre is not stylistic: reaching straight in
  // puts the gripper's palm against the tray's top face and bulldozes the whole
  // tray ~32 mm forward before the jaws arrive. Reaching in 20 mm higher and
  // settling cuts that to ~6 mm.
  // Drain the generator. Headless callers want the whole grasp to happen now.
  grasp() {
    for (const _ of this.graspSteps());
    return this.gripForce();
  }

  *graspSteps() {
    const handles = {};
    for (const side of SIDES) handles[side] = this.handlePos(side);
    const quats = { left: TOOL_QUAT, right: TOOL_QUAT };
    const at = (dz) => {
      const out = {};
      for (const side of SIDES) {
        out[side] = [handles[side][0], handles[side][1], handles[side][2] + dz];
      }
      return out;
    };
    const behind = {};
    for (const side of SIDES) behind[side] = [BEHIND_X, handles[side][1], 0.52];
    const lifted = {};
    for (const side of SIDES) {
      lifted[side] = [BEHIND_X, handles[side][1], handles[side][2] + APPROACH_LIFT];
    }

    const long = Math.round(0.4 * this.controlHz * 8);
    yield* this.#sweep(behind, OPEN, long, quats);
    yield* this.#sweep(lifted, OPEN, long, quats);
    yield* this.#sweep(at(APPROACH_LIFT), OPEN, Math.round(0.5 * this.controlHz * 8), quats);
    // Straight down with the jaws still open: a vertical settle presses on a
    // tray that is still resting on the table, it does not push it.
    yield* this.#sweep(at(CLOSE_LIFT), OPEN, Math.round(0.3 * this.controlHz * 8), quats);
    yield* this.#sweep(at(CLOSE_LIFT), SHUT, Math.round(0.5 * this.controlHz * 8), quats);

    const [trayPos, trayQ] = this.trayPose();
    const rot = quatToMat(trayQ);
    for (const side of SIDES) {
      const pad = this.padPos(side);
      const wristQuat = this.wristQuat(side);
      this.grasp_rel[side] = {
        pos: matTVec(rot, sub(pad, trayPos)),
        quat: quatMul(quatConj(trayQ), wristQuat),
      };
    }
    this.grasped = true;
  }

  // THE coordination step: one tray pose in, both arms' targets out, derived
  // from the transform captured when the jaws closed. The arms are never
  // commanded separately, so they cannot fight each other through the tray.
  commandTray(pos, quat) {
    if (!this.grasped) throw new Error("commandTray called before grasp()");
    const rot = quatToMat(quat);
    const targets = {};
    const quats = {};
    for (const side of SIDES) {
      const rel = this.grasp_rel[side];
      targets[side] = add(pos, matVec(rot, rel.pos));
      quats[side] = quatMul(quat, rel.quat);
    }
    this.#servo(targets, quats, SHUT);
  }

  // The classical balancer's setpoint for the current ball state.
  balanceSetpoint() {
    if (!this.ballOnTray()) return [0, 0];
    const [rel, vel] = this.ballInTray();
    return balanceLaw([rel[0], rel[1]], [vel[0], vel[1]]);
  }

  // Give the ball an in-plane velocity kick, in the tray frame. This is what
  // the page's "shove" button does, and what the disturbances in the measured
  // evaluations are: a velocity in m/s, not an impulse. The ball weighs 2.73 g,
  // so a plausible-looking 0.004 N*s impulse is a 1.5 m/s launch.
  nudgeBall(kickXY) {
    const [, trayQ] = this.trayPose();
    const rot = quatToMat(trayQ);
    const world = matVec(rot, [kickXY[0], kickXY[1], 0]);
    for (let i = 0; i < 3; i++) this.data.qvel[BALL_QVEL + i] += world[i];
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Seat the ball on the tray face, at rest, in the tray's own frame.
  placeBall(offsetXY = [0, 0]) {
    const [trayPos, trayQ] = this.trayPose();
    const rot = quatToMat(trayQ);
    const local = [offsetXY[0], offsetXY[1], 0.025]; // half-thickness + radius
    const world = add(trayPos, matVec(rot, local));
    for (let i = 0; i < 3; i++) this.data.qpos[25 + i] = world[i];
    this.data.qpos[28] = 1;
    for (let i = 29; i < 32; i++) this.data.qpos[i] = 0;
    for (let i = BALL_QVEL; i < BALL_QVEL + 6; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

  // Jitter where the tray starts, inside the envelope the carry was measured
  // over. Asymmetric on purpose: pulling the tray closer than about x=0.33
  // folds the arms up and the carry starts throwing the ball, while reaching
  // out to 0.40 is free.
  randomizeLayout(random = Math.random) {
    const pick = (lo, hi) => lo + (hi - lo) * random();
    const dx = pick(-0.015, 0.04);
    const dy = pick(-0.02, 0.02);
    const yaw = pick(-0.05, 0.05);
    const home = [0.35, 0.0, 0.405];
    this.data.qpos[18] = home[0] + dx;
    this.data.qpos[19] = home[1] + dy;
    this.data.qpos[20] = home[2];
    const spin = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    for (let i = 0; i < 4; i++) this.data.qpos[21 + i] = spin[i];
    for (let i = TRAY_QVEL; i < TRAY_QVEL + 6; i++) this.data.qvel[i] = 0;
    this.mujoco.mj_forward(this.model, this.data);
    this.placeBall([0, 0]);
    return { dx, dy, yaw };
  }

  trayYaw() {
    const [, q] = this.trayPose();
    const rot = quatToMat(q);
    return Math.atan2(rot[3], rot[0]);
  }
}

export { norm, SIDES };
