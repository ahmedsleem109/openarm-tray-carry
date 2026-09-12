# openarm tray carry — plan

Two OpenArm v2 arms doing something difficult **together**: grasping a tray by
its handles and carrying it while keeping a loose ball from rolling off.

> **This file was rewritten.** It previously described a different project — a
> three-task benchmark with demonstration datasets, an action-chunking policy, a
> torque ablation and a browser leaderboard. That direction was scope creep and
> is not being resumed. What it left behind that is still used is the
> environment layer (`env.py`, `ik.py`, `tasks/`, `experts/`), which supplies
> cameras, collisions, torque observations and the real 16-DOF action space. The
> old roadmap is in git history; a short account of why it was dropped is at the
> bottom of this file.

> **Next session: go to [the work queue](#the-work-queue--do-these-in-order) and
> start at P1.** P0 was done and measured on 2026-09-12; its results are in
> STATUS.md under "What P0 bought".

## The five requirements

Every proposed step is checked against these before it is started:

1. **Vision** — camera observations, not privileged state.
2. **A control algorithm applied** — not end-to-end joint-angle RL.
3. **Sim-to-real plausibility.**
4. **Collisions enabled.**
5. **The object grasped, not welded.**

## Why a tray, and why it is hard

A tray held at both ends is the smallest task that is *genuinely* bimanual. The
two arms are not doing two things at once; they are mechanically coupled through
a rigid object they are both holding by friction and form closure. Any
disagreement between them is resolved as force through the grasp, and the grasp
is what fails first. Then the ball makes the coupling matter: the tray's
*orientation* is the control input for a second, faster dynamic system riding on
top of the first.

It also has the property that makes the rest of the plan affordable. Once both
grippers hold the tray, each arm's target is determined by one commanded tray
pose, so **the whole system has a six-dimensional action space** rather than
fourteen joint angles. That is what puts a learned vision policy within reach of
one laptop GPU with no batched renderer.

## Architecture

```
        cameras ──► vision policy ──► ball state estimate  (x, y, vx, vy)
                                              │
                                              ▼
                                      balance_law  (PD)
                                              │
   carry plan ──► tray position, yaw ──► command_tray ──► IK ──► both arms
                                              ▲
                                     grasp transform, captured
                                     when the jaws closed
```

The vision policy **estimates state**; the classical law does the control. That
split is requirement 2 taken literally, and it is also what works: regressing
the tilt end to end was measured to shrink its outputs to about half the required
magnitude, and half the loop gain is a different controller. See STATUS.md.

## Stages

### 1. The scene and the grasp — done

`v2/pedestal/tray_grasp_scene.xml`, derived from the shipped `balance_scene.xml`;
nothing Enactic ships was modified. No welds anywhere, 43 collidable geoms, box
handles that give form closure.

**Gate:** the tray is lifted and carried on friction and form closure alone,
asserted by tests. ✅

### 2. Coordinated carry and classical balancing — done

`BimanualTrayCarry`: one tray pose in, both arms' targets out. PD on the ball's
tray-frame state, mapped to tilt.

**Gate:** an ablation test in which the same carry keeps the ball with balancing
and loses it without. ✅

### 3. Vision — done

A 0.16M-parameter convolutional policy over two 84x112 cameras and a two-frame
stack, estimating the ball's tray-frame state and driving `balance_law` with it.

**Gate:** closed-loop survival on unseen randomized plans, against the classical
ceiling and the no-balancing floor on identical seeds. ✅ **12/12, matching the
classical controller**, holding the ball to 3.2 cm against its 3.6 cm.

The lesson worth carrying: in `estimator` mode the network is a **detector**, and
a detector should be trained on the state distribution you want it accurate over
— which is uniform, not whatever a stabilising controller happens to visit.
Behaviour cloning alone scored 0/12 while being accurate to 4.6 mm on the
expert's own distribution.

### 3b. Vision for the grasp — done

The grasp used to read the handle posts out of the simulator, which left
requirement 1 holding for the balancing half of the task and not the grasping
half. `openarm_gym/policies/handle_vision.py` is a 0.08M-parameter detector over
one frame per camera at the arms' retracted pose; `grasp(handles=...)` takes its
estimate, and the privileged path stays behind the same argument.

**Gate:** grasp success over randomized layouts, from cameras only, tray
displacement under ~10 mm. ✅ **12/12 clean grasps at 5.6 mm**, which is the
privileged approach's own displacement, on a 0.7 mm handle estimate.

### 4. Sim-to-real — measured, not finished

Torque-level actuation, explicit torque limits, command latency, transmission
backlash and dynamics randomization all exist and are measured. What that
bought: the carry survives on 40% of the DM motors' rated torque and 40 ms of
latency, and is hurt most by backlash, which unloads the grip.

The vision policy has now been flown under all of it. It holds **12/12 under
every condition and under all of them at once**, and drops to 8/12 only at 1.5
degrees of backlash, where the classical controller drops to 11/12 as well. The
full table is in STATUS.md.

**Gate for "finished":** a real arm. Until one exists, every number here is a
simulation number and is labelled as one.

## The work queue — do these in order

Written 2026-09-12 for whoever picks this up next. **P0 is done, including the
retraining it opened. Start at P1.** Each item says why it matters, what to
actually do, and how you know it is finished. Effort is a rough
half-day/day/week scale on this machine.

---

### P0 — make the current claim honest — **done 2026-09-12**

All three are built, measured and tested; the numbers are in STATUS.md under
"What P0 bought", and the raw tables are in `results/tray_realism.json`,
`results/visual_grasp.json` and `results/tray_layout.json`. What is left of this
section is the shape of what was done and the two things it left open.

#### P0.1 Evaluate the vision policy under the sim-to-real layer ✅

`openarm_gym/control/tray_eval.py` holds the named conditions;
`scripts/evaluate_tray_realism.py` flies classical and vision under each on
identical seeds, and `scripts/evaluate_tray_vision.py` takes the same flags for
a single condition.

**Result:** 12/12 under every condition and under all of them at once. The
predicted drop did not happen. On the first checkpoint 1.5 degrees of backlash
cost it 8/12 against classical's 11/12; after the layout retraining below the
two rows are identical at 11/12, so that condition is a mechanical limit and not
a perception one. The table lives in STATUS.md.

#### P0.2 Make the grasp visual ✅

A separate single-frame detector rather than a head on the ball policy -- the
handles do not move before they are grasped, so it runs once per episode.

**Result:** 12/12 clean grasps over randomized layouts at 5.6 mm of tray
displacement, against the privileged path's 5.9 mm, on a 0.7 mm handle estimate.

**Left open:** the detector outputs world coordinates, which is well-posed only
because the scene bolts its cameras to the world. On hardware that is extrinsic
calibration, and it is this piece's sim-to-real gap.

#### P0.3 Randomize the scene layout ✅

`BimanualTrayCarry.randomize_layout`, run through `run_plan`'s new `on_reset`
window.

**Result:** classical 12/12 over randomized layouts (the gate), vision 12/12.

**Then closed, the same day:** the ball estimator's sight error had risen from
4.5 mm to 8.0 mm over randomized layouts, because it was trained on exactly one.
The pipeline was re-run with the randomizer on -- demonstrations, detector
frames over 90 layouts, a DAgger round, a retrain -- and it is back to
**4.7 mm** at 12/12, costing 0.8 mm on the single scene it used to specialise
in. It also closed the backlash gap in the P0.1 table, which had been misread as
a mechanical-plus-perceptual compound failure. `D:/openarm_data/tray_vision_layout/policy.pt` is the checkpoint to use from
here; the old one is still at `D:/openarm_data/tray_vision/policy.pt` for
comparison.

**Two things this section turned up that are worth carrying:**

- A plan's yaw goal was **absolute** while its positions were relative, which no
  fixed-layout episode could ever have shown. See STATUS.md.
- The carry's reach envelope is **not** the static grasp's. Pulling the tray
  closer is what breaks it; reaching out is free. The layout ranges lean outward
  for that reason and should not be "corrected" to be symmetric.

### P1 — the capability step: residual RL on the setpoint space — *a week*

**This is the recommended next real piece of work**, because the hard parts are
already done: a classical controller that works, and a **six-dimensional** action
space instead of fourteen joint angles.

Learn a *correction* on top of `balance_law` rather than replacing it. Classical
guarantees are kept, exploration starts from a competent policy instead of
noise, and the failure mode is bounded by clipping the residual. It is also what
lets the task get genuinely harder — faster carries, heavier disturbances, an
object that is not a sphere — where a hand-tuned PD runs out.

- **Train on state, on GPU, in WSL2.** `~/venvs/mjlab` already has
  `mujoco_warp` + `rsl_rl`. STATUS.md records why this and **not MJX** is the GPU
  path, and that ~5.6k env-steps/s at 2048 envs is the throughput to expect.
- **Distil to vision afterwards** through the estimator that already exists —
  do not try to do vision RL, there is no batched GPU renderer
  (no `madrona_mjx`).

**Gate:** the residual policy beats the classical controller on a manoeuvre the
classical controller fails — not on the one it already passes 12/12.

---

### P2 — perception that generalises — *two to three days*

Replace the bespoke CNN detector with **DINOv2/v3 features or SAM2
point-tracking** on the ball. This removes dataset collection entirely and is the
honest answer to "does this generalise, or did you overfit one scene and one
lighting setup?"

The current detector is a strong baseline to measure against — it reaches 4.5 mm
closed loop — which makes this a real experiment rather than a re-skin. Compare on
the P0.3 randomized layouts and under the P0.1 sensor model.

---

### P3 — the browser demo — *one to two days, high impact per hour*

Almost all of it is already written: the repository ships a MuJoCo WASM +
three.js page in `web/`, `scripts/export_onnx.py` exists, and the policy is
**0.16M parameters**. A public URL where anyone watches two arms carry a tray
in-tab, with no install, is disproportionately convincing for the effort.

**Gate:** a rollout renders in-browser at interactive frame rates on a mid-range
laptop.

---

### P4 — needs hardware

- **3D Gaussian Splatting real2sim.** Reconstruct the real workspace, render the
  simulation through it, transfer zero-shot. This is the current answer to the
  visual sim-to-real gap, and it pairs with the camera sensor model already in
  `vision.py`.
- **Actual sim-to-real.** OpenArm is ~$6.5k bimanual. Until one exists, every
  number in this repository is a simulation number and is labelled as one.
- **Upstream a pull request** to `enactic/openarm_mujoco` for the environment
  layer. Needs a fork; nothing is pushed.

---

### Ranked last, deliberately

- **Language conditioning** ("carry the tray left and keep it level"). Cheap to
  add — `env.py` already carries a per-episode `instruction` field — but the
  least load-bearing thing on this list. This task's difficulty is contact and
  coordination; it is not goal-specification-limited.
- **Higher camera resolution.** Only worth it if the velocity channel is ever
  wanted *directly*: at 84x112 one overhead pixel is ~10 mm of tray and the ball
  moves a tenth of a pixel between stacked frames, which is why damping is
  differenced from the position estimate instead. It is not currently a
  bottleneck.

## Non-goals

- **No sim-to-real claims without hardware.** Everything here is a simulation
  result and gets labelled as one.
- **No return to the benchmark direction** — no dataset collection for its own
  sake, no policy leaderboard, no torque ablation.
- **Not a task planner or a VLA.** The interesting problem here is coordination
  and contact, and it is not language-shaped.

## What was dropped, and why

An earlier direction built three scripted tasks (`peg_socket`, `move_puck`,
`valve`), LeRobot demonstration datasets, an ACT-style policy, a
vision-versus-vision-plus-torque ablation and a `bench.html` leaderboard. It
worked end to end and produced one clear result: a policy that reproduced the
expert to 0.014 rad on expert-visited states scored **0/30** driving itself,
because 27 demonstrations do not cover the states it reaches once it starts
making its own errors. The ablation was therefore unanswerable at that data
scale and was reported as such rather than as "0% vs 0%, no effect".

That work is not wrong, and its environment layer is still in use. It simply was
not the project — which is two arms doing something difficult together.
