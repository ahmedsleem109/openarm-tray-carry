# Status — handoff

Rewritten 2026-09-12. Read this first in a new session.

---

**Next session: `PLAN.md` carries a prioritised work queue. P0 is done; start at
P1.** All three P0 items -- evaluate vision under the sim-to-real layer, make the
grasp visual, randomize the scene layout -- were built and measured on
2026-09-12, along with the retraining the third one called for, and their results
are in [what P0 bought](#what-p0-bought). **Use
`D:/openarm_data/tray_vision_layout/policy.pt`**, the checkpoint trained over
randomized layouts. P1 is residual RL on the setpoint space, which is a week and
needs WSL2.

## What this project is for

**Train two arms in MuJoCo to do something difficult together**, with all of:

1. **Vision** — camera observations, not privileged state
2. **A control algorithm** applied, not end-to-end joint-angle RL
3. **Sim-to-real** plausibility
4. **Collisions enabled**
5. **The object grasped, not welded**

The task is a **two-arm tray carry**: both grippers grasp handles on a tray and
carry it while keeping a free ball from rolling off.

`PLAN.md` was rewritten to match this. The old benchmark direction (three
scripted tasks, LeRobot datasets, an ACT policy, a torque ablation, a
leaderboard) is **not being resumed**; its environment layer is still used.

---

## Read this before trusting any earlier number

Six bugs were found and fixed on 2026-09-12 by an explicit audit. One of them
invalidates every vision dataset recorded before that date:

- **The estimator's target was read one control step after the image.**
  `run_plan` observed the cameras before stepping the simulation but read the
  ball's true position *after*, so every frame was labelled with the ball's
  position 20 ms in the future. Proven, not inferred: the recorded target matched
  the next step's state to 1e-11 while differing from the observed state by 2 mm.
- **Disturbance schedules silently lost shoves.** The schedule is a dict keyed by
  control step and the steps were drawn with replacement, so 14 plans in 200 got
  five of six requested shoves.
- **A driving policy with no cameras died on a bare `KeyError: 'pixels'`.**
- **`TrayVisionController` hard-coded 50 Hz** while differencing, which scales the
  damping term by the rate ratio with no other symptom.
- **Proprioception was normalised with `std + 1e-6`** rather than a floor, so a
  channel that happened not to move would divide drift by ~1e-6.
- **`record_episode` raised on a zero-step rollout**, and `write_mp4` silently
  cropped mismatched frames.

A seventh was found later the same day, by the layout randomization in P0.3:

- **A plan's yaw goal was absolute while its positions were relative.** A
  `CarryPlan` is expressed as offsets from the tray pose at grasp time, but the
  commanded yaw went into `tray_quat` as a world angle. With the tray always
  starting at the same heading that was invisible; the moment a layout started
  the tray 1.6 degrees off, step one ordered the arms to untwist it. Tray
  tracking 4.5 mm -> 10.2 mm, and the ball was thrown well before the episode's
  disturbance arrived. `run_plan` now captures the grasp-time heading as the
  origin for yaw too.

An eighth, found while building the demo renderer: **``stop_on_loss=False`` was
a half-measure.** It skipped the outer waypoint loop but still broke out of the
current one, so a run that lost the ball ended early anyway and two runs of one
plan came back different lengths -- which is exactly what a side-by-side
comparison cannot have.

Each now has a regression test. **125 tests pass** (was 98, was 40). The
measurements in this document were taken after the fixes unless marked
otherwise.

---

## What is built and working

Branch `openarm-gym`, nothing pushed, no fork exists.

### The classical controller ✅ — this is the working system

`openarm_gym/control/bimanual_tray.py` — `BimanualTrayCarry`:

- `grasp()` approaches, closes, and captures the grasp transform per hand.
- `command_tray(pos, quat)` — **the coordination step**: one tray pose in, both
  arms' targets out, so the arms can never fight through the tray.
- `balance_law(ball_xy, ball_vxy)` — the PD law, **callable on its own** so a
  vision estimator can drive exactly the same arithmetic.
- `nudge_ball()`, `tray_quat()` for a yaw goal, `observe()` for cameras,
  plus torque control, latency, backlash and dynamics randomization.

`openarm_gym/control/tray_task.py` — `random_carry()` plans and **one rollout
driver**, `run_plan`, shared by collection, evaluation and video. That sharing is
deliberate: a separate loop per purpose is how an evaluation quietly stops
measuring the task the data came from.

**Measured, on 12 randomized carries with disturbances:**

| controller | survived | peak ball offset |
|---|---|---|
| classical | **12/12** | 3.6 cm |
| level (no balancing) | 1/12 | 9.1 cm |

Grip 42 N per hand; the tray tracks a commanded carry to ~2 mm.

### Sim-to-real, measured ✅

`openarm_gym/realism.py` — all off by default.

| condition | survived | grip | tracking |
|---|---|---|---|
| position control (baseline) | 6/6 | 41.7 N | 3.3 mm |
| **torque control** | 6/6 | 41.7 N | 3.3 mm |
| torque, 40% of rated motor torque | 6/6 | 30.3 N | 3.2 mm |
| torque, 25% of rated | **3/6** | 18.9 N | 5.5 mm |
| latency 1 step (20 ms) | 6/6 | 41.7 N | 4.1 mm |
| latency 2 steps (40 ms) | 6/6 | 41.7 N | 4.6 mm |
| **backlash 0.5°** | 5/6 | **17.8 N** | 6.2 mm |
| backlash 1.5° | 5/6 | 22.0 N | 11.3 mm |
| torque + 20 ms + 0.5° | 6/6 | 26.9 N | 6.4 mm |

Three things worth keeping from that table:

- **Torque control reproduces position control exactly.** That is the point: the
  interface changed, not the behaviour, so the torque limit is now explicit.
- **Grip force is set by the torque limit**, because commanding SHUT against a
  post wider than the closed jaws saturates. 100% → 41.7 N, 40% → 30.3 N,
  25% → 18.9 N. A commanded grip force is therefore available today.
- **Backlash hurts far more than latency.** 0.5° of slack more than halves the
  grip, because the grip comes from commanding *past* the closed limit and
  backlash eats exactly that overtravel. 40 ms of latency costs almost nothing.

### The task is harder now ✅

Mid-episode ball shoves, randomized multi-waypoint carries, and a **tray yaw
goal** (measured: 7.8° achieved of 8.0° commanded, ball kept). Yaw composes with
balancing because rotating about gravity adds no in-plane acceleration — but only
if composed as `yaw * tilt`, since the balance law's axes are the tray's own.

### Renderers ✅

`scripts/render_tray.py` writes an MP4 from any scene camera, or drives the live
`mujoco.viewer`. Both run the same `run_plan`. It takes `--randomize-layout` and
`--handle-detector` now, so a clip can show the visual grasp.

`scripts/render_demo.py` is the one to show someone. It flies the same plan twice
-- same layout, same grasp, same shove, balancer off and on -- and lays the two
side by side with the numbers drawn over them: ball offset against the edge of
the tray, commanded tilt, grip force, the handle estimate's error at the moment
of the grasp, and the policy's own 84x112 camera images beside a top-down
schematic of where it thinks the ball is against where it is. It runs under a
named sim-to-real condition and prints that condition on the video, so nothing
about which knobs were on has to be taken on trust.

The reason it exists is worth keeping: shown a single clip of two arms holding a
tray, a viewer cannot tell a working system from a lucky one, and none of the
quantities this project actually controls are visible in it.
`results/tray_demo.mp4` is a current one.

---

## The vision policy — solved

**12 of 12 on unseen randomized plans, matching the classical controller**, and
holding the ball closer to centre while doing it. It sees only two 84x112 camera
images and joint angles; the ball's state is never given to it.

| controller | survived | peak ball | tilt err vs classical | ball sight err |
|---|---|---|---|---|
| classical (true ball state) | 12/12 | 3.6 cm | — | — |
| **vision, estimator + differenced velocity** | **12/12** | **3.2 cm** | 11.9 mrad | 4.5 mm |
| vision, regressed velocity head | 11/12 | 4.9 cm | 21.0 mrad | 5.8 mm |
| vision, end-to-end tilt regression | 11/12 | 4.5 cm | 18.7 mrad | 5.3 mm |
| level (no balancing) | 1/12 | 9.1 cm | 45.6 mrad | — |

Trained on a mix of **23 rollout demonstrations + 40 DAgger episodes + 4 800
uniformly-sampled detector frames**, 0.16M parameters, ~7 s/epoch on the 3060.

### What actually fixed it, in order of how much it mattered

The closed-loop sight error is the number to watch — it went 72.7 mm -> 29.1 ->
9.2 -> **4.5 mm**, and survival followed it:

1. **Stop training the estimator on the controller's own rollouts.**
   `scripts/record_tray_detector.py` samples the ball's position and velocity
   **uniformly** over the tray instead. In ``estimator`` mode the network is a
   *detector* — "where is the ball" — and that is static perception, so there is
   no reason to collect it from trajectories. Measured coverage: RMS ball offset
   **60 mm uniform against 10.5 mm from demonstrations**. Covariate shift cannot
   act on a distribution no policy generated.
2. **DAgger** — drive with the policy, label with the expert, keep the failures.
   40 episodes, RMS offset 28.8 mm. This alone took it from 2/12 to 11/12.
3. **Action noise** in demonstrations (execute perturbed, label clean), 0.02 rad.
4. **Estimate the state, do not regress the tilt.** The estimator arm beats the
   end-to-end arm on the same weights, and the gains stay exact by construction.
5. **Difference the position estimate for damping** rather than regressing
   velocity, which is not observable at this resolution (see below).

### Why the obvious diagnoses were wrong

Each was ruled out by direct measurement, and two of them cost a wasted cycle:

- **Not accuracy.** Noise injected into the classical loop: it survives 16 mm of
  position noise, 50 mm/s of velocity noise, a 20 mm constant bias (5-6/6 each).
  The tolerance was never the problem.
- **Not gain shrinkage alone.** The end-to-end tilt regressor does output about
  half the label magnitude, but fixing that by estimating state did not by itself
  rescue the closed loop.
- **The velocity head genuinely learned nothing** — 11.7 mm/s RMSE against
  12.7 mm/s for a predictor that ignores the images. Observability, not training:
  the ball moves 1.3 mm between stacked frames, a **tenth of a pixel** at the
  overhead camera's ~10 mm/px.
- **Validation ball error is not diagnostic.** The narrow early dataset scored
  4.2 mm largely by predicting the centre, and failed 0/12; a later model scored
  a worse 12.1 mm and did better. **Closed-loop survival is the metric.**

## What P0 bought

Three things had to be true before "vision works, 12/12" survived a reviewer.
All three are now measured. Nothing here was a research problem; the machinery
existed and defaulted to off.

### P0.1 The vision policy under the sim-to-real layer ✅

`scripts/evaluate_tray_realism.py` flies the classical controller and the vision
policy under each named condition in `openarm_gym/control/tray_eval.py`, on
identical plans and identical seeds. 12 plans per row, 43 minutes for the table:

| condition | classical | **vision** | cls peak | vis peak | sight err | grip |
|---|---|---|---|---|---|---|
| clean | 12/12 | **12/12** | 3.6 cm | 3.2 cm | 4.5 mm | 34.0 N |
| camera noise (gain 0.04, read 2.0) | 12/12 | **12/12** | 3.6 cm | 3.3 cm | 5.1 mm | 34.0 N |
| torque control | 12/12 | **12/12** | 3.6 cm | 3.2 cm | 4.6 mm | 34.0 N |
| torque, 40% of rated | 12/12 | **12/12** | 3.4 cm | 3.2 cm | 4.7 mm | 34.3 N |
| latency 1 step (20 ms) | 12/12 | **12/12** | 3.2 cm | 3.3 cm | 4.8 mm | 34.4 N |
| latency 2 steps (40 ms) | 12/12 | **12/12** | 3.6 cm | 4.1 cm | 5.1 mm | 34.1 N |
| backlash 0.5° | 12/12 | **12/12** | 4.6 cm | 4.9 cm | 5.7 mm | 29.1 N |
| **backlash 1.5°** | 11/12 | **8/12** | 6.1 cm | 7.2 cm | 7.1 mm | 20.6 N |
| randomized dynamics | 12/12 | **12/12** | 5.7 cm | 3.6 cm | 4.7 mm | 33.9 N |
| **all of them at once** | 12/12 | **12/12** | 4.6 cm | 4.5 cm | 5.8 mm | 28.5 N |

Read it in this order:

- **The expected drop mostly did not happen.** The policy holds 12/12 under
  every condition except the largest backlash, including all of them together
  (camera noise, torque at 40% of rated, 20 ms of latency, 0.5° of backlash and
  randomized dynamics in one episode). That is the strong result the plan said
  to look for, and it is strong because the classical controller does not do
  better: it also holds, and its ball wanders as far.
- **Backlash is still the one that bites**, at 1.5° rather than 0.5°. It unloads
  the grip (34 N -> 20.6 N) and the tray then tracks worse, which costs the
  *classical* controller an episode too. On this checkpoint vision fell further,
  8/12 against 11/12 -- but see P0.3b below: that gap was the estimator having
  seen one tray layout, not backlash, and it disappears entirely once that is
  fixed.
- **Camera noise costs 0.6 mm of sight error and no episodes.** The sensor model
  was already in the training augmentation, so this measures that the
  augmentation worked, not that the task is easy.
- **Randomized dynamics helps the vision row look better than classical** (3.6 cm
  against 5.7 cm peak). That is not the policy being cleverer: the classical
  controller reads the ball's true state and reacts to everything the draw does,
  where the estimator's slight smoothing rides over it.

### P0.2 The grasp is visual ✅

`grasp()` no longer has to be told where the handles are.
`openarm_gym/policies/handle_vision.py` is a 0.08M-parameter detector over one
frame from each of the two cameras, taken at the arms' retracted pose, returning
both handle posts in world coordinates. `grasp(handles=...)` and `run_plan`'s
`handle_source` take it; the privileged path is still there, behind the same
argument, so the two are directly comparable.

12 randomized layouts, same seeds for both rows:

| grasp | clean | held | ball kept | shove | worst shove | grip | handle err |
|---|---|---|---|---|---|---|---|
| privileged (reads the simulator) | 12/12 | 12/12 | 12/12 | 5.9 mm | 6.3 mm | 43.0 N | — |
| **vision (two 84x112 images)** | **12/12** | **12/12** | 11/12 | **5.6 mm** | 6.2 mm | 43.1 N | **0.7 mm** |

- The gate was tray displacement under 10 mm. It is 5.6 mm, which is the
  privileged approach's own number -- the shove is set by the approach geometry,
  not by the estimate, and 0.7 mm of estimate error does not move it.
- The one lost ball is not a grasp failure: the grasp held for all 12, and the
  0.7 mm difference in where the tray ends up is enough to tip one already
  borderline carry. The privileged row loses different episodes on other seeds.
- It was **not hard**, and the reason is worth keeping: the handles do not move
  before they are grasped, so this is one static frame, not a control problem.
  Training is 3600 uniformly-sampled layouts, 40 epochs, about a minute on the
  3060, validation error 1.05 mm.
- The honest caveat is the frame. The detector outputs *world* coordinates,
  which is well-posed only because both cameras are bolted to the world in the
  scene. On hardware that is the extrinsic calibration, and it is where this
  piece's sim-to-real gap lives.

### P0.3 The layout is randomized ✅

`BimanualTrayCarry.randomize_layout` jitters the tray's starting pose -- position
and yaw -- and re-seats the ball on it. It runs in `run_plan`'s new `on_reset`
window, after the reset and before the grasp, which is the only place it can go:
the grasp captures a transform, so the layout has to be settled before the jaws
close.

- **Classical: 12/12** over randomized layouts, mean peak ball 3.5 cm against
  3.6 cm on the fixed layout, tracking 3.2 mm. The gate.
- **Vision: 12/12**, peak 3.6 cm -- but on the checkpoint trained over one fixed
  layout its sight error rose from 4.5 mm to **8.0 mm**. The loop still held,
  because 8 mm is well inside the tolerance noise injection measured (it survives
  16 mm of position error), but the number could not be left standing. It was
  retrained; see below.

### P0.3b Retrained over randomized layouts ✅

The whole pipeline was re-run with the layout randomizer on -- 20 surviving
demonstrations, 5 400 detector frames spread over 90 different tray poses, then
a 40-episode DAgger round driven by the resulting policy, then a retrain. Both
recorders take `--randomize-layout` now; the detector one also takes `--layouts`,
since a grasp captures a transform and the tray is carried by it, so the layout
cannot change without letting go and re-grasping.

| checkpoint | randomized layouts | fixed layout |
|---|---|---|
| trained on one layout | 12/12, sight err **8.0 mm** | 12/12, sight err **4.5 mm** |
| **trained over layouts** | 12/12, sight err **4.7 mm** | 12/12, sight err 5.3 mm |

The gap closes: 8.0 -> 4.7 mm, back to what the fixed-layout policy achieved on
the scene it was specialised for. The 0.8 mm it gives up on that one scene is the
honest price of not being specialised. Uniform ball coverage survived the change
(60 mm RMS offset in the new detector files, against 60 mm before), which is the
property that made the estimator work in the first place.

**Then the whole sim-to-real sweep was re-run on it, layouts randomized** -- the
honest version of the P0.1 table:

| condition | classical | **vision** | cls peak | vis peak | sight err | grip |
|---|---|---|---|---|---|---|
| clean | 12/12 | **12/12** | 3.7 cm | 3.3 cm | 4.7 mm | 33.8 N |
| camera noise | 12/12 | **12/12** | 3.7 cm | 3.3 cm | 5.7 mm | 33.5 N |
| torque control | 12/12 | **12/12** | 3.8 cm | 3.3 cm | 4.7 mm | 34.0 N |
| torque, 40% of rated | 12/12 | **12/12** | 3.4 cm | 3.3 cm | 4.6 mm | 34.3 N |
| latency 1 step | 12/12 | **12/12** | 3.4 cm | 3.6 cm | 4.9 mm | 33.8 N |
| latency 2 steps | 12/12 | **12/12** | 3.5 cm | 3.9 cm | 5.1 mm | 33.5 N |
| backlash 0.5° | 12/12 | **12/12** | 3.8 cm | 3.8 cm | 4.6 mm | 29.2 N |
| **backlash 1.5°** | 11/12 | **11/12** | 5.7 cm | 5.4 cm | 5.9 mm | 18.0 N |
| randomized dynamics | 12/12 | **12/12** | 4.1 cm | 3.4 cm | 4.6 mm | 33.8 N |
| all of them at once | 12/12 | **12/12** | 3.9 cm | 4.2 cm | 6.0 mm | 29.5 N |

**This corrects a conclusion drawn from the first table.** There, vision fell to
8/12 at 1.5° of backlash against the classical controller's 11/12, and the drop
was read as perception and mechanics compounding -- the grip unloads, the tray
tracks worse, and the estimator's error climbs. The first two are real and still
visible (grip 18 N, peak ball 5.7 cm for *both* controllers). The third was not
backlash at all: it was a detector that had only ever seen one tray pose, and
with that fixed the vision row matches the classical row exactly, 11/12, at
5.9 mm of sight error. **Backlash is a mechanical limit, not a perception one.**
The lesson generalises past this project: an estimator narrow in one dimension
degrades under perturbations that have nothing to do with that dimension, and it
looks exactly like the perturbation being the cause.

## Measured facts — do not re-derive these

### Grasp geometry

- The jaws close along world y and reach along **+x** at the home pose.
- The pads sit ~0.125 m ahead of the wrist site — *not* the 0.155 m
  `GRASP_OFFSET` the peg experts use.
- `PoseController.solve` drives the wrist **site**, not the fingertips. Convert
  with `site = pad - tool_dir * PAD_OFFSET`.
- Jaw aperture ~9 mm closed, ~85 mm open.
- IK reaches handle pairs at x = 0.30–0.35 with residual ~0; it degrades to 0.03
  by x = 0.40. **That is the static reach, and the carry envelope is the other
  way round.** Flying randomized plans with the tray started at an x offset:
  pulling it *closer* than about 0.33 makes the arms fold up, tracking goes from
  3 mm to 7–16 mm, and carries start throwing the ball (2 of 3 lost at -0.04).
  Pushing it out to 0.40 is free and tracking slightly improves. `LayoutRanges`
  therefore leans outward, x in (-0.015, +0.04). Do not "fix" it to be
  symmetric.

### The approach used to bulldoze the tray — fixed

Reaching in at the handle centre put the gripper's **`ee_base_link` — the palm,
not the fingers** — against the tray's top face and shoved the whole tray
**32.5 mm** forward before the jaws reached the posts. `grasp()` now reaches in
20 mm higher and settles down: **5.8 mm**.

The honest cost: worst-case finger contacts fell from 4 to 2 (grip 83.6 → 89.5 N).
The extra two were a *consequence of the fault* — the shove drove the post deep
between the fingers. Reaching deliberately further in reproduces contacts and
displacement together in a 1:1 trade, so there is nothing to recover. Two
contacts on a flat face is still form closure; one contact on a cylinder is what
previously failed.

### Why the first grasp attempts failed

- The home keyframe puts the closed jaws inside the handle posts; starting there
  throws the tray. `reset()` teleports the arms clear.
- **Smooth cylindrical handles cannot be held.** Box handles with flat faces
  fixed it: worst-case contacts 1 → 9, grip 47 → 83 N. Two apparent *control*
  problems vanished with it. Suspect mechanics before control.

### Disturbance recovery limits

The classical balancer recovers a **0.30 m/s** shove along the tray's short axis
and **0.20 m/s** along the long one. The long axis is harder despite being longer,
because the carry is already moving that way. Disturbances are specified as a
**velocity kick in m/s**, not an impulse: the ball weighs 2.73 g, so a
plausible-looking 0.004 N·s impulse is a 1.5 m/s launch.

### Gravity compensation is mandatory; MuJoCo's own flag does not work

`model.body_gravcomp` was measured to have **no effect**. Use the manual form on
the **arm DOFs only** — including the tray and ball makes the ball weightless:

```python
data.qfrc_applied[:18] = data.qfrc_bias[:18]
```

### GPU physics: MJX is the wrong backend

JAX has no CUDA wheels for native Windows. MJX `put_model` takes >10 minutes on
the arm's collision meshes; disabling them drops it to 0.9 s, but disabling
collisions and welding the tray is what made that attempt a toy. With collisions
and a real grasp, **`mujoco_warp` (mjlab, in WSL2) is the GPU path**. There is no
`madrona_mjx` anywhere, so there is no batched GPU renderer.

### The visual-randomization RNG bug (fixed earlier, commit f5e8140)

`_randomize_domain` consumed draws before `_reset_task`, so enabling visual
randomization moved every object for a given seed. Visual randomization now draws
from its own spawned generator. The same trap is why **action noise takes a
separate generator from camera noise** in `run_plan`: action noise moves the tray
and camera noise does not, so sharing a stream would make enabling the camera
sensor model change the physics.

---

## Environment

Both venvs are Python 3.12.13 AMD64 and either runs everything. **Prefer the
project venv**, `D:\openarm mujoco\openarm_mujoco\.venv` — mujoco 3.13.0, torch
2.11.0+cu128 (CUDA works on the RTX 3060), lerobot 0.6.1.

- **mujoco >= 3.13 is required.** The 3.12 contact solver drops the clean
  `peg_socket` expert from 6/6 to 1/6.
- `export HF_HUB_OFFLINE=1` before anything touching lerobot datasets.
- **RAM is the binding constraint, not disk.** 16 GB total, ~3–5 GB available.
  VS Code holds ~3.8 GB and must stay open. The vision dataset is held in RAM as
  uint8 on purpose: 28 episodes of two 84×112 cameras is ~700 MB as uint8 and
  2.8 GB as float32. `num_workers=0` in the trainer is deliberate — Windows
  spawns rather than forks, so each worker would copy the whole dataset.
- Data lives on `D:\openarm_data\`. Disk: D: ~38 GB free.
- **Never bound a collection run with `timeout`** — it corrupts the metadata
  parquet and has cost two datasets.
- Lint: `uvx ruff@0.14.11 check openarm_gym/` — 4 findings remain, all
  pre-existing (`experts/base.py`, `recording.py`). `ruff format` is *not* clean
  on the pre-existing code either, so do not reformat only new files.

---

## Things already tried that measured worse — do not retry

- **Welding the tray and disabling arm collisions** to make MJX fast. Deletes two
  of the five requirements.
- **`regulate_grip` as a grip booster.** Commanding SHUT already saturates; there
  is no headroom to squeeze, only to open. Under torque control the limit sets the
  force directly, which is the better tool.
- **Acceleration feedforward in `balance_tilt`.** With form closure the reactive
  PD tracks the full distance alone, and the extra tilt authority costs grip.
- **Regressing the tilt end to end.** Predictions come out at roughly **half** the
  label magnitude — MSE regression to the mean — and half the loop gain is a
  different controller. Estimating the state keeps the gains exact.
- **Regressing the ball's velocity.** Not observable at this resolution; see above.
- **Reaching further in to recover finger contacts.** Trades 1:1 against tray
  displacement.
- **Randomizing the layout symmetrically about the authored tray pose.** The
  near side is where the carry fails; see the envelope note above.
- **More disturbances to widen the data distribution.** The expert recentres too
  fast for it to matter (10.5 → 13 mm RMS).
- **Commanding a 10 cm tray step in one control step** — the jerk throws the ball.
- From the earlier direction: rotating the tool with the valve lever; chasing the
  puck continuously; overshooting the valve's angle; tracking an object's live
  pose through the final approach; a top-down grasp on `peg_socket`.

## Non-obvious facts worth keeping

- The home keyframe puts both grippers at their closed geometric limit, where the
  pads touch; a hand landing a hair inside jams. `env.reset` cracks both jaws open.
- **The right gripper opens toward negative values, the left toward positive.**
- `camera_names` defaults to `()`, and `run_plan` renders only when something
  consumes the pixels. Never render when measuring success rates.
- A torchcodec DLL load failure printed at import is **benign**.
- Scripted-expert rates quoted as "6/6" are clean-environment rates: no action
  noise, no waypoint noise, `domain_randomize=False`.
- `ctx_read` in `full` mode drops comment-only lines — it is not verbatim. Use
  `cat` when exact text matters.
