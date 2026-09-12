---
license: apache-2.0
tags:
  - robotics
  - manipulation
  - bimanual
  - mujoco
  - visuomotor
  - sim2real
library_name: pytorch
pipeline_tag: robotics
---

# Tray carry — vision policies for two OpenArm v2 arms

Two 7-DOF arms grasp a tray by its handle posts and carry it while keeping a
loose ball from rolling off, in MuJoCo, with collisions enabled and nothing
welded. Both halves of the task are driven by camera images.

Code, measurements and the full write-up:
**https://github.com/ahmedsleem109/openarm-tray-carry**

## What is here

| file | what it is | parameters |
|---|---|---|
| `policy.pt` | ball state estimator — two 84×112 cameras, two stacked frames | 0.16 M |
| `detector.pt` | handle post detector — one frame per camera, run once per episode | 0.08 M |
| `tray_ball_estimator.onnx` | the same estimator, ONNX opset 17 | — |
| `tray_handle_detector.onnx` | the same detector, ONNX opset 17 | — |
| `manifest.json` | input layout, normalisation statistics, output scales | — |

`manifest.json` matters: proprioception is standardised with the training set's
mean and standard deviation, and the networks emit O(1) numbers that become
metres and radians only after multiplying by the scales recorded there.

## How they are used

The estimator does **not** output an action. It estimates the ball's position in
the tray frame, and a classical PD law turns that into a tray tilt:

```
accel = -kp * ball_xy - kd * ball_vxy      kp = 12.0, kd = 4.5
pitch = accel_x / g                        roll = -accel_y / g
```

That split is deliberate and was measured. Regressing the tilt end-to-end from
the same weights produces outputs at roughly **half** the required magnitude —
MSE regression toward the mean — and half the loop gain is a different
controller. Estimating state keeps the gains exact by construction.

The velocity channel the network outputs is **not** used by the controller: at
84×112 the ball moves about a tenth of a pixel between stacked frames, and the
regressed velocity was measured at 11.7 mm/s RMSE against 12.7 mm/s for a
predictor that ignores the images entirely. The controller differences the
position estimate instead.

## Measured

12 unseen randomized carries, randomized tray layouts, identical plans and seeds
across rows. `classical` reads the ball's true state out of the simulator and is
the ceiling, not a competitor. The floor — carrying the tray level with no
balancing — is 1/12.

| condition | classical | vision | ball offset | sight error | grip |
|---|---|---|---|---|---|
| clean | 12/12 | **12/12** | 3.3 cm | 4.7 mm | 33.8 N |
| camera-noise | 12/12 | **12/12** | 3.3 cm | 5.7 mm | 33.5 N |
| torque | 12/12 | **12/12** | 3.3 cm | 4.7 mm | 34.0 N |
| torque-40 | 12/12 | **12/12** | 3.3 cm | 4.6 mm | 34.3 N |
| latency-1 | 12/12 | **12/12** | 3.6 cm | 4.9 mm | 33.8 N |
| latency-2 | 12/12 | **12/12** | 3.9 cm | 5.1 mm | 33.5 N |
| backlash-0.5 | 12/12 | **12/12** | 3.8 cm | 4.6 mm | 29.2 N |
| backlash-1.5 | 11/12 | **11/12** | 5.4 cm | 5.9 mm | 18.0 N |
| dynamics | 12/12 | **12/12** | 3.4 cm | 4.6 mm | 33.8 N |
| all | 12/12 | **12/12** | 4.2 cm | 6.0 mm | 29.5 N |

Grasping from `detector.pt` instead of from simulator state: **12/12** clean
grasps, tray displaced 5.6 mm during the approach against 5.9 mm
for the privileged path, handle estimate error 0.7 mm.

## Training, in one paragraph

The lesson that made it work: in estimator mode the network is a **detector**,
so it has to be trained on the state distribution you want it accurate over —
which is *uniform*, not whatever a stabilising controller visits. Behaviour
cloning on the expert's own rollouts scored 0/12 while being accurate to 4.6 mm
on the expert's own states, because a good balancer keeps the ball centred and
the resulting data is one picture over and over. Training data here is
uniformly-sampled ball positions over the tray face, plus DAgger episodes, plus
demonstrations — all over randomized tray layouts.

## Limits

Every number above is a **simulation** number. There is no real arm. The cameras
are fixed to the world, so the handle detector predicts world coordinates and
extrinsic calibration is assumed away — on hardware that is a real step and it
is where the transfer gap lives. Rendering is MuJoCo's, with a gain-and-read
noise sensor model and no photorealism.

## Credit

The OpenArm v2 robot model is [Enactic's](https://github.com/enactic/openarm_mujoco),
used unmodified. Apache-2.0.
