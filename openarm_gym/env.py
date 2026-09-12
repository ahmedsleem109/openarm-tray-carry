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

"""Gymnasium base environment for the OpenArm v2 bimanual MuJoCo scenes.

The action and observation layouts deliberately mirror ``openarm_driver``'s
16-value convention (``right[0:8] + left[8:16]``, each arm being seven joints
plus a gripper) so a policy trained here can be handed to the real robot
without a remapping layer. The index mapping itself comes from the upstream
``openarm_mujoco.v2.JointResolver``.

Observations carry joint torque alongside position and velocity. OpenArm's
quasi-direct-drive joints are backdrivable and report torque on the real
hardware, so contact-rich tasks can be learned with a force channel rather
than from vision alone.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from openarm_mujoco.v2 import JointResolver, asset_path

from .ik import PoseController
from .vision import CameraRig

# Driver convention: right arm first, then left, gripper last within each arm.
DRIVER_DIM = 16

#: Jaw opening, in radians, that every episode starts from. Enough to clear the
#: fingertip self-contact at the keyframe's closed pose; small enough that the
#: start pose still matches the authored "home" keyframe for every other joint.
FINGER_REST = 0.03
_SEGMENTS = ("right", "left")


class OpenArmEnv(gym.Env):
    """Base environment wrapping one OpenArm v2 MJCF scene.

    Subclasses set :attr:`SCENE` and implement :meth:`_reset_task`,
    :meth:`compute_reward` and :meth:`is_success`.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    #: Scene path relative to the repository's ``v2/`` asset root.
    SCENE: str = "cell/cell.xml"
    #: Keyframe used as the start pose.
    KEYFRAME: str = "home"
    #: Episode cap, in control steps.
    MAX_STEPS: int = 500

    def __init__(
        self,
        *,
        control_hz: float = 50.0,
        camera_names: tuple[str, ...] = (),
        image_size: tuple[int, int] = (240, 320),
        domain_randomize: bool = False,
        gravity_compensation: bool = True,
        torque_noise: float = 0.05,
        camera_gain_noise: float = 0.0,
        camera_read_noise: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        """Load the scene and build the action/observation spaces."""
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(asset_path(self.SCENE))
        self.data = mujoco.MjData(self.model)
        self.resolver = JointResolver(self.model)
        self.ik = PoseController(self.model)

        self.control_hz = control_hz
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.metadata = dict(self.metadata, render_fps=int(control_hz))
        self.domain_randomize = domain_randomize
        self.gravity_compensation = gravity_compensation
        # Real DM actuators infer torque from current and report it noisily.
        # A noiseless torque channel would make the force-conditioning ablation
        # measure an oracle that no hardware can supply.
        self.torque_noise = torque_noise
        self.render_mode = render_mode
        #: Per-episode natural-language task description, recorded with
        #: demonstrations and used as the instruction for language-conditioned
        #: policies. Tasks that vary their goal set this in ``_reset_task``.
        self.instruction = ""

        self.camera_names = tuple(camera_names)
        self.image_size = image_size
        # One rig for every camera: a renderer per camera would cost a GL
        # context and a framebuffer each. Sensor noise is off by default so
        # recorded rates stay comparable with what came before.
        self.cameras = CameraRig(
            self.model,
            self.camera_names,
            image_size=image_size,
            gain_noise=camera_gain_noise,
            read_noise=camera_read_noise,
        )

        self._key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.KEYFRAME
        )
        if self._key_id < 0:
            raise ValueError(f"Keyframe '{self.KEYFRAME}' not found in {self.SCENE}")

        self._build_indices()
        self._build_spaces()

        self._steps = 0
        self._success_streak = 0
        self._home_ctrl = np.zeros(self.model.nu)

    # ---------------------------------------------------------------- setup

    def _build_indices(self) -> None:
        """Resolve the qpos/DOF/actuator slots behind the 16-value convention."""

        def joint_id(name: str) -> int:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            return jid

        def actuator_id(name: str) -> int:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator '{name}' not found in model")
            return aid

        joints: list[int] = []
        actuators: list[int] = []
        finger_joints: list[tuple[int, int]] = []
        for side in _SEGMENTS:
            joints += [joint_id(f"openarm_{side}_joint{i}") for i in range(1, 8)]
            joints.append(joint_id(f"openarm_{side}_finger_joint1"))
            actuators += [actuator_id(f"{side}_joint{i}_ctrl") for i in range(1, 8)]
            actuators.append(actuator_id(f"{side}_finger1_ctrl"))
            finger_joints.append(
                (
                    joint_id(f"openarm_{side}_finger_joint1"),
                    joint_id(f"openarm_{side}_finger_joint2"),
                )
            )

        joints_arr = np.asarray(joints, dtype=np.intp)
        self._qpos_idx = self.model.jnt_qposadr[joints_arr].astype(np.intp)
        self._dof_idx = self.model.jnt_dofadr[joints_arr].astype(np.intp)
        self._act_idx = np.asarray(actuators, dtype=np.intp)

        # Arm joints only, no fingers: the DOFs that get gravity compensation.
        self._gravcomp_dof = np.concatenate(
            [self.ik.arm_dof_indices(side) for side in _SEGMENTS]
        )

        # Each hand closes at joint zero and opens towards the far end of its
        # actuator range -- negative on the right, positive on the left.
        self._finger_qpos = np.array(
            [
                [self.model.jnt_qposadr[j1], self.model.jnt_qposadr[j2]]
                for j1, j2 in finger_joints
            ],
            dtype=np.intp,
        )
        finger_ranges = self.model.actuator_ctrlrange[
            [self._act_idx[7], self._act_idx[15]]
        ]
        self._finger_open_sign = np.sign(
            finger_ranges[np.arange(2), np.abs(finger_ranges).argmax(axis=1)]
        )

        # Torque sensor noise scales with each actuator's force range, so a
        # DM8009 shoulder and a DM4310 wrist get proportionate noise.
        ranges = self.model.actuator_forcerange[self._act_idx]
        self._torque_scale = np.maximum(np.abs(ranges).max(axis=1), 1.0)

        # The lifter is optional: only the cell scenes have one.
        self._lifter_act = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lifter_ctrl"
        )

    def _build_spaces(self) -> None:
        """Build the 16-value action space and the observation dict."""
        low = self.model.actuator_ctrlrange[self._act_idx, 0].astype(np.float32)
        high = self.model.actuator_ctrlrange[self._act_idx, 1].astype(np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        inf = np.full(DRIVER_DIM, np.inf, dtype=np.float32)
        obs: dict[str, spaces.Space] = {
            "agent_pos": spaces.Box(low=low, high=high, dtype=np.float32),
            "agent_vel": spaces.Box(low=-inf, high=inf, dtype=np.float32),
            "agent_torque": spaces.Box(low=-inf, high=inf, dtype=np.float32),
        }
        if self.camera_names:
            height, width = self.image_size
            obs["pixels"] = spaces.Dict(
                {
                    name: spaces.Box(
                        low=0, high=255, shape=(height, width, 3), dtype=np.uint8
                    )
                    for name in self.camera_names
                }
            )
        self.observation_space = spaces.Dict(obs)

    # ------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset to the scene keyframe, then apply task and domain randomization."""
        super().reset(seed=seed)

        # Visual randomization draws from its own stream. Sharing
        # ``self.np_random`` let camera and light jitter advance the generator
        # before ``_reset_task`` drew the object poses, so merely enabling
        # visual randomization resampled the *physical* layout for a given
        # seed. That silently confounds any domain-randomization comparison:
        # the two arms of the ablation see different layouts, not the same
        # layout under different rendering.
        self._visual_rng = np.random.default_rng(
            self.np_random.bit_generator.seed_seq.spawn(1)[0]
        )

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key_id)

        # Crack the jaws open before anything else. The home keyframe puts the
        # finger joints at exactly zero, which is the closed geometric limit --
        # the fingertip pads touch there. Numerical drift decides which side of
        # that boundary each hand lands on, and a hand that lands inside it
        # locks: the fingertip self-contact is stiff enough (solref 0.005) that
        # the 7 N*m finger actuator saturates against it and the jaws never open
        # again. The right hand happens to drift clear and the left happens to
        # drift into it, which is why only the left arm was affected.
        for row, sign in zip(self._finger_qpos, self._finger_open_sign):
            self.data.qpos[row] = sign * FINGER_REST

        # Hold the keyframe: position actuators need their targets synced to qpos,
        # or the arm snaps to zero on the first step.
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i, 0]
            if jid >= 0:
                self.data.ctrl[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        self._home_ctrl[:] = self.data.ctrl

        if self.domain_randomize:
            self._randomize_domain(self._visual_rng)
        self._reset_task(self.np_random)

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._success_streak = 0
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply a 16-value joint position target and advance one control step."""
        action = np.clip(
            np.asarray(action, dtype=np.float64),
            self.action_space.low,
            self.action_space.high,
        )
        self.resolver.set_ctrl(self.data.ctrl, action[0:8], "right")
        self.resolver.set_ctrl(self.data.ctrl, action[8:16], "left")

        for _ in range(self.n_substeps):
            if self.gravity_compensation:
                # Cancel gravity and Coriolis on the arm DOFs, as web/ik.js does.
                # The wrist actuators run at kp=30 against link masses that sag
                # centimetres without this, and the real arm's driver applies
                # the same feedforward term.
                self.data.qfrc_applied[self._gravcomp_dof] = self.data.qfrc_bias[
                    self._gravcomp_dof
                ]
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        success = self.is_success()
        self._success_streak = self._success_streak + 1 if success else 0
        # Require the success condition to hold briefly, so a peg passing
        # through the goal region on its way elsewhere does not count.
        terminated = self._success_streak >= 5
        truncated = self._steps >= self.MAX_STEPS

        return (
            self._observation(),
            float(self.compute_reward()),
            terminated,
            truncated,
            self._info(),
        )

    def render(self) -> np.ndarray | None:
        """Render the first requested camera as an RGB array."""
        if self.render_mode != "rgb_array" or not self.camera_names:
            return None
        return self._render_camera(self.camera_names[0])

    def close(self) -> None:
        """Release the offscreen renderer, if one was created."""
        self.cameras.close()

    # --------------------------------------------------------- observation

    def _observation(self) -> dict[str, Any]:
        """Assemble the observation dict in driver order."""
        obs: dict[str, Any] = {
            "agent_pos": self.data.qpos[self._qpos_idx].astype(np.float32),
            "agent_vel": self.data.qvel[self._dof_idx].astype(np.float32),
            # qfrc_actuator is the torque the motors apply at each joint -- the
            # quantity the real DM actuators report back over CAN.
            "agent_torque": self._sensed_torque(),
        }
        if self.camera_names:
            obs["pixels"] = {
                name: self._render_camera(name) for name in self.camera_names
            }
        return obs

    def _sensed_torque(self) -> np.ndarray:
        """Return joint torque as a sensor would report it.

        Gaussian noise proportional to each actuator's force range, so the
        force channel carries a realistic signal-to-noise ratio rather than
        simulator ground truth.
        """
        torque = self.data.qfrc_actuator[self._dof_idx]
        if self.torque_noise <= 0.0:
            return torque.astype(np.float32)
        scale = self.torque_noise * self._torque_scale
        return (torque + self.np_random.normal(0.0, scale)).astype(np.float32)

    def _render_camera(self, name: str) -> np.ndarray:
        """Render one named camera offscreen, through the sensor model.

        Drawing the sensor noise from ``self.np_random`` is deliberate but worth
        being explicit about, given the visual-randomization bug this project
        already paid for once: these draws happen in ``_observation``, which runs
        strictly *after* ``_reset_task``, so enabling camera noise cannot move
        the physical layout for a given seed. It does shift the torque channel's
        noise stream, which is an observation, not a state.

        The rig short-circuits when no noise is configured, so a camera-only
        change consumes no draws at all by default.
        """
        return self.cameras.render(self.data, name, self.np_random)

    def _info(self) -> dict[str, Any]:
        """Return the per-step info dict."""
        return {
            "is_success": self.is_success(),
            "steps": self._steps,
            "instruction": self.instruction,
        }

    # ------------------------------------------------------------ helpers

    def driver_position(self) -> np.ndarray:
        """Return the current joint configuration in the 16-value convention."""
        return self.data.qpos[self._qpos_idx].copy()

    def set_lifter(self, height: float) -> None:
        """Command the cell lifter, if the scene has one."""
        if self._lifter_act >= 0:
            low, high = self.model.actuator_ctrlrange[self._lifter_act]
            self.data.ctrl[self._lifter_act] = float(np.clip(height, low, high))

    def body_id(self, name: str) -> int:
        """Return a body id by name, raising if it is absent."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in {self.SCENE}")
        return bid

    def site_id(self, name: str) -> int:
        """Return a site id by name, raising if it is absent."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site '{name}' not found in {self.SCENE}")
        return sid

    def geom_id(self, name: str) -> int:
        """Return a geom id by name, raising if it is absent."""
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Geom '{name}' not found in {self.SCENE}")
        return gid

    def touching(self, geom_a: int, body_b: int) -> bool:
        """Return whether a geom is currently in contact with a body."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if (con.geom1 == geom_a and b2 == body_b) or (
                con.geom2 == geom_a and b1 == body_b
            ):
                return True
        return False

    # ------------------------------------------------------ randomization

    def _randomize_domain(self, rng: np.random.Generator) -> None:
        """Jitter camera poses and lighting, for sim-to-real robustness.

        Task-specific randomization (object poses, masses, friction) belongs in
        :meth:`_reset_task`.
        """
        for cam in range(self.model.ncam):
            self.model.cam_pos[cam] = self._nominal_cam_pos[cam] + rng.normal(
                0.0, 0.005, size=3
            )
            self.model.cam_fovy[cam] = self._nominal_cam_fovy[cam] * rng.uniform(
                0.97, 1.03
            )
        for light in range(self.model.nlight):
            self.model.light_diffuse[light] = np.clip(
                self._nominal_light_diffuse[light] * rng.uniform(0.7, 1.3), 0.0, 1.0
            )

    @property
    def _nominal_cam_pos(self) -> np.ndarray:
        """Return the scene's as-authored camera positions."""
        if not hasattr(self, "_cam_pos_backup"):
            self._cam_pos_backup = self.model.cam_pos.copy()
        return self._cam_pos_backup

    @property
    def _nominal_cam_fovy(self) -> np.ndarray:
        """Return the scene's as-authored camera FOVs."""
        if not hasattr(self, "_cam_fovy_backup"):
            self._cam_fovy_backup = self.model.cam_fovy.copy()
        return self._cam_fovy_backup

    @property
    def _nominal_light_diffuse(self) -> np.ndarray:
        """Return the scene's as-authored light intensities."""
        if not hasattr(self, "_light_backup"):
            self._light_backup = self.model.light_diffuse.copy()
        return self._light_backup

    # ---------------------------------------------------------- task hooks

    def _reset_task(self, rng: np.random.Generator) -> None:
        """Place task objects for a new episode."""
        raise NotImplementedError

    def compute_reward(self) -> float:
        """Return the shaped reward for the current state."""
        raise NotImplementedError

    def is_success(self) -> bool:
        """Return whether the task goal is currently satisfied."""
        raise NotImplementedError
