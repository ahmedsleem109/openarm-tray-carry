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

"""Damped-least-squares differential IK for the OpenArm bimanual model.

Python port of ``web/ik.js`` (``PoseController``). The solver runs on a
kinematics-only scratch ``MjData`` so it never disturbs the simulation it is
driving: ``mj_kinematics`` + ``mj_comPos`` give site poses, ``mj_jacSite``
gives the 6xN site Jacobian, and each iteration takes a damped least-squares
step restricted to that arm's seven joint DOFs.

Poses are ``(pos, quat)`` in the world frame, targeting the
``left_ee_control_point`` / ``right_ee_control_point`` sites. Quaternions are
MuJoCo's ``[w, x, y, z]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")

# The tool axis: the fingers extend along the control-point site's local -Z.
# A pose whose rotation is identity therefore points the gripper straight down.
TOOL_AXIS_LOCAL = np.array([0.0, 0.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamilton product ``a * b`` of two ``[w, x, y, z]`` quats."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a unit quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the quaternion for intrinsic Z-Y-X (yaw, pitch, roll) angles."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def quat_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Return the world-frame rotation vector taking ``q_current`` to ``q_target``.

    The result pairs with the rotational rows of ``mj_jacSite``, which are also
    expressed in the world frame.
    """
    err = quat_mul(q_target, quat_conj(q_current))
    if err[0] < 0.0:  # shortest path
        err = -err
    vec = err[1:]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros(3)
    return vec * (2.0 * np.arctan2(norm, err[0]) / norm)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Return the quaternion of a flat or 3x3 rotation matrix."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


def _solve_damped(jac: np.ndarray, err: np.ndarray, lambda2: float) -> np.ndarray:
    """Return the damped least-squares step ``J^T (J J^T + lambda2 I)^-1 err``."""
    jjt = jac @ jac.T
    jjt[np.diag_indices_from(jjt)] += lambda2
    return jac.T @ np.linalg.solve(jjt, err)


@dataclass(frozen=True)
class _Arm:
    """Index bundle for one arm: its site, joint DOFs and qpos slots."""

    site_id: int
    dof: np.ndarray  # (7,) DOF indices for joint1..joint7
    qpos: np.ndarray  # (7,) qpos indices for joint1..joint7
    lower: np.ndarray  # (7,) joint lower limits
    upper: np.ndarray  # (7,) joint upper limits

    @classmethod
    def build(cls, model: mujoco.MjModel, side: str) -> _Arm:
        """Resolve every index this arm needs by name."""
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point"
        )
        if site_id < 0:
            raise ValueError(f"Site '{side}_ee_control_point' not found in model")

        joint_ids = []
        for i in range(1, 8):
            name = f"openarm_{side}_joint{i}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in model")
            joint_ids.append(jid)
        joint_ids = np.asarray(joint_ids, dtype=np.intp)

        return cls(
            site_id=site_id,
            dof=model.jnt_dofadr[joint_ids].astype(np.intp),
            qpos=model.jnt_qposadr[joint_ids].astype(np.intp),
            lower=model.jnt_range[joint_ids, 0].copy(),
            upper=model.jnt_range[joint_ids, 1].copy(),
        )


class PoseController:
    """Differential IK over the two OpenArm end-effector control points."""

    def __init__(self, model: mujoco.MjModel) -> None:
        """Build the index tables and the scratch state used for solving."""
        self._model = model
        self._scratch = mujoco.MjData(model)
        self._arms = {side: _Arm.build(model, side) for side in SIDES}
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        origin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "arm_origin")
        if origin_id < 0:
            raise ValueError("Site 'arm_origin' not found in model")
        self._origin_id = origin_id

    def arm_qpos_indices(self, side: str) -> np.ndarray:
        """Return the (7,) qpos indices for one arm's joints."""
        return self._arms[side].qpos.copy()

    def arm_dof_indices(self, side: str) -> np.ndarray:
        """Return the (7,) DOF indices for one arm's joints."""
        return self._arms[side].dof.copy()

    def origin_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return the world pose of the ``arm_origin`` site.

        Teleoperation poses (and the scenes' home poses) are expressed in this
        frame, which rides the cell's lifter.
        """
        return (
            data.site_xpos[self._origin_id].copy(),
            mat_to_quat(data.site_xmat[self._origin_id]),
        )

    def local_to_world(
        self, data: mujoco.MjData, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map a pose from the ``arm_origin`` frame into the world frame."""
        origin_pos, origin_quat = self.origin_pose(data)
        rotated = np.empty(3)
        mujoco.mju_rotVecQuat(rotated, np.asarray(pos, dtype=np.float64), origin_quat)
        return origin_pos + rotated, quat_mul(origin_quat, quat)

    def pose(self, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the current world pose of one arm's control point."""
        site_id = self._arms[side].site_id
        return (
            data.site_xpos[site_id].copy(),
            mat_to_quat(data.site_xmat[site_id]),
        )

    def grasp_point(
        self, data: mujoco.MjData, side: str, tool_offset: float
    ) -> np.ndarray:
        """Return the world point ``tool_offset`` metres down the tool axis."""
        site_id = self._arms[side].site_id
        rot = data.site_xmat[site_id].reshape(3, 3)
        return data.site_xpos[site_id] + rot @ (TOOL_AXIS_LOCAL * tool_offset)

    def solve(
        self,
        data: mujoco.MjData,
        side: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        iters: int = 50,
        tol: float = 1e-4,
        max_step: float = 0.2,
        lambda2: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """Solve for the seven joint angles reaching a world-frame pose.

        Seeds from ``data``'s current configuration and iterates on the scratch
        state, so calling this every control step tracks a moving target
        smoothly rather than jumping between IK branches.

        Returns:
            ``(qpos, residual)`` where ``qpos`` is the (7,) joint solution and
            ``residual`` is the final 6-D pose error norm.

        """
        arm = self._arms[side]
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0.0

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)
        err = np.zeros(6)
        residual = np.inf

        for _ in range(iters):
            mujoco.mj_kinematics(self._model, scratch)
            mujoco.mj_comPos(self._model, scratch)

            err[:3] = target_pos - scratch.site_xpos[arm.site_id]
            err[3:] = quat_error(target_quat, mat_to_quat(scratch.site_xmat[arm.site_id]))
            residual = float(np.linalg.norm(err))
            if residual < tol:
                break

            mujoco.mj_jacSite(self._model, scratch, self._jacp, self._jacr, arm.site_id)
            jac = np.vstack((self._jacp[:, arm.dof], self._jacr[:, arm.dof]))

            step = _solve_damped(jac, err, lambda2)
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step:
                step *= max_step / step_norm

            scratch.qpos[arm.qpos] = np.clip(
                scratch.qpos[arm.qpos] + step, arm.lower, arm.upper
            )

        return scratch.qpos[arm.qpos].copy(), residual
