"""Base MuJoCo environment for xArm6 tasks.

Joint-delta action space, state-based observations. Designed so the same
policy interface can later be ported to the real xArm6 via XArmAPI.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"

# xArm6 joint limits (rad) — match xarm6.xml actuator ctrlrange
JOINT_LIMITS_LOW  = np.array([-6.283, -2.059, -3.927, -6.283, -1.693, -6.283], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([ 6.283,  2.094,  0.191,  6.283,  3.142,  6.283], dtype=np.float32)

# Home pose: arm pointing up & slightly forward
HOME_QPOS = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)


class XArm6BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_xml: str,
        action_scale: float = 0.05,   # rad per step per joint
        control_dt: float = 0.02,     # 50 Hz control
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / scene_xml))
        self.data = mujoco.MjData(self.model)

        self.action_scale = action_scale
        self.control_dt = control_dt
        self.frame_skip = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.render_mode = render_mode
        self._viewer = None

        # Arm joint ids and actuator ids
        self.arm_joint_names = [f"joint{i+1}" for i in range(6)]
        self.arm_qpos_addrs = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.arm_joint_names
        ]
        self.arm_qvel_addrs = [
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.arm_joint_names
        ]
        self.arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act{i+1}")
            for i in range(6)
        ]
        self.grip_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_l"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_grip_r"),
        ]
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        # Subclasses set observation_space and action_space
        self.action_space = None
        self.observation_space = None

    # ---- helpers ----
    def get_arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.arm_qpos_addrs], dtype=np.float32)

    def get_arm_qvel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self.arm_qvel_addrs], dtype=np.float32)

    def get_ee_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site_id], dtype=np.float32)

    def set_arm_qpos(self, qpos: np.ndarray):
        for a, q in zip(self.arm_qpos_addrs, qpos):
            self.data.qpos[a] = q

    def apply_arm_action(self, target_qpos: np.ndarray):
        for aid, q in zip(self.arm_act_ids, target_qpos):
            self.data.ctrl[aid] = q

    def apply_gripper(self, open_close: float):
        """open_close in [-1, 1]; +1 = fully open, -1 = fully closed."""
        # Map [-1, 1] -> finger ranges
        # left finger: 0 (closed) .. 0.04 (open)
        # right finger: -0.04 (open) .. 0 (closed)  [mirrored]
        s = (open_close + 1.0) * 0.5  # [0, 1]
        self.data.ctrl[self.grip_act_ids[0]] =  0.04 * s
        self.data.ctrl[self.grip_act_ids[1]] = -0.04 * s

    def step_sim(self):
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            # Off-screen rendering omitted for simplicity in v0
            return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
