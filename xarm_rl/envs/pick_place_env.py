"""xArm6 pick-and-place: grasp cube, transport to target position."""
from __future__ import annotations

import numpy as np
import mujoco
from gymnasium import spaces

from .base_env import XArm6BaseEnv, HOME_QPOS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH


# Sampling regions — subset of real xArm6 safe zone (meters, base frame)
# Real safe zone (mm): x:0..570, y:-540..550, z:180..600
CUBE_X_RANGE = (0.35, 0.55)
CUBE_Y_RANGE = (-0.25, 0.25)
CUBE_Z       = 0.43          # table_top(0.40) + cube_half(0.022)

TARGET_X_RANGE = (0.30, 0.55)
TARGET_Y_RANGE = (-0.30, 0.30)
TARGET_Z_RANGE = (0.45, 0.58)

# Hard safe-zone clip
SAFE_LOW  = np.array([0.00, -0.54, 0.18], dtype=np.float32)
SAFE_HIGH = np.array([0.57,  0.55, 0.60], dtype=np.float32)

GRASP_HEIGHT_THRESH = 0.46    # cube z above this means "lifted"
SUCCESS_DIST = 0.05           # cube to target distance for success
ACTION_DIM = 7                # 6 joint deltas + 1 gripper


class XArm6PickPlaceEnv(XArm6BaseEnv):
    def __init__(self, render_mode: str | None = None, action_scale: float = 0.05):
        super().__init__("scene_pick_place.xml", action_scale=action_scale, render_mode=render_mode)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        # obs: q(6) + qd(6) + ee_pos(3) + grip_state(1) + cube_pos(3) + cube_quat(4)
        #    + target_pos(3) + (cube - ee)(3) + (target - cube)(3) = 32
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

        self.target_pos = np.zeros(3, dtype=np.float32)
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_qpos_addr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]
        self._cube_qvel_addr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]
        self._grip_state = 1.0  # +1 open, -1 closed

    # ---- helpers ----
    def get_cube_pose(self):
        addr = self._cube_qpos_addr
        pos = np.array(self.data.qpos[addr:addr+3], dtype=np.float32)
        quat = np.array(self.data.qpos[addr+3:addr+7], dtype=np.float32)
        return pos, quat

    def set_cube_pose(self, pos, quat=(1, 0, 0, 0)):
        addr = self._cube_qpos_addr
        self.data.qpos[addr:addr+3] = pos
        self.data.qpos[addr+3:addr+7] = quat
        # zero velocity
        v = self._cube_qvel_addr
        self.data.qvel[v:v+6] = 0

    def _sample_cube_pos(self):
        x = self.np_random.uniform(*CUBE_X_RANGE)
        y = self.np_random.uniform(*CUBE_Y_RANGE)
        return np.array([x, y, CUBE_Z], dtype=np.float32)

    def _sample_target(self):
        x = self.np_random.uniform(*TARGET_X_RANGE)
        y = self.np_random.uniform(*TARGET_Y_RANGE)
        z = self.np_random.uniform(*TARGET_Z_RANGE)
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        q = self.get_arm_qpos()
        qd = self.get_arm_qvel()
        ee = self.get_ee_pos()
        cube_pos, cube_quat = self.get_cube_pose()
        diff_ee_cube = cube_pos - ee
        diff_cube_target = self.target_pos - cube_pos
        return np.concatenate([
            q, qd, ee, [self._grip_state], cube_pos, cube_quat,
            self.target_pos, diff_ee_cube, diff_cube_target,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        noise = self.np_random.uniform(-0.05, 0.05, size=6).astype(np.float32)
        self.set_arm_qpos(HOME_QPOS + noise)
        self.apply_arm_action(HOME_QPOS + noise)
        self.apply_gripper(1.0)
        self._grip_state = 1.0

        cube_pos = self._sample_cube_pos()
        self.set_cube_pose(cube_pos)

        self.target_pos = self._sample_target()
        self.data.mocap_pos[self._target_mocap_id] = self.target_pos

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Arm
        current_q = self.get_arm_qpos()
        target_q = current_q + action[:6] * self.action_scale
        target_q = np.clip(target_q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        self.apply_arm_action(target_q)

        # Gripper: smooth toward action[6]
        self._grip_state = float(np.clip(self._grip_state + 0.2 * (action[6] - self._grip_state), -1.0, 1.0))
        self.apply_gripper(self._grip_state)

        self.step_sim()

        # Reward shaping
        ee = self.get_ee_pos()
        cube_pos, _ = self.get_cube_pose()
        d_ee_cube = float(np.linalg.norm(cube_pos - ee))
        d_cube_target = float(np.linalg.norm(self.target_pos - cube_pos))
        lifted = cube_pos[2] > GRASP_HEIGHT_THRESH

        reward = (
            -d_ee_cube                       # approach cube
            + (2.0 if lifted else 0.0)       # bonus once lifted
            - (d_cube_target if lifted else 0.0)  # only penalize transport once lifted
            - 0.001 * float(np.sum(action ** 2))
        )

        success = lifted and d_cube_target < SUCCESS_DIST
        if success:
            reward += 50.0

        # Fail if cube falls off table
        cube_fell = cube_pos[2] < 0.38
        terminated = bool(success or cube_fell)
        if cube_fell:
            reward -= 5.0

        info = {
            "d_ee_cube": d_ee_cube,
            "d_cube_target": d_cube_target,
            "lifted": float(lifted),
            "is_success": float(success),
        }
        return self._get_obs(), reward, terminated, False, info
