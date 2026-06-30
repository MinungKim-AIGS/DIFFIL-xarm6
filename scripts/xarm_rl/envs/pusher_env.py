"""xArm6 Pusher task — d3il / manipulation_envs compatible.

The arm pushes an object (cylinder) toward a goal position on a table surface.

Observation (27-dim):
    [joint_pos(6), joint_vel(6), ee_pos(3), obj_pos(3), goal_pos(3), (obj-goal)(3), (ee-obj)(3)]

Action: 6 joint deltas in [-1, 1], scaled by action_scale (rad/step).

Reward (matches manipulation_envs PusherEnv):
    reward = 1.25 * reward_dist
    where reward_dist = -||obj_pos - goal_pos||

Episode termination:
    done=False always — episode ends only via time limit (matches manipulation_envs).

Image collection:
    get_ims() -> [past_frames, H, W, C] uint8  (d3il frame-stack interface)
    render_mode="rgb_array" also supported for external collection.

Note:
    Requires scene_pusher.xml with:
      - a free-joint body named "object" (cylinder on table)
      - a mocap body named "goal"
      - ee_site on the end-effector
"""
from __future__ import annotations

import numpy as np
import mujoco
from gymnasium import spaces

from .base_env import XArm6BaseEnv, HOME_QPOS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH


# Table surface area where object and goal are placed
OBJ_AREA_LOW  = np.array([0.25, -0.20], dtype=np.float32)  # x, y on table
OBJ_AREA_HIGH = np.array([0.45,  0.20], dtype=np.float32)
GOAL_AREA_LOW  = np.array([0.30, -0.25], dtype=np.float32)
GOAL_AREA_HIGH = np.array([0.50,  0.25], dtype=np.float32)
TABLE_Z = 0.42   # fixed z height of objects on table surface

ACTION_DIM = 6
OBS_DIM = 27


class XArm6PusherEnv(XArm6BaseEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        action_scale: float = 0.05,
        past_frames: int = 4,
        img_size: tuple = (64, 64),
    ):
        super().__init__(
            "scene_pusher.xml",
            action_scale=action_scale,
            render_mode=render_mode,
            past_frames=past_frames,
            img_size=img_size,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # Object body id (free joint)
        self._obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._obj_jnt_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        ]

        # Goal mocap body
        self._goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self._goal_mocap_id = self.model.body_mocapid[self._goal_body_id]

        self.goal_pos = np.zeros(3, dtype=np.float32)

    # ---- helpers ----
    def get_obj_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self._obj_body_id], dtype=np.float32)

    def get_goal_pos(self) -> np.ndarray:
        return np.array(self.data.mocap_pos[self._goal_mocap_id], dtype=np.float32)

    def _set_obj_pos(self, xy: np.ndarray):
        adr = self._obj_jnt_qposadr
        self.data.qpos[adr:adr+3] = [xy[0], xy[1], TABLE_Z]
        self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]  # unit quaternion

    def _set_goal_pos(self, xy: np.ndarray):
        self.data.mocap_pos[self._goal_mocap_id] = [xy[0], xy[1], TABLE_Z]

    def _sample_obj_pos(self) -> np.ndarray:
        return self.np_random.uniform(OBJ_AREA_LOW, OBJ_AREA_HIGH).astype(np.float32)

    def _sample_goal_pos(self) -> np.ndarray:
        return self.np_random.uniform(GOAL_AREA_LOW, GOAL_AREA_HIGH).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        q   = self.get_arm_qpos()       # 6
        qd  = self.get_arm_qvel()       # 6
        ee  = self.get_ee_pos()         # 3
        obj = self.get_obj_pos()        # 3
        goal = self.get_goal_pos()      # 3
        obj_to_goal = goal - obj        # 3
        ee_to_obj   = obj - ee          # 3
        return np.concatenate([q, qd, ee, obj, goal, obj_to_goal, ee_to_obj]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._reset_buffer()

        # Reset arm to home
        noise = self.np_random.uniform(-0.05, 0.05, size=6).astype(np.float32)
        self.set_arm_qpos(HOME_QPOS + noise)
        self.apply_arm_action(HOME_QPOS + noise)
        self.apply_gripper(-1.0)  # closed gripper for pushing

        # Randomize object and goal positions (ensure they don't overlap)
        obj_xy = self._sample_obj_pos()
        goal_xy = self._sample_goal_pos()
        # Resample if too close
        while np.linalg.norm(obj_xy - goal_xy) < 0.08:
            goal_xy = self._sample_goal_pos()

        self._set_obj_pos(obj_xy)
        self._set_goal_pos(goal_xy)
        self.goal_pos = np.array([goal_xy[0], goal_xy[1], TABLE_Z], dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        current_q = self.get_arm_qpos()
        target_q = np.clip(
            current_q + action * self.action_scale,
            JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
        )
        self.apply_arm_action(target_q)
        self.step_sim()

        ee  = self.get_ee_pos()
        obj = self.get_obj_pos()
        goal = self.get_goal_pos()

        # Reward matches manipulation_envs PusherEnv
        vec_1 = obj - ee           # ee → object
        vec_2 = obj - goal         # object → goal
        reward_near = -np.sum(np.abs(vec_1))
        reward_dist = -np.sum(np.abs(vec_2))
        reward = 1.25 * reward_dist  # same as manipulation_envs

        # done=False always, matching manipulation_envs
        terminated = False
        truncated = False
        info = {
            "dist_obj_goal": float(np.linalg.norm(vec_2)),
            "dist_ee_obj":   float(np.linalg.norm(vec_1)),
            "reward_near":   float(reward_near),
            "reward_dist":   float(reward_dist),
        }
        return self._get_obs(), reward, terminated, truncated, info
