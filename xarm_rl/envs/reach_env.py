"""xArm6 Reach task — d3il / manipulation_envs compatible.

Observation (21-dim):
    [joint_pos(6), joint_vel(6), ee_pos(3), target_pos(3), (target - ee)(3)]

Action: 6 joint deltas in [-1, 1], scaled by action_scale (rad/step).

Reward (matches manipulation_envs ReachEnv):
    reward = -||ee - target||  (reward_near only, no ctrl penalty)

Episode termination:
    done=False always — episode ends only via time limit (matches manipulation_envs).

Image collection:
    get_ims() -> [past_frames, H, W, C] uint8  (d3il frame-stack interface)
    render_mode="rgb_array" also supported for external collection.

Domain randomization (optional, xarm6-specific):
    mass / friction / PD gains / obs noise / action latency
"""
from __future__ import annotations

import collections
import numpy as np
import mujoco
from gymnasium import spaces

from .base_env import XArm6BaseEnv, HOME_QPOS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH


# Single FIXED goal: the farthest point that is clearly visible in the front
# camera (marker fully in frame, not clipped at the edge) AND reachable/successful
# within the 200-step budget. Verified: 0.315 m from home, IK residual ~0.
# (To restore multi-goal sampling, change _sample_target back to random.)
GOAL_FIXED = np.array([0.48, -0.30, 0.42], dtype=np.float32)
WORKSPACE_LOW  = GOAL_FIXED.copy()
WORKSPACE_HIGH = GOAL_FIXED.copy()

# Real xArm6 safe zone
SAFE_LOW  = np.array([0.00, -0.54, 0.18], dtype=np.float32)
SAFE_HIGH = np.array([0.57,  0.55, 0.60], dtype=np.float32)

# Selectable front-style camera presets (defined in assets/scene_reach.xml).
# Approx downward elevation:  front ~28deg, ob_b ~47, ob_c ~62, ob_d ~80 (near top).
# Choose via  gym.make("XArm6Reach-v0", render_camera="ob_c")  or the per-camera
# env ids registered in envs/__init__.py (XArm6Reach-obC-v0, ...).
CAMERA_CHOICES = ("front", "ob_b", "ob_c", "ob_d", "topdown")

ACTION_DIM = 6
SAFE_ZONE_PENALTY = 1.0

# Domain randomization ranges
DR_MASS_RANGE     = (0.8, 1.2)
DR_FRICTION_RANGE = (0.5, 1.5)
DR_PD_RANGE       = (0.7, 1.3)
DR_OBS_NOISE_STD  = 0.0087
DR_LATENCY_RANGE  = (0, 2)


class XArm6ReachEnv(XArm6BaseEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        action_scale: float = 0.05,
        domain_rand: bool = False,
        past_frames: int = 4,
        img_size: tuple = (64, 64),
        render_camera: str = "front",
        action_rate_penalty: float = 0.0,
    ):
        if render_camera not in CAMERA_CHOICES:
            raise ValueError(
                f"render_camera must be one of {CAMERA_CHOICES}, got {render_camera!r}"
            )
        super().__init__(
            "scene_reach.xml",
            action_scale=action_scale,
            render_mode=render_mode,
            past_frames=past_frames,
            img_size=img_size,
            render_camera=render_camera,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.target_pos = np.zeros(3, dtype=np.float32)
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]

        # End-effector position at the home pose (used to reject near-home goals)
        self.set_arm_qpos(HOME_QPOS)
        mujoco.mj_forward(self.model, self.data)
        self._home_ee = self.get_ee_pos().copy()

        # Domain randomization
        self.domain_rand = domain_rand
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_dof_frictionloss = self.model.dof_frictionloss.copy()
        self._nominal_gainprm = self.model.actuator_gainprm.copy()
        self._nominal_biasprm = self.model.actuator_biasprm.copy()
        self._obs_noise_std = 0.0
        self._action_latency = 0
        self._action_queue: collections.deque = collections.deque()

        # Optional smoothness shaping: penalize |a_t - a_{t-1}| (default off).
        self.action_rate_penalty = float(action_rate_penalty)
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # ---- domain randomization ----
    def _apply_domain_rand(self):
        if not self.domain_rand:
            self._obs_noise_std = 0.0
            self._action_latency = 0
            self._action_queue.clear()
            self.model.body_mass[:] = self._nominal_body_mass
            self.model.dof_frictionloss[:] = self._nominal_dof_frictionloss
            self.model.actuator_gainprm[:] = self._nominal_gainprm
            self.model.actuator_biasprm[:] = self._nominal_biasprm
            return

        rng = self.np_random
        for i in range(1, 7):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
            if bid >= 0:
                self.model.body_mass[bid] = self._nominal_body_mass[bid] * rng.uniform(*DR_MASS_RANGE)

        for jname in [f"joint{i+1}" for i in range(6)]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                dof = self.model.jnt_dofadr[jid]
                self.model.dof_frictionloss[dof] = self._nominal_dof_frictionloss[dof] * rng.uniform(*DR_FRICTION_RANGE)

        for i, aid in enumerate(self.arm_act_ids):
            kp_scale = rng.uniform(*DR_PD_RANGE)
            kv_scale = rng.uniform(*DR_PD_RANGE)
            self.model.actuator_gainprm[aid, 0] = self._nominal_gainprm[aid, 0] * kp_scale
            self.model.actuator_biasprm[aid, 1] = self._nominal_biasprm[aid, 1] * kp_scale
            self.model.actuator_biasprm[aid, 2] = self._nominal_biasprm[aid, 2] * kv_scale

        self._obs_noise_std = float(DR_OBS_NOISE_STD)
        self._action_latency = int(rng.integers(DR_LATENCY_RANGE[0], DR_LATENCY_RANGE[1] + 1))
        self._action_queue.clear()

    def _sample_target(self):
        # Single fixed goal (GOAL_FIXED), deterministic across episodes.
        return GOAL_FIXED.copy()

    def _set_target(self, pos):
        self.data.mocap_pos[self._target_mocap_id] = pos

    def _get_obs(self) -> np.ndarray:
        q = self.get_arm_qpos()
        qd = self.get_arm_qvel()
        if self._obs_noise_std > 0:
            q = q + self.np_random.normal(0.0, self._obs_noise_std, size=q.shape).astype(np.float32)
        ee = self.get_ee_pos()
        diff = self.target_pos - ee
        return np.concatenate([q, qd, ee, self.target_pos, diff]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_rand()
        self._reset_buffer()  # reset frame stack on episode start
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        noise = self.np_random.uniform(-0.05, 0.05, size=6).astype(np.float32)
        self.set_arm_qpos(HOME_QPOS + noise)
        self.apply_arm_action(HOME_QPOS + noise)
        self.apply_gripper(1.0)

        self.target_pos = self._sample_target()
        self._set_target(self.target_pos)

        if self.domain_rand and self._action_latency > 0:
            init_cmd = HOME_QPOS + noise
            for _ in range(self._action_latency):
                self._action_queue.append(init_cmd.copy())

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        current_q = self.get_arm_qpos()
        target_q = current_q + action * self.action_scale
        target_q = np.clip(target_q, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

        if self.domain_rand and self._action_latency > 0:
            self._action_queue.append(target_q)
            apply_q = self._action_queue.popleft()
        else:
            apply_q = target_q

        self.apply_arm_action(apply_q)
        self.step_sim()

        ee = self.get_ee_pos()
        dist = float(np.linalg.norm(self.target_pos - ee))

        # Reward matches manipulation_envs ReachEnv: reward_near only
        reward = -dist

        # Optional safe-zone penalty
        in_safe = bool(np.all(ee >= SAFE_LOW) and np.all(ee <= SAFE_HIGH))
        if not in_safe:
            reward -= SAFE_ZONE_PENALTY

        # Optional action-rate (smoothness) penalty — encourages smooth, real-safe
        # policies that are less likely to trip the controller's overspeed guard.
        if self.action_rate_penalty > 0.0:
            reward -= self.action_rate_penalty * float(np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()

        # done=False always (time limit only), matching manipulation_envs
        terminated = False
        truncated = False
        info = {
            "distance": dist,
            "is_success": float(dist < 0.03),
            "in_safe_zone": float(in_safe),
        }
        return self._get_obs(), reward, terminated, truncated, info
