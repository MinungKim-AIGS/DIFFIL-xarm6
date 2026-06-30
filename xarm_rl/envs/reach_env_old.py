"""xArm6 reach task with optional domain randomization + safe-zone penalty.

Observation (state-only):
    [joint_pos(6), joint_vel(6), ee_pos(3), target_pos(3), (target - ee)(3)]  -> 21

Action: 6 joint deltas in [-1, 1], scaled by self.action_scale (rad/step).

Reward:
    -dist - 0.001 * ||a||^2
    + 10  (success bonus: dist < 3 cm)
    - SAFE_ZONE_PENALTY  (if TCP outside the real xArm6 safe zone)

Domain randomization (when domain_rand=True):
    - cube/link mass scale (±20 %)            -> we use only mass on link3..link6 (arm-dominant)
    - joint friction loss   (0.5 .. 1.5x)
    - actuator kp / kv      (±30 %)
    - observation noise on joint_pos (±0.5 deg = 0.0087 rad)
    - action latency 0..2 steps (random per episode, fixed in episode)
"""
from __future__ import annotations

import collections
import numpy as np
import mujoco
from gymnasium import spaces

from .base_env import XArm6BaseEnv, HOME_QPOS, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH


# Workspace box — subset of real xArm6 safe zone
WORKSPACE_LOW  = np.array([0.25, -0.30, 0.30], dtype=np.float32)
WORKSPACE_HIGH = np.array([0.55,  0.30, 0.55], dtype=np.float32)

# Real xArm6 safe zone (meters, base frame). Real spec: x:0..570, y:-540..550, z:180..600 mm.
SAFE_LOW  = np.array([0.00, -0.54, 0.18], dtype=np.float32)
SAFE_HIGH = np.array([0.57,  0.55, 0.60], dtype=np.float32)

SUCCESS_DIST = 0.03
ACTION_DIM = 6
SAFE_ZONE_PENALTY = 1.0      # per-step penalty when TCP exits safe zone

# Domain randomization ranges
DR_MASS_RANGE     = (0.8, 1.2)
DR_FRICTION_RANGE = (0.5, 1.5)
DR_PD_RANGE       = (0.7, 1.3)
DR_OBS_NOISE_STD  = 0.0087     # ~0.5 deg
DR_LATENCY_RANGE  = (0, 2)     # action delay in steps


class XArm6ReachEnv(XArm6BaseEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        action_scale: float = 0.05,
        domain_rand: bool = False,
    ):
        super().__init__("scene_reach.xml", action_scale=action_scale, render_mode=render_mode)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        self.target_pos = np.zeros(3, dtype=np.float32)
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]

        # ---- domain randomization state ----
        self.domain_rand = domain_rand
        # Snapshot nominal physical parameters so we can scale relative to them every reset.
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_dof_frictionloss = self.model.dof_frictionloss.copy()
        # Actuator gains: position actuator gainprm[0] is kp; biasprm[1] is kv (sign already negative in MJCF).
        self._nominal_gainprm = self.model.actuator_gainprm.copy()
        self._nominal_biasprm = self.model.actuator_biasprm.copy()
        self._obs_noise_std = 0.0
        self._action_latency = 0
        self._action_queue: collections.deque = collections.deque()

    # ---- domain randomization helpers ----
    def _apply_domain_rand(self):
        if not self.domain_rand:
            self._obs_noise_std = 0.0
            self._action_latency = 0
            self._action_queue.clear()
            # Reset to nominal in case prior episode had DR on (e.g. shared model)
            self.model.body_mass[:] = self._nominal_body_mass
            self.model.dof_frictionloss[:] = self._nominal_dof_frictionloss
            self.model.actuator_gainprm[:] = self._nominal_gainprm
            self.model.actuator_biasprm[:] = self._nominal_biasprm
            return

        rng = self.np_random
        # Mass scaling on arm links (skip world / target / etc.). We scale links 1..6 (body ids found by name).
        for i in range(1, 7):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
            if bid >= 0:
                self.model.body_mass[bid] = self._nominal_body_mass[bid] * rng.uniform(*DR_MASS_RANGE)

        # Friction loss on the 6 arm joints
        for jname in [f"joint{i+1}" for i in range(6)]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                dof = self.model.jnt_dofadr[jid]
                self.model.dof_frictionloss[dof] = self._nominal_dof_frictionloss[dof] * rng.uniform(*DR_FRICTION_RANGE)

        # PD gains on the 6 arm position actuators (kp in gainprm[:,0]; kv in biasprm[:,2] for position actuator)
        for i, aid in enumerate(self.arm_act_ids):
            kp_scale = rng.uniform(*DR_PD_RANGE)
            kv_scale = rng.uniform(*DR_PD_RANGE)
            self.model.actuator_gainprm[aid, 0] = self._nominal_gainprm[aid, 0] * kp_scale
            # SB3 position actuators: biasprm = [0, -kp, -kv]; we scale -kv (index 2) by kv_scale
            self.model.actuator_biasprm[aid, 1] = self._nominal_biasprm[aid, 1] * kp_scale
            self.model.actuator_biasprm[aid, 2] = self._nominal_biasprm[aid, 2] * kv_scale

        self._obs_noise_std = float(DR_OBS_NOISE_STD)
        self._action_latency = int(rng.integers(DR_LATENCY_RANGE[0], DR_LATENCY_RANGE[1] + 1))
        self._action_queue.clear()

    def _sample_target(self):
        return self.np_random.uniform(WORKSPACE_LOW, WORKSPACE_HIGH).astype(np.float32)

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

        noise = self.np_random.uniform(-0.05, 0.05, size=6).astype(np.float32)
        self.set_arm_qpos(HOME_QPOS + noise)
        self.apply_arm_action(HOME_QPOS + noise)
        self.apply_gripper(1.0)

        self.target_pos = self._sample_target()
        self._set_target(self.target_pos)

        # Pre-fill action queue with the current commanded angles so latency works on step 1
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

        # Action latency: push new command, pop oldest to actually apply
        if self.domain_rand and self._action_latency > 0:
            self._action_queue.append(target_q)
            apply_q = self._action_queue.popleft()
        else:
            apply_q = target_q

        self.apply_arm_action(apply_q)
        self.step_sim()

        ee = self.get_ee_pos()
        dist = float(np.linalg.norm(self.target_pos - ee))

        reward = -dist - 0.001 * float(np.sum(action ** 2))

        # Safe-zone penalty (continuous, applied each step the TCP is outside).
        in_safe = bool(np.all(ee >= SAFE_LOW) and np.all(ee <= SAFE_HIGH))
        if not in_safe:
            reward -= SAFE_ZONE_PENALTY

        success = dist < SUCCESS_DIST
        if success:
            reward += 10.0

        terminated = bool(success)
        truncated = False
        info = {
            "distance": dist,
            "is_success": float(success),
            "in_safe_zone": float(in_safe),
            "action_latency": self._action_latency,
        }
        return self._get_obs(), reward, terminated, truncated, info
