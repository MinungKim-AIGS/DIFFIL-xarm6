"""RealRobotEnv — the TARGET environment for DIFF-IL, backed by the real xArm6.

Implements the *old-gym* + visual interface that DIFF-IL's Sampler expects:
    reset() -> obs(21)
    step(act) -> (obs, rew, done, info)            # 4-tuple, old gym
    get_ims() -> [past_frames, H, W, C] uint8       # d3il frame stack
    action_space.sample(), action_space.shape, seed()

It REUSES real_reach_collector (open_arm, threaded RealSenseCamera, obs builder,
safe-zone + fault guards, reset_to_home, GOAL_FIXED). All safety stays local so
the actor is safe even if the network/learner drops.

Used by:
  - actor_node.py  with the learned policy  -> collects B^TL (online target)
  - a random policy                          -> collects B^TR (target random)
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

# import the sibling collector module (scripts/real_reach_collector.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import real_reach_collector as rrc


class ActionSpace:
    """Minimal Box([-1,1]^dim) with the bits Sampler uses."""
    def __init__(self, dim: int = rrc.ACTION_DIM, seed: int = 0):
        self.shape = (dim,)
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self._dim).astype(np.float32)

    def seed(self, s):
        self._rng = np.random.default_rng(s)


class RealRobotEnv:
    def __init__(self, ip: str, front_serial: str | None = None, dry_run: bool = False,
                 action_scale: float = rrc.ACTION_SCALE, control_hz: float = rrc.CONTROL_HZ,
                 action_filter: float = 0.3, max_steps: int = 200, home_jitter: float = 0.05,
                 past_frames: int = rrc.PAST_FRAMES, img_size=rrc.IMG_SIZE, seed: int = 0):
        self.arm = rrc.open_arm(ip, dry_run=dry_run)
        self.camera = rrc.DummyCamera() if (dry_run or not front_serial) \
            else rrc.RealSenseCamera(front_serial)
        self.action_scale = float(action_scale)
        self.control_dt = 1.0 / float(control_hz)
        self.action_filter = float(action_filter)
        self.max_steps = int(max_steps)
        self.home_jitter = float(home_jitter)
        self.past_frames = past_frames
        self.img_size = img_size
        self.target = rrc.GOAL_FIXED.copy()
        self.action_space = ActionSpace(rrc.ACTION_DIM, seed)
        self.rng = np.random.default_rng(seed)

        self._prev_q = None
        self._prev_action = np.zeros(rrc.ACTION_DIM, dtype=np.float32)
        self._frames = None
        self._ct = 0
        self._stopped = False   # safety/fault latch for the current episode

    # ---- gym-ish API ----
    def seed(self, s):
        self.rng = np.random.default_rng(s)
        self.action_space.seed(s)

    def reset(self):
        noise = self.rng.uniform(-self.home_jitter, self.home_jitter, size=6).astype(np.float32)
        try:
            self.arm.clean_error(); self.arm.clean_warn(); self.arm.motion_enable(enable=True)
        except Exception:
            pass
        self.arm.set_mode(0); self.arm.set_state(0); time.sleep(0.2)
        home = (rrc.HOME_QPOS + noise).astype(np.float32)
        self.arm.set_servo_angle(angle=home.tolist(), speed=0.35, is_radian=True, wait=True)
        time.sleep(0.3)
        self.arm.set_mode(1); self.arm.set_state(0); time.sleep(0.2)

        self._prev_q = None
        self._prev_action[:] = 0.0
        self._frames = None
        self._ct = 0
        self._stopped = False
        return self._obs()

    def step(self, act: np.ndarray):
        act = np.clip(np.asarray(act, np.float32), -1.0, 1.0)
        # action low-pass (smoothness -> avoids overspeed/collision trips)
        act = self.action_filter * self._prev_action + (1.0 - self.action_filter) * act
        self._prev_action = act.copy()

        t0 = time.time()
        q = self._read_q()
        ee = self._read_ee()

        done = False
        info = {"is_success": float(np.linalg.norm(self.target - ee) < 0.03)}

        # hard safe-zone guard
        if not (bool(np.all(ee >= rrc.SAFE_LOW)) and bool(np.all(ee <= rrc.SAFE_HIGH))):
            print(f"  [SAFETY] TCP {np.round(ee,3)} left safe zone - ending episode")
            self._stopped = True
            done = True
        else:
            target_q = np.clip(q + act * self.action_scale, rrc.JOINT_LIMITS_LOW, rrc.JOINT_LIMITS_HIGH)
            ret = self.arm.set_servo_angle_j(angles=target_q.tolist(), is_radian=True)
            if ret != 0:
                print(f"  warn: set_servo_angle_j ret={ret} - ending episode")
                self._stopped = True; done = True
            else:
                err, _ = rrc.arm_fault(self.arm)
                if err != 0:
                    print(f"  [FAULT] error_code={err} - clearing + ending episode")
                    self.arm.clean_error(); self.arm.clean_warn()
                    self._stopped = True; done = True

        # pace control loop
        dt = time.time() - t0
        if dt < self.control_dt:
            time.sleep(self.control_dt - dt)

        self._ct += 1
        if self._ct >= self.max_steps:
            done = True

        obs = self._obs()
        rew = rrc.reach_reward(self._read_ee(), self.target)   # placeholder; learner recomputes
        return obs, rew, bool(done), info

    def get_ims(self) -> np.ndarray:
        """Capture current frame, advance the past_frames stack, return [T,H,W,C]."""
        im = self.camera.get_frame()
        if self._frames is None:
            self._frames = np.stack([im] * self.past_frames, axis=0)
        else:
            self._frames = np.roll(self._frames, -1, axis=0)
            self._frames[-1] = im
        return self._frames.astype(np.uint8)

    def close(self):
        try:
            self.arm.set_mode(0); self.arm.set_state(0); self.arm.disconnect()
        except Exception:
            pass
        self.camera.stop()

    # ---- helpers ----
    def _read_q(self):
        code, ang = self.arm.get_servo_angle(is_radian=True)
        return np.asarray(ang[:6], np.float32)

    def _read_ee(self):
        code, tcp = self.arm.get_position(is_radian=True)
        return np.asarray(tcp[:3], np.float32) / 1000.0

    def _obs(self):
        q = self._read_q()
        qd = np.zeros(6, np.float32) if self._prev_q is None \
            else ((q - self._prev_q) / self.control_dt).astype(np.float32)
        self._prev_q = q
        ee = self._read_ee()
        return rrc.build_reach_obs(q, qd, ee, self.target)
