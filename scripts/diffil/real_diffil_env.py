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
                 past_frames: int = rrc.PAST_FRAMES, img_size=rrc.IMG_SIZE, seed: int = 0,
                 safe_margin: float = 0.03, max_faults: int = 10):
        self.arm = rrc.open_arm(ip, dry_run=dry_run)
        self.camera = rrc.DummyCamera() if (dry_run or not front_serial) \
            else rrc.RealSenseCamera(front_serial)
        self.action_scale = float(action_scale)
        self.control_dt = 1.0 / float(control_hz)
        self.action_filter = float(action_filter)
        self.max_steps = int(max_steps)
        self.home_jitter = float(home_jitter)
        self.safe_margin = float(safe_margin)   # steer home this far INSIDE the safe box
        self.max_faults = int(max_faults)       # per-episode fault recoveries before giving up
        self.past_frames = past_frames
        self.img_size = img_size
        self.target = rrc.GOAL_FIXED.copy()
        self.action_space = ActionSpace(rrc.ACTION_DIM, seed)
        self.rng = np.random.default_rng(seed)

        self._prev_q = None
        self._prev_action = np.zeros(rrc.ACTION_DIM, dtype=np.float32)
        self._frames = None
        self._ct = 0
        self._fault_count = 0   # fault recoveries used in the current episode
        self._stopped = False   # safety/fault latch for the current episode

    # ---- gym-ish API ----
    def seed(self, s):
        self.rng = np.random.default_rng(s)
        self.action_space.seed(s)

    # ---- robustness helpers ----
    def _ensure_ready(self, mode: int, retries: int = 4, settle: float = 0.4) -> bool:
        """Robustly bring the arm to a movable/ready state: clear residual faults,
        motion_enable, set mode+state, then VERIFY (err==0 and state not 'stop')
        before returning. Retries the sequence. Prevents the 'xArm is not ready to
        move' (code=1) race that happens when a command is issued before enable
        settles (typical right after a collision left the arm latched in error)."""
        ecode, state = 0, 2
        for _ in range(retries):
            try:
                self.arm.clean_error(); self.arm.clean_warn()
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(mode); self.arm.set_state(0)
            except Exception:
                pass
            time.sleep(settle)
            try:
                ecode, _w = rrc.arm_fault(self.arm)
            except Exception:
                ecode = 0
            try:
                _code, state = self.arm.get_state()
            except Exception:
                state = 2                      # dummy / dry-run -> treat as ready
            if ecode == 0 and (state is None or state < 4):
                return True
        print(f"  [ready] WARNING: arm not confirmed ready (err={ecode}, state={state})")
        return False

    def _recover(self):
        """Clear a collision/overspeed/servo fault and re-home WITHOUT ending the
        episode, so the episode keeps its FIXED length. Mirrors reset()'s home move:
        position mode -> blocking home -> back to servo-streaming mode."""
        try:
            self.arm.clean_error(); self.arm.clean_warn()
        except Exception:
            pass
        self._ensure_ready(mode=0)
        try:
            self.arm.set_servo_angle(angle=rrc.HOME_QPOS.tolist(), speed=0.35,
                                     is_radian=True, wait=True)
        except Exception:
            pass
        time.sleep(0.2)
        self._ensure_ready(mode=1)             # back to servo mode for the episode
        self._prev_action[:] = 0.0

    def reset(self):
        noise = self.rng.uniform(-self.home_jitter, self.home_jitter, size=6).astype(np.float32)
        home = (rrc.HOME_QPOS + noise).astype(np.float32)
        # robustly reach a ready state FIRST (clears any residual fault from a prior
        # collision) so the home command doesn't hit "xArm is not ready to move".
        self._ensure_ready(mode=0)
        self.arm.set_servo_angle(angle=home.tolist(), speed=0.35, is_radian=True, wait=True)
        time.sleep(0.3)
        self._ensure_ready(mode=1)             # servo-streaming mode for the episode

        self._prev_q = None
        self._prev_action[:] = 0.0
        self._frames = None
        self._ct = 0
        self._fault_count = 0
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
        # NON-TERMINATING safe-zone recovery (Option A): if the TCP is near/outside the
        # margin'd safe box, override the action with a home-ward move and KEEP the
        # episode running -> FIXED episode length even on safety events (mirrors the
        # collector's explore.SafetyRecovery). Only genuine hardware faults (below)
        # end the episode early. `act` becomes the actual EXECUTED action.
        lo = rrc.SAFE_LOW + self.safe_margin
        hi = rrc.SAFE_HIGH - self.safe_margin
        if not (bool(np.all(ee >= lo)) and bool(np.all(ee <= hi))):
            act = np.clip((rrc.HOME_QPOS - q) / self.action_scale, -1.0, 1.0)   # steer home
            print(f"  [SAFETY] TCP {np.round(ee,3)} near/left safe zone - steering home (episode continues)")

        # Expose the EXECUTED action (post EMA + safety override) so the actor stores
        # what the robot actually ran, not the policy's raw proposal — otherwise
        # off-policy SAC learns from (s, a_raw, s') while the robot ran a_executed.
        info = {"is_success": float(np.linalg.norm(self.target - ee) < 0.03),
                "applied_action": act.copy()}

        target_q = np.clip(q + act * self.action_scale, rrc.JOINT_LIMITS_LOW, rrc.JOINT_LIMITS_HIGH)
        ret = self.arm.set_servo_angle_j(angles=target_q.tolist(), is_radian=True)
        fault = False
        if ret != 0:
            print(f"  warn: set_servo_angle_j ret={ret}")
            fault = True
        else:
            err, _ = rrc.arm_fault(self.arm)
            if err != 0:
                print(f"  [FAULT] error_code={err}")
                fault = True

        # NON-TERMINATING fault recovery: on a collision/overspeed/servo error, clear
        # + re-home + keep going (episode stays FIXED length). Only give up after
        # max_faults recoveries in one episode (avoids hammering the hardware in a
        # tight fault loop). done stays False here -> _ct keeps counting to max_steps.
        if fault:
            self._fault_count += 1
            if self._fault_count > self.max_faults:
                print(f"  [FAULT] exceeded {self.max_faults} recoveries - ending episode")
                self._stopped = True; done = True
            else:
                print(f"  [FAULT] recovering #{self._fault_count}/{self.max_faults} "
                      f"(re-home, episode continues)")
                self._recover()

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
