#!/usr/bin/env python3
"""Real-world xArm6 **Reach** data collector (modified).

This is the *logic-only* module. It is executed by ``run_real_reach_collect.py``.

Why this file exists
--------------------
The previous ``random_collect_d3il.py`` stored data in a structure that did NOT
match the validated state-based sim2real transfer pipeline:

    old real obs : 6-dim   (joint angles only)
    old real act : raw radian delta in [-0.03, 0.03]

The sim env (``XArm6ReachEnv``) and the *working* deploy path (``deploy_real.py``)
both use a 21-dim observation and a normalized action. This collector reproduces
that exact structure so sim-expert data and real data live in the same space:

    obs  : 21-dim  [ q(6), qd(6), ee_xyz(3), target_xyz(3), (target - ee)(3) ]
    act  :  6-dim  normalized joint deltas in [-1, 1]
                   (applied as  target_q = clip(q + act * ACTION_SCALE, limits))
    rew  :  1-dim  -||target - ee||  (- safe-zone penalty)   [reach reward_near]
    ims  :  [past_frames, H, W, C] uint8 frame stack (camera unchanged for now)

Dataset keys match both d3il datasets: obs, nobs, act, rew, don, ims, ids, n.

NOTE: camera viewpoint alignment (sim top-down vs real front) is intentionally
left unchanged here and handled as a separate follow-up.

Constants below MIRROR ``xarm_rl/envs/reach_env.py`` and ``base_env.py`` so the
real laptop does not need mujoco/gymnasium installed. Keep them in sync if the
sim env changes.
"""
from __future__ import annotations

import time
import threading
import numpy as np

from explore import WaypointBabbler, SafetyRecovery

# ----------------------------------------------------------------------------
# Constants — mirror xarm_rl/envs/base_env.py and reach_env.py
# ----------------------------------------------------------------------------
JOINT_LIMITS_LOW  = np.array([-6.283, -2.059, -3.927, -6.283, -1.693, -6.283], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([ 6.283,  2.094,  0.191,  6.283,  3.142,  6.283], dtype=np.float32)
HOME_QPOS         = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)

# Reach target sampling box (== reach_env.WORKSPACE_LOW/HIGH), base frame, meters
WORKSPACE_LOW  = np.array([0.44, -0.33, 0.41], dtype=np.float32)
WORKSPACE_HIGH = np.array([0.55,  0.33, 0.46], dtype=np.float32)
# Single fixed goal label (mirror of reach_env.GOAL_FIXED)
# x,y aligned to the real post-it (teach_goal.py); z = real post-it surface.
# Differs from sim's z (0.42) because the real table is lower — fine since eval
# ignores z and the IRL reward aligns visually.
GOAL_FIXED = np.array([0.5119, -0.2945, 0.3354], dtype=np.float32)

# Real xArm6 safe zone (== reach_env.SAFE_LOW/HIGH), meters
SAFE_LOW  = np.array([0.00, -0.54, 0.18], dtype=np.float32)
SAFE_HIGH = np.array([0.57,  0.55, 0.65], dtype=np.float32)   # z raised a bit for headroom
SAFE_ZONE_PENALTY = 1.0

ACTION_SCALE = 0.01          # rad/step per joint. Intentionally LOWER than sim (0.05):
                             # the real servo tracks stiffly, so 0.01 makes the real
                             # reach take ~sim-many steps (timestep-aligned) + safer.
ACTION_DIM   = 6
OBS_DIM      = 21

CONTROL_HZ  = 50             # == base_env 50 Hz control
CONTROL_DT  = 1.0 / CONTROL_HZ
IMG_SIZE    = (64, 64)
PAST_FRAMES = 4


# ----------------------------------------------------------------------------
# Observation builder — identical ordering to reach_env._get_obs / deploy_real
# ----------------------------------------------------------------------------
def build_reach_obs(q: np.ndarray, qd: np.ndarray,
                    ee: np.ndarray, target: np.ndarray) -> np.ndarray:
    """[q(6), qd(6), ee(3), target(3), (target-ee)(3)] -> 21-dim float32."""
    diff = target - ee
    return np.concatenate([q, qd, ee, target, diff]).astype(np.float32)


def reach_reward(ee: np.ndarray, target: np.ndarray) -> float:
    """Matches reach_env.step: reward_near (-dist) minus safe-zone penalty."""
    dist = float(np.linalg.norm(target - ee))
    r = -dist
    in_safe = bool(np.all(ee >= SAFE_LOW) and np.all(ee <= SAFE_HIGH))
    if not in_safe:
        r -= SAFE_ZONE_PENALTY
    return r


def stack_frames(images: list, past_frames: int = PAST_FRAMES) -> np.ndarray:
    """List of [H,W,C] -> [T, past_frames, H, W, C]; early steps pad first frame."""
    stacked = []
    for i in range(len(images)):
        window = [images[max(0, i - j)] for j in range(past_frames - 1, -1, -1)]
        stacked.append(np.stack(window, axis=0))
    return np.stack(stacked, axis=0)


# ----------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------
class RealSenseCamera:
    """RealSense color stream -> 64x64 RGB uint8.

    Frames are grabbed in a BACKGROUND THREAD so get_frame() is non-blocking and
    the control loop keeps a steady rate (a blocking wait_for_frames at 30 fps
    would otherwise cap the loop at ~30 Hz and add timing jitter -> servo judder).
    """

    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs  # lazy import (hardware only)
        import cv2
        self._cv2 = cv2
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        for _ in range(10):           # warm-up
            self.pipeline.wait_for_frames()
        self._latest = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        print(f"[camera] started serial={serial_number}, {width}x{height}@{fps} (threaded)")

    def _grab_loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = self._cv2.cvtColor(np.asanyarray(color_frame.get_data()), self._cv2.COLOR_BGR2RGB)
            img = self._cv2.resize(img, IMG_SIZE, interpolation=self._cv2.INTER_AREA).astype(np.uint8)
            with self._lock:
                self._latest = img

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self._latest.copy()

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        self.pipeline.stop()
        print("[camera] stopped")


class DummyCamera:
    """Zero-frame camera for --dry-run (no hardware)."""

    def __init__(self, *_, **__):
        print("[camera] DUMMY (dry-run)")

    def get_frame(self) -> np.ndarray:
        return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)

    def stop(self):
        pass


# ----------------------------------------------------------------------------
# Arm wrappers
# ----------------------------------------------------------------------------
def open_arm(ip: str, dry_run: bool):
    """Return a connected XArmAPI (real) or a MockArm (dry-run)."""
    if dry_run:
        return MockArm()
    from xarm.wrapper import XArmAPI  # lazy import (hardware only)
    arm = XArmAPI(ip, is_radian=True)
    time.sleep(0.5)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(1)      # servo / streaming
    arm.set_state(0)
    time.sleep(0.5)
    return arm


class MockArm:
    """Minimal stand-in so the pipeline runs without hardware.

    Integrates commanded joint targets instantly and exposes a crude forward
    map for the TCP so obs values move and qd is non-zero.
    """

    def __init__(self):
        self._q = HOME_QPOS.copy()
        self.error_code = 0
        self.warn_code = 0

    # --- API surface used by this collector / open_arm ---
    def clean_error(self): pass
    def clean_warn(self): pass
    def motion_enable(self, enable=True): pass
    def set_mode(self, m): pass
    def set_state(self, s): pass
    def disconnect(self): pass

    def set_servo_angle(self, angle=None, speed=None, is_radian=True, wait=False):
        if angle is not None:
            self._q = np.asarray(angle, dtype=np.float32)[:6].copy()
        return 0

    def set_servo_angle_j(self, angles=None, is_radian=True, **kw):
        if angles is not None:
            self._q = np.asarray(angles, dtype=np.float32)[:6].copy()
        return 0

    def get_servo_angle(self, is_radian=True):
        return 0, self._q.tolist()

    def get_position(self, is_radian=True):
        # crude pseudo-FK -> a TCP inside the workspace, in millimetres
        x = 400.0 + 60.0 * np.sin(self._q[1])
        y = 100.0 * np.sin(self._q[0])
        z = 450.0 + 80.0 * np.cos(self._q[1] + self._q[2])
        return 0, [x, y, z, 0.0, 0.0, 0.0]


# ----------------------------------------------------------------------------
# Controller fault helper
# ----------------------------------------------------------------------------
def arm_fault(arm):
    """(error_code, warn_code); 0 == OK. Works for XArmAPI and MockArm."""
    return int(getattr(arm, "error_code", 0) or 0), int(getattr(arm, "warn_code", 0) or 0)


# ----------------------------------------------------------------------------
# Collector
# ----------------------------------------------------------------------------
class RealReachCollector:
    """Collects Reach trajectories from the real (or mock) xArm6."""

    def __init__(self, arm, camera, action_scale: float = ACTION_SCALE,
                 control_hz: float = CONTROL_HZ, past_frames: int = PAST_FRAMES,
                 seed: int | None = None, waypoints=None, recovery_margin: float = 0.03,
                 hold_steps=(15, 50), min_steps: int = 50, max_faults: int = 10,
                 max_action: float = 1.0, smooth: float = 0.0,
                 ou_sigma: float = 0.3, ou_mix: float = 0.2):
        self.arm = arm
        self.camera = camera
        self.action_scale = float(action_scale)
        self.control_dt = 1.0 / float(control_hz)
        self.past_frames = past_frames
        self.rng = np.random.default_rng(seed)
        self.min_steps = int(min_steps)
        self.max_faults = int(max_faults)       # per-episode fault recoveries before giving up
        # max_action / smooth control how gently the arm moves (see explore.py).
        self.babbler = WaypointBabbler(action_scale, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH,
                                       HOME_QPOS, waypoints=waypoints, hold_steps=hold_steps,
                                       ou_sigma=ou_sigma, ou_mix=ou_mix,
                                       max_action=max_action, smooth=smooth, seed=(seed or 0))
        self.recovery = SafetyRecovery(SAFE_LOW, SAFE_HIGH, HOME_QPOS, action_scale,
                                       margin=recovery_margin, max_action=max_action)

    # --- low-level reads ---
    def read_q(self) -> np.ndarray:
        code, angles = self.arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_servo_angle failed: code={code}")
        return np.asarray(angles[:6], dtype=np.float32)

    def read_ee(self) -> np.ndarray:
        code, tcp = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed: code={code}")
        return np.asarray(tcp[:3], dtype=np.float32) / 1000.0  # mm -> m

    # --- robustness helpers (mirror scripts/diffil/real_diffil_env.py) ---
    def _home_ward(self, q: np.ndarray) -> np.ndarray:
        """Joint-space action that moves toward HOME_QPOS (clipped to [-1,1])."""
        return np.clip((HOME_QPOS - q) / self.action_scale, -1.0, 1.0).astype(np.float32)

    def _ensure_ready(self, mode: int, retries: int = 4, settle: float = 0.4) -> bool:
        """Robustly bring the arm to a movable/ready state (clear faults, enable,
        set mode+state) and VERIFY (err==0 and state not 'stop') before returning.
        Prevents the 'xArm is not ready to move' race after a collision."""
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
                ecode, _w = arm_fault(self.arm)
            except Exception:
                ecode = 0
            try:
                _code, state = self.arm.get_state()
            except Exception:
                state = 2                      # MockArm / dry-run -> treat as ready
            if ecode == 0 and (state is None or state < 4):
                return True
        print(f"  [ready] WARNING: arm not confirmed ready (err={ecode}, state={state})")
        return False

    def _recover(self) -> None:
        """Clear a collision/overspeed/servo fault by re-enabling the servo IN PLACE
        (NO teleport). The caller then latches home-steering so the arm returns to
        HOME_QPOS SMOOTHLY over the control loop -> continuous frames, fixed length."""
        try:
            self.arm.clean_error(); self.arm.clean_warn()
        except Exception:
            pass
        self._ensure_ready(mode=1)             # re-enable servo mode in place (no motion)

    def reset_to_home(self, home_jitter: float = 0.0) -> None:
        """Position-mode move to home (+ optional jitter), then back to servo mode."""
        noise = self.rng.uniform(-home_jitter, home_jitter, size=6).astype(np.float32) \
            if home_jitter > 0 else np.zeros(6, dtype=np.float32)
        home = (HOME_QPOS + noise).astype(np.float32)
        # robustly reach a ready state FIRST (clears residual faults from a prior
        # collision) so the home command doesn't hit "xArm is not ready to move".
        self._ensure_ready(mode=0)
        ret = self.arm.set_servo_angle(angle=home.tolist(), speed=0.35,
                                       is_radian=True, wait=True)
        if ret != 0:
            print(f"  warn: home move ret={ret} (continuing)")
        time.sleep(0.3)
        self._ensure_ready(mode=1)             # servo-streaming mode for the episode

    def sample_target(self) -> np.ndarray:
        # Single fixed goal label (mirror of reach_env.GOAL_FIXED).
        return GOAL_FIXED.copy()

    def sample_action(self, action_std: float) -> np.ndarray:
        """Normalized joint delta in [-1, 1] (== sim action space)."""
        a = self.rng.normal(0.0, action_std, size=ACTION_DIM)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    # --- one episode ---
    def collect_episode(self, ep_idx: int, max_steps: int,
                        action_std: float, home_jitter: float):
        self.reset_to_home(home_jitter)
        target = self.sample_target()
        self.babbler.reset(self.read_q())

        ep_obs, ep_act, ep_rew, ep_don, ep_ims = [], [], [], [], []
        prev_q = None
        homing = False          # post-fault: latched smooth home-steer (no teleport)
        fault_count = 0

        for step in range(max_steps):
            t0 = time.time()

            q = self.read_q()
            qd = np.zeros(6, dtype=np.float32) if prev_q is None \
                else ((q - prev_q) / self.control_dt).astype(np.float32)
            ee = self.read_ee()
            obs = build_reach_obs(q, qd, ee, target)
            frame = self.camera.get_frame()

            # action: post-fault home-steer latch > safe-zone recovery > goal-babbling
            if homing:
                # steer home SMOOTHLY until back near HOME_QPOS, then resume babbling
                # (no teleport -> the recorded frames change gradually).
                action = self._home_ward(q)
                if float(np.linalg.norm(q - HOME_QPOS)) < 0.10:
                    homing = False
            else:
                base_a = self.babbler.act(q)
                action, _recovered = self.recovery.wrap(q, ee, base_a)   # safe-zone (non-terminating)
            target_q = np.clip(q + action * self.action_scale,
                               JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            ret = self.arm.set_servo_angle_j(angles=target_q.tolist(), is_radian=True)

            fault = (ret != 0)
            if not fault:
                err, warn = arm_fault(self.arm)
                fault = (err != 0)
            # NON-TERMINATING fault recovery: on collision/overspeed/servo error, re-enable
            # in place + latch smooth home-steer, and KEEP collecting so the episode length
            # stays FIXED. Give up only after max_faults recoveries in one episode.
            if fault:
                fault_count += 1
                if fault_count > self.max_faults:
                    print(f"  [FAULT] exceeded {self.max_faults} recoveries at step {step} - ending episode")
                    break
                print(f"  [FAULT] recovering #{fault_count}/{self.max_faults} at step {step} "
                      f"(re-enable + smooth home-steer, episode continues)")
                self._recover()
                homing = True

            ep_obs.append(obs)
            ep_act.append(action)
            ep_rew.append(reach_reward(ee, target))
            ep_don.append(step == max_steps - 1)
            ep_ims.append(frame)

            prev_q = q
            dt = time.time() - t0
            if dt < self.control_dt:
                time.sleep(self.control_dt - dt)

        return ep_obs, ep_act, ep_rew, ep_don, ep_ims

    # --- full run ---
    def collect(self, num_episodes: int, max_steps: int,
                action_std: float = 0.5, home_jitter: float = 0.05,
                num_samples: int = 0) -> dict:
        all_obs, all_nobs, all_act = [], [], []
        all_rew, all_don, all_ims, all_ids = [], [], [], []
        total = 0

        ep = 0
        while (total < num_samples) if num_samples > 0 else (ep < num_episodes):
            _tgt = f"{total}/{num_samples} samples" if num_samples > 0 else f"ep {ep+1}/{num_episodes}"
            print(f"\n[episode {ep+1}] homing + collecting (max_steps={max_steps}) [{_tgt}]")
            ep_obs, ep_act, ep_rew, ep_don, ep_ims = self.collect_episode(
                ep, max_steps, action_std, home_jitter)
            n = len(ep_act)
            if n < self.min_steps:
                print(f"  [episode {ep+1}] too short ({n} < {self.min_steps}), discarded")
                ep += 1
                continue

            obs_arr = np.asarray(ep_obs, dtype=np.float32)        # [T, 21]
            nobs_arr = np.roll(obs_arr, -1, axis=0)
            nobs_arr[-1] = obs_arr[-1]

            all_obs.append(obs_arr)
            all_nobs.append(nobs_arr)
            all_act.append(np.asarray(ep_act, dtype=np.float32))  # [T, 6]
            all_rew.append(np.asarray(ep_rew, dtype=np.float32))
            all_don.append(np.asarray(ep_don, dtype=bool))
            all_ims.append(stack_frames(ep_ims, self.past_frames).astype(np.uint8))
            all_ids.append(np.full(n, ep, dtype=np.int32))
            total += n
            ep += 1
            print(f"  [episode {ep}] done ({n} steps, total {total})")

        if not all_obs:
            return {}

        return {
            "obs":  np.concatenate(all_obs,  axis=0),   # [N, 21]
            "nobs": np.concatenate(all_nobs, axis=0),   # [N, 21]
            "act":  np.concatenate(all_act,  axis=0),   # [N, 6] in [-1, 1]
            "rew":  np.concatenate(all_rew,  axis=0),   # [N]
            "don":  np.concatenate(all_don,  axis=0),   # [N]
            "ims":  np.concatenate(all_ims,  axis=0),   # [N, 4, 64, 64, 3]
            "ids":  np.concatenate(all_ids,  axis=0),   # [N]
            "n":    total,
        }
