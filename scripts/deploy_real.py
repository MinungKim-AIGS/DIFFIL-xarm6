"""Deploy a trained policy on the REAL xArm6 via XArmAPI.

DANGER: This drives a real robot. Start with --dry-run, then --speed 30,
keep the e-stop in hand. Tested for Reach task (no gripper) first.

Controller default: IP 192.168.1.199, Modbus TCP port 502 (SDK uses 502 internally).

Usage:
    # dry-run: print actions only
    python scripts/deploy_real.py --task reach --model outputs/reach/final_model.zip \\
        --dry-run

    # real run at 30% speed (uses default --ip 192.168.1.199)
    python scripts/deploy_real.py --task reach --model outputs/reach/final_model.zip \\
        --speed 30
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO, SAC

# Lazy import — only required when --dry-run is NOT used
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


HOME_DEG = [0.0, -17.2, -68.8, 0.0, 86.0, 0.0]  # ≈ HOME_QPOS in degrees
HOME_QPOS_RAD = np.deg2rad(HOME_DEG).astype(np.float32)
_JL_MARGIN = 0.02
JOINT_LIMITS_LOW  = np.array([-6.283, -2.059, -3.927, -6.283, -1.693, -6.283], dtype=np.float32) + _JL_MARGIN
JOINT_LIMITS_HIGH = np.array([ 6.283,  2.094,  0.191,  6.283,  3.142,  6.283], dtype=np.float32) - _JL_MARGIN

# Real xArm6 safe zone (meters, base frame).
# Hard guard at deploy time — policy actions will be rejected if predicted TCP exits this box.
SAFE_LOW_M  = np.array([0.000, -0.540, 0.180], dtype=np.float32)
SAFE_HIGH_M = np.array([0.720,  0.550, 0.600], dtype=np.float32)   # x extended for goal 0.65 (== reach_env.SAFE_HIGH)

# Training start pose (== base_env.HOME_QPOS). The policy expects to begin here;
# the xArm factory move_gohome() leaves the EE high/out-of-distribution.
HOME_QPOS = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)


def in_safe_zone(ee_pos_m: np.ndarray) -> bool:
    return bool(np.all(ee_pos_m >= SAFE_LOW_M) and np.all(ee_pos_m <= SAFE_HIGH_M))


class FakeArm:
    """Dummy arm for --dry-run."""
    def __init__(self):
        self._q = np.array(HOME_DEG, dtype=np.float32)
        self.error_code = 0
        self.warn_code = 0

    def get_position(self, is_radian=True):
        # home-ish TCP inside the safe zone, in mm (so dry-run safe-guard passes)
        return 0, [450.0, 0.0, 500.0, 0.0, 0.0, 0.0]

    def clean_error(self, *a, **kw): pass
    def clean_warn(self, *a, **kw): pass
    def motion_enable(self, *a, **kw): pass
    def set_mode(self, *a, **kw): pass
    def set_state(self, *a, **kw): pass
    def move_gohome(self, *a, **kw): pass
    def set_servo_angle_j(self, angles, **kw):
        self._q = np.array(angles, dtype=np.float32)
    def get_servo_angle(self, is_radian=False):
        return 0, list(self._q if not is_radian else np.deg2rad(self._q))
    def disconnect(self): pass


def build_reach_obs(q_rad, qd_rad, ee_pos, target_pos):
    diff = target_pos - ee_pos
    return np.concatenate([q_rad, qd_rad, ee_pos, target_pos, diff]).astype(np.float32)


def fk_ee_from_joints(q_rad: np.ndarray) -> np.ndarray:
    """Placeholder forward kinematics. Replace with proper FK (e.g. pinocchio / xArm SDK).

    Reads TCP position directly from the real arm via XArmAPI in production —
    here we return a dummy value. For deployment you should use:
        code, tcp = arm.get_position(is_radian=True)
        return np.array(tcp[:3]) / 1000.0  # mm -> m
    """
    return np.zeros(3, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach"], required=True,
                    help="pick_place not yet supported on real (gripper integration TODO)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"],
                    help="algorithm the model was trained with (must match the .zip)")
    ap.add_argument("--ip", default="192.168.1.199", help="xArm controller IP (default: 192.168.1.199)")
    ap.add_argument("--port", type=int, default=502,
                    help="Modbus TCP port — informational only; XArmAPI uses 502 internally")
    ap.add_argument("--target", nargs=3, type=float, default=[0.65, -0.15, 0.42],
                    help="target xyz in meters (world frame); default = reach_env.GOAL_FIXED")
    ap.add_argument("--speed", type=int, default=30, help="0-100, xArm servo speed")
    ap.add_argument("--hz", type=float, default=20.0, help="control loop frequency")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--action-scale", type=float, default=0.05,
                    help="rad per step per joint (match training)")
    ap.add_argument("--action-filter", type=float, default=0.3,
                    help="EMA smoothing on policy action in [0,1); 0 = off. "
                         "Damps abrupt reversals to avoid overspeed/collision trips.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save", default=None,
                    help="record the rollout to this .npz (d3il format; inspect with "
                         "inspect_dataset.py / dataset_to_gif.py)")
    ap.add_argument("--front-serial", default=None,
                    help="RealSense serial to also capture 64x64 frames (enables the GIF); "
                         "state-only if omitted")
    args = ap.parse_args()

    target = np.array(args.target, dtype=np.float32)
    print(f"[deploy] target = {target}")

    if args.dry_run:
        arm = FakeArm()
        print("[deploy] DRY RUN — no real motion")
    else:
        if XArmAPI is None:
            raise SystemExit("xArm-Python-SDK not installed. pip install xArm-Python-SDK")
        arm = XArmAPI(args.ip, is_radian=True)
        arm.motion_enable(enable=True)
        # Move to the TRAINING home (HOME_QPOS) in position mode, NOT the factory
        # gohome — the policy was trained starting from this folded pose.
        arm.set_mode(0); arm.set_state(0); time.sleep(0.2)
        arm.set_servo_angle(angle=HOME_QPOS.tolist(), speed=0.35, is_radian=True, wait=True)
        time.sleep(0.3)
        arm.set_mode(1)        # servo motion (real-time joint streaming)
        arm.set_state(0)
        time.sleep(0.2)

    model = (PPO if args.algo == "ppo" else SAC).load(args.model)
    dt = 1.0 / args.hz
    prev_q = None
    prev_action = np.zeros(6, dtype=np.float32)

    # optional rollout recording (d3il format) -> inspect_dataset.py / dataset_to_gif.py
    rec = args.save is not None
    OBS, ACT, REW, DON, RAWF = [], [], [], [], []
    cam = None
    if rec and args.front_serial and not args.dry_run:
        import real_reach_collector as rrc
        cam = rrc.RealSenseCamera(args.front_serial)

    for step in range(args.max_steps):
        t0 = time.time()

        # Read joint state + TCP (same code path for dry-run and real)
        _, q_list = arm.get_servo_angle(is_radian=True)
        q = np.array(q_list[:6], dtype=np.float32)
        _, tcp = arm.get_position(is_radian=True)
        ee = np.array(tcp[:3], dtype=np.float32) / 1000.0  # mm -> m

        # Controller fault guard: stop if the arm has faulted (collision/overspeed)
        if not args.dry_run:
            ec = int(getattr(arm, "error_code", 0) or 0)
            if ec != 0:
                print(f"[FAULT] controller error_code={ec} - STOPPING")
                break

        qd = np.zeros(6, dtype=np.float32) if prev_q is None else (q - prev_q) / dt
        prev_q = q

        obs = build_reach_obs(q, qd, ee, target)
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)[:6]

        # Action low-pass (EMA): damp abrupt reversals -> smoother joint motion,
        # less likely to trip the controller's overspeed/collision protection.
        action = args.action_filter * prev_action + (1.0 - args.action_filter) * action
        prev_action = action.copy()

        if rec:                                # record the EXECUTED (post-EMA) action
            OBS.append(obs); ACT.append(action.copy())
            REW.append(-float(np.linalg.norm(target - ee))); DON.append(False)
            if cam is not None:
                RAWF.append(cam.get_frame())

        target_q = q + action * args.action_scale

        # Safe-zone guard: if current TCP outside box, abort immediately.
        if not in_safe_zone(ee):
            print(f"[SAFETY] TCP {ee} outside safe zone {SAFE_LOW_M}..{SAFE_HIGH_M} — STOPPING")
            break

        if args.dry_run:
            arm.set_servo_angle_j(target_q.tolist(), is_radian=True)
            print(f"step {step:03d}  q={np.round(np.rad2deg(q),1)}  "
                  f"a={np.round(action,2)}  ee={np.round(ee,3)}")
        else:
            arm.set_servo_angle_j(target_q.tolist(), is_radian=True,
                                  speed=np.deg2rad(args.speed * 3))

        dist = float(np.linalg.norm(target - ee))
        if dist < 0.03 and not args.dry_run:
            print(f"[deploy] reached target (d={dist:.3f}m) in {step} steps")
            break

        # Pace the loop
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    if rec and len(ACT) > 0:
        obs_arr = np.asarray(OBS, np.float32)
        nobs = np.roll(obs_arr, -1, axis=0); nobs[-1] = obs_arr[-1]
        DON[-1] = True                          # last recorded step is terminal
        data = {"obs": obs_arr, "nobs": nobs, "act": np.asarray(ACT, np.float32),
                "rew": np.asarray(REW, np.float32), "don": np.asarray(DON, bool),
                "ids": np.zeros(len(ACT), np.int32),
                "step": np.arange(1, len(ACT) + 1, dtype=np.int32), "n": len(ACT)}
        if cam is not None:
            import real_reach_collector as rrc
            data["ims"] = rrc.stack_frames(RAWF, rrc.PAST_FRAMES).astype(np.uint8)
            cam.stop()
        np.savez_compressed(args.save, **data)
        print(f"[deploy] saved rollout -> {args.save}  (n={len(ACT)}, "
              + (f"ims {data['ims'].shape})" if 'ims' in data else "state-only)"))

    if not args.dry_run:
        arm.disconnect()
    print("[deploy] done")


if __name__ == "__main__":
    main()
