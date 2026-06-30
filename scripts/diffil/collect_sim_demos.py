#!/usr/bin/env python3
"""Collect the fixed SOURCE (sim) datasets for DIFF-IL:

    B^SE  source expert  : --mode policy  (rolls out a trained SB3 reach policy)
    B^SR  source random  : --mode random  (random actions)

Self-contained: uses xarm_rl.envs.diffil_adapter.SimDiffilEnv (the same gymnasium
reach env, with a selectable camera) and writes a d3il-format .npz
(obs,nobs,act,rew,don,ims,ids,step,n) that DIFF-IL's DemonstrationsReplayBuffer
consumes. Runs dataset_conform at the end.

    # B^SR (random) under the front camera
    python scripts/diffil/collect_sim_demos.py --mode random --render-camera front \
        --num-episodes 50 --name XArm6Reach_random --out-dir prior_data

    # B^SE (expert) from a trained SB3 policy
    python scripts/diffil/collect_sim_demos.py --mode policy --algo ppo \
        --model outputs/reach_ppo_dr/final_model.zip --name XArm6Reach --out-dir expert_data
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # dataset_conform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (explore)
# repo root (for `import xarm_rl`) — scripts/diffil -> repo root is two up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataset_conform import check_demo_npz
from explore import WaypointBabbler, SafetyRecovery, load_waypoints


def load_policy(algo: str, model_path: str):
    """Return a fn obs->action using an SB3 model (lazy import; server only)."""
    if algo.lower() == "ppo":
        from stable_baselines3 import PPO as Algo
    elif algo.lower() == "sac":
        from stable_baselines3 import SAC as Algo
    else:
        raise ValueError(f"unknown algo {algo}")
    model = Algo.load(model_path)
    return lambda obs: np.asarray(model.predict(obs, deterministic=True)[0], np.float32)


def collect(args):
    from xarm_rl.envs.diffil_adapter import SimDiffilEnv
    env = SimDiffilEnv(env_id=args.env_id, render_camera=args.render_camera)
    env.seed(args.seed)
    max_steps = args.max_steps or env.spec_max_steps

    pi = None
    if args.mode == "policy":
        if not args.model:
            raise SystemExit("--mode policy requires --model <sb3.zip>")
        pi = load_policy(args.algo, args.model)

    babbler = recovery = None
    if pi is None:                       # B^SR: same goal-babbling explorer as the real collector
        from xarm_rl.envs.base_env import JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, HOME_QPOS
        from xarm_rl.envs.reach_env import SAFE_LOW, SAFE_HIGH
        a_scale = float(env.env.unwrapped.action_scale)
        wp = load_waypoints(args.waypoints)
        print(f"[*] random explore: {('%d safe waypoints' % len(wp)) if wp is not None else 'OU fallback'}")
        babbler = WaypointBabbler(a_scale, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, HOME_QPOS,
                                  waypoints=wp, seed=args.seed)
        recovery = SafetyRecovery(SAFE_LOW, SAFE_HIGH, HOME_QPOS, a_scale, margin=0.03)

    OBS, NOBS, ACT, REW, DON, IMS, IDS, STEP = [], [], [], [], [], [], [], []
    ep, total = 0, 0
    target = args.num_samples
    while (total < target) if target > 0 else (ep < args.num_episodes):
        ob = env.reset()
        _ = env.get_ims()                      # init frame stack
        if babbler is not None:
            babbler.reset(ob[:6])
        done, t = False, 0
        while not done and t < max_steps:
            if pi is None:
                base_a = babbler.act(ob[:6])
                act, _ = recovery.wrap(ob[:6], ob[12:15], base_a)   # q=ob[0:6], ee=ob[12:15]
            else:
                act = np.clip(pi(ob), -1.0, 1.0).astype(np.float32)
            nob, rew, done, info = env.step(act)
            im = env.get_ims()
            OBS.append(ob); ACT.append(act); NOBS.append(nob); REW.append(rew)
            DON.append(done); IMS.append(im); IDS.append(ep); STEP.append(t + 1)
            ob = nob; t += 1
        total += t; ep += 1
        _tgt = f"{total}/{target} samples" if target > 0 else f"ep {ep}/{args.num_episodes}"
        print(f"  ep {ep}: {t} steps  ({_tgt})")

    data = {"obs": np.asarray(OBS, np.float32), "nobs": np.asarray(NOBS, np.float32),
            "act": np.asarray(ACT, np.float32), "rew": np.asarray(REW, np.float32),
            "don": np.asarray(DON, bool), "ims": np.asarray(IMS, np.uint8),
            "ids": np.asarray(IDS, np.int32), "step": np.asarray(STEP, np.int32),
            "n": len(ACT)}
    env.close()

    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.name}.npz")
    np.savez_compressed(path, **data)
    print(f"[saved] {path}  obs{data['obs'].shape} ims{data['ims'].shape} n={data['n']}")
    check_demo_npz(path, past_frames=data["ims"].shape[1], epi_len=max_steps)
    return path


def main():
    ap = argparse.ArgumentParser(description="Collect source (sim) DIFF-IL demos")
    ap.add_argument("--mode", choices=["random", "policy"], required=True)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--model", default=None, help="SB3 .zip for --mode policy")
    ap.add_argument("--env-id", default="XArm6Reach-v0")
    ap.add_argument("--render-camera", default="front",
                    choices=["front", "ob_b", "ob_c", "ob_d", "topdown"])
    ap.add_argument("--num-samples", type=int, default=10000,
                    help="total transitions to collect (0 -> use --num-episodes)")
    ap.add_argument("--num-episodes", type=int, default=0,
                    help="episodes if --num-samples=0")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = env default (200)")
    ap.add_argument("--name", default="XArm6Reach_random", help="dataset subdir + file name")
    ap.add_argument("--out-dir", default="prior_data", help="expert_data (B^SE) or prior_data (B^SR/B^TR)")
    ap.add_argument("--waypoints", type=str, default="data/safe_waypoints.npz",
                    help="safe joint-waypoint pool (make_safe_waypoints.py); missing -> OU fallback")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
