#!/usr/bin/env python3
"""Measure how many steps the TRAINED expert actually takes to reach the goal in
sim, at one or more action_scales — to check sim/real timestep alignment.

Run in the SIM venv (gymnasium + SB3 + mujoco):
    MUJOCO_GL=egl python scripts/measure_reach_steps.py \
        --model outputs/reach_ppo_v2/final_model.zip --algo ppo \
        --action-scales 0.05 0.02 --reach-dist 0.05 --episodes 5
"""
from __future__ import annotations

import argparse
import numpy as np
import gymnasium as gym
import xarm_rl  # noqa: F401  registers XArm6Reach-v0


def run_scale(model, env_id, scale, reach_dist, episodes):
    steps_to_reach, finals = [], []
    for ep in range(episodes):
        env = gym.make(env_id, action_scale=scale)
        obs, _ = env.reset(seed=ep)
        horizon = env.spec.max_episode_steps or 200
        reached_at, last_d = None, np.inf
        for t in range(horizon):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            ee, goal = obs[12:15], obs[15:18]           # reach obs layout
            last_d = float(np.linalg.norm(goal - ee))
            if reached_at is None and last_d < reach_dist:
                reached_at = t + 1
            if term or trunc:
                break
        env.close()
        steps_to_reach.append(reached_at)
        finals.append(last_d)
    return steps_to_reach, finals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--env-id", default="XArm6Reach-v0")
    ap.add_argument("--reach-dist", type=float, default=0.05, help="'reached' threshold (m)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--action-scales", type=float, nargs="+", default=[0.05, 0.02])
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    model = (PPO if args.algo == "ppo" else SAC).load(args.model)

    print(f"model={args.model}  reach_dist={args.reach_dist}m  episodes={args.episodes}")
    print("-" * 64)
    for sc in args.action_scales:
        steps, finals = run_scale(model, args.env_id, sc, args.reach_dist, args.episodes)
        reached = [s for s in steps if s is not None]
        n_ok = len(reached)
        mean_steps = f"{np.mean(reached):.0f}" if n_ok else "n/a"
        print(f"action_scale={sc:<5}: reached {n_ok}/{args.episodes}  "
              f"steps-to-reach(mean)={mean_steps}  final-dist(mean)={np.mean(finals):.3f}m  "
              f"(per-ep steps={steps})")


if __name__ == "__main__":
    main()
