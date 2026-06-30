"""Headless evaluation: rollout policy, report success rate / reward stats.
No viewer (works on servers without DISPLAY).

Usage:
    python scripts/eval_headless.py --task reach --algo ppo \\
        --model outputs/reach_ppo/final_model.zip --episodes 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC

import xarm_rl  # noqa: F401


TASK_TO_ENV = {"reach": "XArm6Reach-v0", "pick_place": "XArm6PickPlace-v0"}
ALGO_CLS = {"ppo": PPO, "sac": SAC}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--algo", choices=list(ALGO_CLS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    env = gym.make(TASK_TO_ENV[args.task])
    model = ALGO_CLS[args.algo].load(args.model, device="auto")

    rewards, successes, ep_lens = [], [], []
    last_dists = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ep_r, t = False, 0.0, 0
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r; t += 1
            done = term or trunc
        rewards.append(ep_r)
        successes.append(float(info.get("is_success", 0.0)))
        ep_lens.append(t)
        if args.task == "reach":
            last_dists.append(float(info.get("distance", -1)))
        else:
            last_dists.append(float(info.get("d_cube_target", -1)))

    succ_rate = float(np.mean(successes))
    summary = {
        "task": args.task, "algo": args.algo, "model": args.model,
        "episodes": args.episodes,
        "success_rate": succ_rate,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "mean_final_dist": float(np.mean(last_dists)),
    }
    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    env.close()
    return succ_rate


if __name__ == "__main__":
    main()
