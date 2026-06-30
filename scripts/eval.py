"""Evaluate a trained PPO policy with the MuJoCo viewer.

Usage:
    python scripts/eval.py --task reach --model outputs/reach/final_model.zip
    python scripts/eval.py --task pick_place --model outputs/pick_place/final_model.zip
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import xarm_rl  # noqa: F401


TASK_TO_ENV = {"reach": "XArm6Reach-v0", "pick_place": "XArm6PickPlace-v0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--render", action="store_true", default=True)
    args = ap.parse_args()

    env = gym.make(TASK_TO_ENV[args.task], render_mode="human" if args.render else None)
    model = PPO.load(args.model)

    successes, rewards = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
            if args.render:
                env.render()
                time.sleep(env.unwrapped.control_dt)
        successes.append(info.get("is_success", 0.0))
        rewards.append(ep_r)
        print(f"ep {ep:02d}  reward={ep_r:7.2f}  success={successes[-1]}")

    print(f"\nmean reward: {np.mean(rewards):.2f}   success rate: {np.mean(successes):.1%}")
    env.close()


if __name__ == "__main__":
    main()
