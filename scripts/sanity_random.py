"""Sanity check: load env, run random actions with viewer. No training needed.

Usage:
    python scripts/sanity_random.py --task reach
    python scripts/sanity_random.py --task pick_place
"""
import argparse
import time

import gymnasium as gym
import xarm_rl  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "pick_place"], default="reach")
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()

    env_id = "XArm6Reach-v0" if args.task == "reach" else "XArm6PickPlace-v0"
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=0)
    for t in range(args.steps):
        a = env.action_space.sample() * 0.3
        obs, r, term, trunc, info = env.step(a)
        env.render()
        time.sleep(env.unwrapped.control_dt)
        if term or trunc:
            print(f"t={t}  reward={r:.2f}  info={info}")
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
