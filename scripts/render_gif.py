"""Render a trained policy rollout to GIF (headless, no DISPLAY required).

Uses MuJoCo's offscreen Renderer with EGL backend. Set MUJOCO_GL=egl in env
(this script does so automatically).

Usage:
    python scripts/render_gif.py --task reach --algo ppo \\
        --model outputs/reach_ppo_v2/final_model.zip \\
        --out outputs/reach_ppo_v2/rollout.gif --episodes 3
"""
from __future__ import annotations

import argparse
import os
# Must be set BEFORE importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
import gymnasium as gym
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO, SAC

import xarm_rl  # noqa: F401  registers envs


TASK_TO_ENV = {"reach": "XArm6Reach-v0", "pick_place": "XArm6PickPlace-v0"}
ALGO_CLS = {"ppo": PPO, "sac": SAC}


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData,
                 camera: str | int = -1) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--algo", choices=list(ALGO_CLS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output gif path")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--render-every", type=int, default=2,
                    help="render every N sim steps (lower = smoother, larger gif)")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--camera", default="-1",
                    help="camera name or id (-1 = free camera default)")
    args = ap.parse_args()

    env = gym.make(TASK_TO_ENV[args.task])
    # Drill into the underlying MjModel/MjData
    base = env.unwrapped
    model: mujoco.MjModel = base.model
    data: mujoco.MjData = base.data

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    policy = ALGO_CLS[args.algo].load(args.model, device="auto")

    cam = args.camera
    try:
        cam = int(cam)
    except ValueError:
        pass

    frames: list[np.ndarray] = []
    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, t, info = False, 0, {}
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            t += 1
            if t % args.render_every == 0 or done:
                frames.append(render_frame(renderer, data, camera=cam))
        if info.get("is_success", 0.0) > 0.5:
            successes += 1
        print(f"  ep {ep+1}/{args.episodes}: success={info.get('is_success', 0)}  steps={t}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(frames)} frames → {out_path}")
    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
    print(f"[done] {out_path}  ({successes}/{args.episodes} successful)")
    env.close()


if __name__ == "__main__":
    main()
