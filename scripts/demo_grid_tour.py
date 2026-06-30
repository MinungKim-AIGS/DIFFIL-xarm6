"""9-point grid tour demo: policy visits 9 evenly-spaced targets in the safe zone,
returning to home between each. Records a GIF.

Targets form a 3x3 grid on a constant z-plane inside the safe zone.

Usage:
    python scripts/demo_grid_tour.py \
        --task reach --algo ppo \
        --model outputs/reach_ppo_v2/final_model.zip \
        --out outputs/reach_ppo_v2/grid_tour.gif
"""
from __future__ import annotations

import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
import gymnasium as gym
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO, SAC

import xarm_rl  # noqa: F401


TASK_TO_ENV = {"reach": "XArm6Reach-v0"}
ALGO_CLS = {"ppo": PPO, "sac": SAC}

# 3x3 grid in the safe zone, on a fixed z-plane.
# Safe zone: x:0..0.57, y:-0.54..0.55, z:0.18..0.60 (meters, base frame)
# We pick interior values to stay clear of joint singularities.
GRID_X = [0.32, 0.42, 0.52]
GRID_Y = [-0.20, 0.00, 0.20]
GRID_Z = 0.45

HOME_TARGET = np.array([0.42, 0.00, 0.55], dtype=np.float32)   # near-home above center


def build_targets():
    pts = []
    # snake order: row 0 left→right, row 1 right→left, row 2 left→right
    for i, y in enumerate(GRID_Y):
        xs = GRID_X if i % 2 == 0 else list(reversed(GRID_X))
        for x in xs:
            pts.append(np.array([x, y, GRID_Z], dtype=np.float32))
    return pts


def overlay_marker(scene, pos, rgba, size=0.018):
    """Add a small sphere to the scene at world `pos` for visualization."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0, 0]),
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def render_frame(renderer, model, data, all_pts, visited, current_idx, camera=-1):
    """Render one frame with grid markers overlaid (visited green, current red, future grey)."""
    renderer.update_scene(data, camera=camera)
    scene = renderer.scene
    for i, p in enumerate(all_pts):
        if i == current_idx:
            rgba = (1.0, 0.2, 0.2, 1.0)      # active target — red
            size = 0.025
        elif visited[i]:
            rgba = (0.2, 0.9, 0.2, 1.0)      # visited — green
            size = 0.018
        else:
            rgba = (0.6, 0.6, 0.6, 0.6)      # pending — grey
            size = 0.015
        overlay_marker(scene, p, rgba, size)
    return renderer.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), default="reach")
    ap.add_argument("--algo", choices=list(ALGO_CLS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--render_every", type=int, default=2)
    ap.add_argument("--max_steps_per_segment", type=int, default=120)
    ap.add_argument("--success_dist", type=float, default=0.03)
    ap.add_argument("--camera", default="-1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make(TASK_TO_ENV[args.task])
    base = env.unwrapped
    model_mj = base.model
    data = base.data

    renderer = mujoco.Renderer(model_mj, height=args.height, width=args.width)
    policy = ALGO_CLS[args.algo].load(args.model, device="auto")

    cam = args.camera
    try: cam = int(cam)
    except ValueError: pass

    targets = build_targets()
    print(f"[demo] {len(targets)} targets")
    for i, p in enumerate(targets):
        print(f"  P{i}: x={p[0]:.2f}  y={p[1]:+.2f}  z={p[2]:.2f}")

    # Reset env once
    obs, _ = env.reset(seed=args.seed)

    # Override target programmatically — we'll write directly to base.target_pos
    # and the mocap body each iteration.
    target_body_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = model_mj.body_mocapid[target_body_id]

    frames = []
    visited = [False] * len(targets)
    successes = 0

    def run_segment(goal_xyz: np.ndarray, current_idx: int,
                    label: str, success_count_this: bool,
                    reset_to_home: bool = True):
        """Run policy until close to goal_xyz or max steps. Returns success bool.

        With reset_to_home=True, reset env to HOME_QPOS first so the state
        distribution matches training. This means a clean per-segment reach.
        """
        nonlocal successes
        if reset_to_home:
            # Reset env then override target (env.reset samples its own target which we discard)
            env.reset(seed=args.seed + current_idx + 1)
        base.target_pos = goal_xyz.copy()
        data.mocap_pos[target_mocap_id] = goal_xyz
        mujoco.mj_forward(model_mj, data)
        ok = False
        for t in range(args.max_steps_per_segment):
            # Build obs in the same way the env does
            q = base.get_arm_qpos()
            qd = base.get_arm_qvel()
            ee = base.get_ee_pos()
            obs_local = np.concatenate([q, qd, ee, goal_xyz, goal_xyz - ee]).astype(np.float32)
            action, _ = policy.predict(obs_local, deterministic=True)
            # Step physics via base helpers (avoid env.step which auto-resets on success)
            from xarm_rl.envs.base_env import JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH
            target_q = np.clip(q + action[:6] * base.action_scale, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
            base.apply_arm_action(target_q)
            base.step_sim()

            ee = base.get_ee_pos()
            dist = float(np.linalg.norm(goal_xyz - ee))
            if t % args.render_every == 0:
                frames.append(render_frame(renderer, model_mj, data, targets, visited, current_idx, cam))
            if dist < args.success_dist:
                ok = True
                # Hold a moment so the viewer can register the reach
                for _ in range(8):
                    base.apply_arm_action(target_q)
                    base.step_sim()
                frames.append(render_frame(renderer, model_mj, data, targets, visited, current_idx, cam))
                break
        if success_count_this and ok:
            successes += 1
        print(f"  [{label}] reached={ok}  dist={dist:.3f}m  steps={t+1}")
        return ok

    # Sequence: home -> P0 -> home -> P1 -> ... -> P8 -> home
    print(">>> warmup: move to home_target")
    run_segment(HOME_TARGET, current_idx=-1, label="home_init", success_count_this=False)

    for i, p in enumerate(targets):
        ok = run_segment(p, current_idx=i, label=f"P{i}", success_count_this=True)
        if ok:
            visited[i] = True
        run_segment(HOME_TARGET, current_idx=-1, label=f"home_after_P{i}", success_count_this=False)

    print(f"\n[demo] {successes}/{len(targets)} targets reached")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {len(frames)} frames -> {out_path}")
    imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
    print(f"[done] {out_path}")
    env.close()


if __name__ == "__main__":
    main()
