#!/usr/bin/env python3
"""Precompute a pool of FK-verified SAFE joint waypoints (sim venv).

Samples random joint configs, forward-kinematics the end-effector, and keeps those
whose EE lies inside a margin'd safe box (and above the table). The resulting pool
[K,6] is loaded by both collectors (explore.WaypointBabbler) so the random policy
sweeps the reachable workspace while staying safe.

Run once in the sim venv (needs gymnasium+mujoco):
    MUJOCO_GL=egl python scripts/diffil/make_safe_waypoints.py --num 4000 --out data/safe_waypoints.npz
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from xarm_rl.envs.base_env import (JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, HOME_QPOS, ASSETS_DIR)
from xarm_rl.envs.reach_env import SAFE_LOW, SAFE_HIGH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=4000, help="number of safe waypoints to keep")
    ap.add_argument("--margin", type=float, default=0.04, help="shrink safe box by this (m)")
    ap.add_argument("--z-min", type=float, default=0.41, help="keep EE above the table top")
    ap.add_argument("--max-tries", type=int, default=400000)
    ap.add_argument("--out", default="data/safe_waypoints.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(ASSETS_DIR / "scene_reach.xml"))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}")] for i in range(6)]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    def fk(q):
        for a, qq in zip(qadr, q):
            d.qpos[a] = qq
        mujoco.mj_forward(m, d)
        return d.site_xpos[sid].copy()

    # finite sweep of joint limits (some are full-turn) + bias toward the front workspace
    lo = np.maximum(JOINT_LIMITS_LOW, [-3.14, -2.0, -3.9, -3.14, -1.69, -3.14]).astype(np.float32)
    hi = np.minimum(JOINT_LIMITS_HIGH, [3.14, 2.0, 0.19, 3.14, 3.14, 3.14]).astype(np.float32)
    box_lo = SAFE_LOW + args.margin
    box_hi = SAFE_HIGH - args.margin

    rng = np.random.default_rng(args.seed)
    keep = []
    tries = 0
    while len(keep) < args.num and tries < args.max_tries:
        # 70% sample near home (smaller joint excursions -> smoother babbling paths),
        # 30% sample the full sweep (coverage of the wider workspace)
        if rng.random() < 0.7:
            q = np.clip(HOME_QPOS + rng.uniform(-1.0, 1.0, 6).astype(np.float32), lo, hi)
        else:
            q = rng.uniform(lo, hi).astype(np.float32)
        ee = fk(q)
        if np.all(ee >= box_lo) and np.all(ee <= box_hi) and ee[2] >= args.z_min:
            keep.append(q)
        tries += 1

    wp = np.asarray(keep, np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, waypoints=wp)
    ee_all = np.array([fk(q) for q in wp]) if len(wp) else np.zeros((0, 3))
    print(f"[saved] {args.out}  waypoints={wp.shape}  (tries={tries})")
    if len(wp):
        print("  EE coverage  x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]" % (
            ee_all[:, 0].min(), ee_all[:, 0].max(), ee_all[:, 1].min(),
            ee_all[:, 1].max(), ee_all[:, 2].min(), ee_all[:, 2].max()))


if __name__ == "__main__":
    main()
