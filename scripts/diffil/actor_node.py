#!/usr/bin/env python3
"""DIFF-IL ACTOR node (robot laptop).

Runs the SAC actor (TF-free numpy forward) on the real xArm6 at control rate,
collects full episodes, and ships them to the learner. Pulls fresh actor weights
between episodes and hot-swaps. All safety is local (RealRobotEnv guards).

Real run:
    python scripts/diffil/actor_node.py --ip 192.168.1.199 \
        --front-serial 817512070394 --server-host 10.0.0.2 \
        --init-weights /path/actor_v0.npz --num-episodes 0   # 0 = run forever

Dry-run (no hardware; pairs with: learner_node.py --mock):
    python scripts/diffil/actor_node.py --dry-run --server-host 127.0.0.1 \
        --num-episodes 3 --max-steps 20
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from comm import TrajectorySender, WeightPuller
from policy_runtime import NumpyActor, random_actor
import weight_io
from real_diffil_env import RealRobotEnv


def collect_episode(env, actor, ep_id: int, explore_noise: float):
    obs_l, nobs_l, act_l, rew_l, don_l, ims_l = [], [], [], [], [], []
    ob = env.reset()
    _ = env.get_ims()                         # init frame stack (matches Sampler)
    done = False
    while not done:
        act = actor.get_action(ob, explore_noise)
        nob, rew, done, info = env.step(act)
        im = env.get_ims()
        # store the EXECUTED action (post clip + EMA filter + safety), not the raw
        # policy output, so the off-policy buffer is consistent with what the arm did.
        applied = info.get("applied_action", act)
        obs_l.append(ob); act_l.append(applied); nobs_l.append(nob)
        rew_l.append(rew); don_l.append(done); ims_l.append(im)
        ob = nob
    T = len(act_l)
    return {"obs": np.asarray(obs_l, np.float32), "nobs": np.asarray(nobs_l, np.float32),
            "act": np.asarray(act_l, np.float32), "rew": np.asarray(rew_l, np.float32),
            "don": np.asarray(don_l, bool), "ims": np.asarray(ims_l, np.uint8),
            "ids": np.full(T, ep_id, np.int32), "step": np.arange(1, T + 1, dtype=np.int32),
            "n": T, "version": actor.version}


def main():
    ap = argparse.ArgumentParser(description="DIFF-IL actor (robot laptop)")
    ap.add_argument("--ip", default="192.168.1.199")
    ap.add_argument("--front-serial", default=None)
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--traj-port", type=int, default=5557)
    ap.add_argument("--weight-port", type=int, default=5558)
    ap.add_argument("--init-weights", default=None, help=".npz from weight_io.save_weights")
    ap.add_argument("--num-episodes", type=int, default=0, help="0 = run forever")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--action-scale", type=float, default=0.05,
                    help="rad/step per joint; lower = slower/safer on real (sim uses 0.05)")
    ap.add_argument("--action-filter", type=float, default=0.3)
    ap.add_argument("--explore-noise", type=float, default=0.1)
    ap.add_argument("--home-jitter", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    actor = NumpyActor(weight_io.load_weights(args.init_weights) if args.init_weights
                       else random_actor(version=0, seed=args.seed))
    if args.init_weights is None:
        print("[actor] WARNING: no --init-weights; starting from random actor (waiting for learner)")

    env = RealRobotEnv(ip=args.ip, front_serial=args.front_serial, dry_run=args.dry_run,
                       action_scale=args.action_scale,
                       control_hz=args.control_hz, action_filter=args.action_filter,
                       max_steps=args.max_steps, home_jitter=args.home_jitter, seed=args.seed)
    sender = TrajectorySender(args.server_host, args.traj_port)
    puller = WeightPuller(args.server_host, args.weight_port)

    ep = 0
    try:
        while args.num_episodes == 0 or ep < args.num_episodes:
            w = puller.latest()                       # hot-swap newest policy
            if w is not None and actor.update_weights(w):
                print(f"[actor] swapped to policy v{actor.version}")
            traj = collect_episode(env, actor, ep_id=ep, explore_noise=args.explore_noise)
            ok = sender.send(traj)
            print(f"[actor] ep {ep}: {traj['n']} steps, policy v{traj['version']}, "
                  f"sent={ok}, success_last={float(traj['rew'][-1]):.3f}")
            ep += 1
    except KeyboardInterrupt:
        print("\n[actor] stopped by user")
    finally:
        env.close(); sender.close(); puller.close()


if __name__ == "__main__":
    main()
