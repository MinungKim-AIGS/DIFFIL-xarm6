#!/usr/bin/env python3
"""D3IL EVALUATION node (robot laptop) — also works for DIFFIL policies.

Runs a trained SAC policy on the real xArm6 and reports the reach success rate.
The policy is the SAME format both algorithms export (weight_io / NumpyActor), so
this evaluator is shared.

Two ways to get the policy:
  --weights actor.npz     : load a saved policy (weight_io.save_weights output)
  --server-host <ip>      : pull the latest published policy over ZeroMQ (5558)

Real eval:
    python d3il_eval_node.py --ip 192.168.1.199 --front-serial <SERIAL> \
        --weights actor_v50.npz --episodes 20 --success-dist 0.05
Dry-run (mock arm, no hardware):
    python d3il_eval_node.py --dry-run --weights actor_v50.npz --episodes 3 --max-steps 20
"""
from __future__ import annotations

import argparse
import time
import numpy as np

import weight_io
from policy_runtime import NumpyActor, random_actor
from real_diffil_env import RealRobotEnv


def get_policy(args):
    if args.weights:
        print(f"[eval] loading policy from {args.weights}")
        return NumpyActor(weight_io.load_weights(args.weights))
    if args.server_host:
        from comm import WeightPuller
        puller = WeightPuller(args.server_host, args.weight_port)
        print(f"[eval] waiting for a published policy from {args.server_host}:{args.weight_port} ...")
        w = None
        t0 = time.time()
        while w is None and time.time() - t0 < args.pull_timeout:
            w = puller.latest(); time.sleep(0.2)
        puller.close()
        if w is None:
            raise SystemExit("no policy received within --pull-timeout")
        print(f"[eval] received policy v{w['version']}")
        return NumpyActor(w)
    print("[eval] WARNING: no --weights / --server-host; using a RANDOM policy (sanity only)")
    return NumpyActor(random_actor(obs_dim=args.obs_dim, act_dim=args.action_dim))


def run_episode(env, actor, success_dist):
    ob = env.reset()
    done, reached, min_dist = False, False, np.inf
    while not done:
        act = actor.get_action(ob, 0.0)              # deterministic for eval
        ob, rew, done, info = env.step(act)
        ee, goal = ob[12:15], ob[15:18]              # reach obs layout
        d = float(np.linalg.norm(goal - ee))
        min_dist = min(min_dist, d)
        if d <= success_dist:
            reached = True
    return reached, min_dist


def main():
    ap = argparse.ArgumentParser(description="Evaluate a SAC policy on the real xArm6 (D3IL/DIFFIL)")
    ap.add_argument("--ip", default="192.168.1.199")
    ap.add_argument("--front-serial", default=None)
    ap.add_argument("--weights", default=None, help=".npz from weight_io.save_weights")
    ap.add_argument("--server-host", default=None, help="pull latest policy over ZeroMQ instead")
    ap.add_argument("--weight-port", type=int, default=5558)
    ap.add_argument("--pull-timeout", type=float, default=30.0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--success-dist", type=float, default=0.05, help="reach success threshold (m)")
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--action-filter", type=float, default=0.3)
    ap.add_argument("--home-jitter", type=float, default=0.05)
    ap.add_argument("--obs-dim", type=int, default=21)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    actor = get_policy(args)
    env = RealRobotEnv(ip=args.ip, front_serial=args.front_serial, dry_run=args.dry_run,
                       control_hz=args.control_hz, action_filter=args.action_filter,
                       max_steps=args.max_steps, home_jitter=args.home_jitter, seed=args.seed)
    successes, dists = 0, []
    try:
        for ep in range(args.episodes):
            reached, min_dist = run_episode(env, actor, args.success_dist)
            successes += int(reached)
            dists.append(min_dist)
            print(f"[eval] ep {ep + 1}/{args.episodes}: {'SUCCESS' if reached else 'fail'} "
                  f"(min dist {min_dist:.3f} m)")
    except KeyboardInterrupt:
        print("\n[eval] interrupted")
    finally:
        env.close()

    n = len(dists)
    if n:
        print("=" * 48)
        print(f"reach success rate : {successes}/{n} = {100.0 * successes / n:.1f}%")
        print(f"mean min-distance  : {np.mean(dists):.3f} m")
        print("=" * 48)


if __name__ == "__main__":
    main()
