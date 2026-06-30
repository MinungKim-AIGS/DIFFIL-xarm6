#!/usr/bin/env python3
"""Executor for real xArm6 data collection.

This file only *drives* the collector defined in ``real_reach_collector.py``.
Keeping execution separate from logic makes the collector importable/testable.

Default task: reach.

Examples
--------
Dry-run (no hardware — mock arm + dummy camera), verify shapes/format:
    python scripts/run_real_reach_collect.py --dry-run --num-episodes 2 --max-steps 20

Real collection:
    python scripts/run_real_reach_collect.py \
        --ip 192.168.1.199 --front-serial 817512070394 \
        --num-episodes 20 --max-steps 200 --action-scale 0.05
"""
from __future__ import annotations

import os
import argparse
import numpy as np

from explore import load_waypoints
from real_reach_collector import (
    RealReachCollector, RealSenseCamera, DummyCamera, open_arm,
    ACTION_SCALE, CONTROL_HZ, OBS_DIM, ACTION_DIM,
)


def parse_args():
    p = argparse.ArgumentParser(description="Real xArm6 data collection (state-based, transfer-aligned)")
    p.add_argument("--task", type=str, default="reach", choices=["reach"],
                   help="task to collect (default: reach; pusher is a later follow-up)")
    p.add_argument("--ip", type=str, default="192.168.1.199", help="xArm6 IP")
    p.add_argument("--front-serial", type=str, default=None, help="front RealSense serial")
    p.add_argument("--num-samples", type=int, default=10000,
                   help="total transitions to collect (0 -> use --num-episodes)")
    p.add_argument("--num-episodes", type=int, default=0,
                   help="episodes if --num-samples=0")
    p.add_argument("--max-steps", type=int, default=200, help="steps/episode (reach env uses 200)")
    p.add_argument("--action-scale", type=float, default=ACTION_SCALE, help="rad/step per joint")
    p.add_argument("--action-std", type=float, default=0.5,
                   help="std of normalized random action before clipping to [-1,1]")
    p.add_argument("--home-jitter", type=float, default=0.05, help="rad uniform jitter on home pose")
    p.add_argument("--control-hz", type=float, default=CONTROL_HZ)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="./data/real_reach")
    p.add_argument("--waypoints", type=str, default="data/safe_waypoints.npz",
                   help="safe joint-waypoint pool (make_safe_waypoints.py); missing -> OU fallback")
    p.add_argument("--min-steps", type=int, default=50, help="discard episodes shorter than this")
    p.add_argument("--dry-run", action="store_true", help="mock arm + dummy camera, no hardware")
    return p.parse_args()


def main():
    args = parse_args()
    if args.task != "reach":
        raise NotImplementedError(f"task '{args.task}' not supported yet (reach only)")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] connecting arm ({'DRY-RUN mock' if args.dry_run else args.ip}) ...")
    arm = open_arm(args.ip, dry_run=args.dry_run)

    if args.dry_run:
        camera = DummyCamera()
    else:
        if not args.front_serial:
            raise SystemExit("--front-serial is required for a real run")
        print(f"[*] connecting front camera (serial={args.front_serial}) ...")
        camera = RealSenseCamera(serial_number=args.front_serial)

    wp = load_waypoints(args.waypoints)
    print(f"[*] exploration: {('%d safe waypoints' % len(wp)) if wp is not None else 'OU fallback (no pool)'}")
    collector = RealReachCollector(
        arm=arm, camera=camera,
        action_scale=args.action_scale, control_hz=args.control_hz,
        seed=args.seed, waypoints=wp, min_steps=args.min_steps,
    )

    dataset = {}
    try:
        dataset = collector.collect(
            num_episodes=args.num_episodes, max_steps=args.max_steps,
            action_std=args.action_std, home_jitter=args.home_jitter,
            num_samples=args.num_samples,
        )
    except KeyboardInterrupt:
        print("\n[*] interrupted — saving what was collected ...")
    finally:
        print("[*] stopping arm + camera")
        try:
            arm.set_mode(0); arm.set_state(0); arm.disconnect()
        except Exception as e:
            print(f"  arm shutdown error: {e}")
        camera.stop()

    if not dataset:
        print("[!] no data collected.")
        return

    save_path = os.path.join(args.output_dir, "xarm6_real_reach_dataset.npz")
    np.savez_compressed(save_path, **dataset)

    print(f"\n[done] saved: {save_path}  (total {dataset['n']} steps)")
    print(f"  obs : {dataset['obs'].shape}  (expected [*, {OBS_DIM}])")
    print(f"  act : {dataset['act'].shape}  (expected [*, {ACTION_DIM}], range "
          f"[{dataset['act'].min():.2f}, {dataset['act'].max():.2f}])")
    print(f"  ims : {dataset['ims'].shape}")


if __name__ == "__main__":
    main()
