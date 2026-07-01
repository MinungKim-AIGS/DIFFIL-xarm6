#!/usr/bin/env python3
"""Read the base-frame coordinate of the physical goal (post-it) by touching it
with the arm — no camera calibration / table-height guessing needed. The arm's
forward kinematics gives you the exact (x,y,z); use it as GOAL_FIXED.

Run on the robot laptop (xArm connected):
    # hand-drag the arm to the post-it (if drag/manual mode is supported):
    python scripts/teach_goal.py --ip 192.168.1.199 --drag
    # or jog with the pendant / xArm Studio, then just read:
    python scripts/teach_goal.py --ip 192.168.1.199
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.199")
    ap.add_argument("--drag", action="store_true",
                    help="enable hand-drag (manual) mode so you can move the arm by hand")
    args = ap.parse_args()

    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.ip, is_radian=True)
    time.sleep(0.5)
    arm.clean_error(); arm.clean_warn(); arm.motion_enable(True)

    if args.drag:
        arm.set_mode(2)          # manual / drag-teach mode
        arm.set_state(0)
        print("[teach] DRAG mode ON — move the arm BY HAND so the TCP (or a pen tip in "
              "the gripper) sits on the post-it center.")
    else:
        arm.set_mode(0)
        arm.set_state(0)
        print("[teach] jog the arm (pendant / xArm Studio) so the TCP touches the post-it.")

    input("   -> press Enter when the TCP is exactly on the post-it ...")

    _, tcp = arm.get_position(is_radian=True)
    _, q = arm.get_servo_angle(is_radian=True)
    xyz = [round(v / 1000.0, 4) for v in tcp[:3]]      # mm -> m
    qr = [round(v, 4) for v in q[:6]]

    print("\n" + "=" * 56)
    print(f"[teach] GOAL (base frame, meters): {xyz}")
    print(f"[teach] joint config (rad):        {qr}")
    print("=" * 56)
    print("Set this in BOTH:")
    print(f"  xarm_rl/envs/reach_env.py       GOAL_FIXED = np.array({xyz}, np.float32)")
    print(f"  scripts/real_reach_collector.py GOAL_FIXED = np.array({xyz}, np.float32)")
    print("Then retrain the expert (train.py) + recollect B^SE/B^SR at this goal.")

    if args.drag:
        arm.set_mode(0); arm.set_state(0)
    arm.disconnect()


if __name__ == "__main__":
    main()
