"""Minimal xArm6 servo command diagnostic.

Sends ONE tiny servo_angle_j command (joint1 +0.01 rad) and reports the
SDK return code + any error/warning that appears. Use this to isolate
SDK-call issues from policy-loop issues before running deploy_grid_tour.

Usage:
    python scripts/diag_servo.py
    python scripts/diag_servo.py --ip 192.168.1.199 --delta 0.01
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from xarm.wrapper import XArmAPI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.199")
    ap.add_argument("--delta", type=float, default=0.01,
                    help="rad to add to joint1 (default 0.01 rad ≈ 0.57°)")
    args = ap.parse_args()

    arm = XArmAPI(args.ip)

    arm.motion_enable(True)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.5)

    print(f"state before     : {arm.get_state()}")
    print(f"err_warn before  : {arm.get_err_warn_code()}")

    _, q = arm.get_servo_angle(is_radian=True)
    q6 = list(q[:6])
    target = list(q6)
    target[0] += args.delta

    print(f"current q6       : {np.round(q6, 4)}")
    print(f"target  q6       : {np.round(target, 4)}")

    code = arm.set_servo_angle_j(target, is_radian=True, speed=0.1, mvacc=0.2)
    print(f"return code      : {code}")

    print(f"state after      : {arm.get_state()}")
    print(f"err_warn after   : {arm.get_err_warn_code()}")

    arm.disconnect()


if __name__ == "__main__":
    main()
