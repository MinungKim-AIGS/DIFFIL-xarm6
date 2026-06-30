#!/usr/bin/env python3
"""
xArm6 실제 로봇 무작위 정책(Random Policy) 데이터 수집 스크립트.
d3il 시뮬레이터와 호환되는 [obs, nobs, act, ims, rew, don, ids, n] 구조로 저장합니다.

Usage:
    python scripts/random_collect_d3il.py \
        --ip 192.168.1.199 \
        --wrist-serial 817512070394 \
        --num-episodes 20 \
        --max-steps 150 \
        --action-scale 0.03 \
        --output-dir ./data/d3il_random_xarm6
"""

import os
import time
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

# ==========================================
# 시뮬레이터 표준 환경 세팅과 동기화 (Radian 단위)
# ==========================================
JOINT_LIMITS_LOW  = np.array([-6.283, -2.059, -3.927, -6.283, -1.693, -6.283], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([ 6.283,  2.094,  0.191,  6.283,  3.142,  6.283], dtype=np.float32)
HOME_QPOS         = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)

CONTROL_HZ  = 50       # 시뮬레이터와 동일한 50Hz
CONTROL_DT  = 1.0 / CONTROL_HZ
IMG_SIZE    = (64, 64)
PAST_FRAMES = 4


class RealSenseCamera:
    """RealSense 카메라 스트리밍 클래스"""

    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        # 카메라 웜업
        for _ in range(10):
            self.pipeline.wait_for_frames()
        print(f"[camera] started serial={serial_number}, {width}x{height}@{fps}")

    def get_frame(self) -> np.ndarray:
        """프레임을 가져와 d3il 규격(64x64, RGB)으로 가공. 실패 시 zero frame 반환."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        img_bgr = np.asanyarray(color_frame.get_data())
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        return img_resized.astype(np.uint8)

    def stop(self):
        self.pipeline.stop()
        print("[camera] stopped")


def generate_random_action(action_scale: float, dim: int = 6) -> np.ndarray:
    """가우시안 노이즈 기반 부드러운 랜덤 관절 델타 생성 (단위: Radian)"""
    raw = np.random.normal(0, action_scale * 0.5, size=dim)
    return np.clip(raw, -action_scale, action_scale).astype(np.float32)


def stack_frames(images: list, past_frames: int = PAST_FRAMES) -> np.ndarray:
    """
    단일 에피소드 이미지 리스트 -> d3il 스타일 [T, past_frames, H, W, C] 변환.
    초기 스텝은 첫 프레임으로 패딩.
    """
    stacked = []
    for i in range(len(images)):
        window = []
        for j in range(past_frames - 1, -1, -1):
            idx = max(0, i - j)
            window.append(images[idx])
        stacked.append(np.stack(window, axis=0))  # [past_frames, H, W, C]
    return np.stack(stacked, axis=0)              # [T, past_frames, H, W, C]


def reset_to_home(robot: XArmAPI) -> None:
    """에피소드 시작 전 홈 포즈로 복귀 (Mode 0 position control)"""
    robot.set_mode(0)
    robot.set_state(0)
    time.sleep(0.2)
    home_deg = np.degrees(HOME_QPOS).tolist()
    ret = robot.set_servo_angle(angle=home_deg, speed=20, is_radian=False, wait=True)
    if ret != 0:
        raise RuntimeError(f"홈 포즈 복귀 실패: ret={ret}")
    time.sleep(0.5)
    # 실시간 서보 모드로 복귀
    robot.set_mode(1)
    robot.set_state(0)
    time.sleep(0.3)


def read_joint_state(robot: XArmAPI) -> np.ndarray:
    """현재 관절 상태 읽기 -> Radian 변환, 6DoF만 반환"""
    code, angles_deg = robot.get_servo_angle(is_radian=False)
    if code != 0:
        raise RuntimeError(f"관절 상태 읽기 실패: code={code}")
    return np.radians(angles_deg[:6]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="xArm6 랜덤 정책 d3il 데이터 수집")
    parser.add_argument("--ip",           type=str,   required=True,          help="xArm6 IP 주소")
    parser.add_argument("--wrist-serial", type=str,   required=True,          help="손목 RealSense 시리얼 번호")
    parser.add_argument("--front-serial", type=str, required=True, help="전면 RealSense 시리얼 번호")
    parser.add_argument("--num-episodes", type=int,   default=20,             help="수집할 총 에피소드 수")
    parser.add_argument("--max-steps",    type=int,   default=150,            help="에피소드당 최대 타임스텝")
    parser.add_argument("--action-scale", type=float, default=0.03,           help="스텝당 최대 관절 변화량 (rad)")
    parser.add_argument("--output-dir",   type=str,   default="./data",       help="데이터셋 저장 경로")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 장치 초기화 ----
    print(f"[*] 로봇 연결 중... ({args.ip})")
    robot = XArmAPI(args.ip, is_radian=True)
    time.sleep(0.5)
    robot.clean_error()
    robot.clean_warn()
    robot.motion_enable(enable=True)
    robot.set_mode(1)
    robot.set_state(0)
    time.sleep(0.5)

    # print(f"[*] 카메라 연결 중... (serial={args.wrist_serial})")
    # camera = RealSenseCamera(serial_number=args.wrist_serial)
    print(f"[*] 전면 카메라 연결 중... (Serial: {args.front_serial})")
    camera = RealSenseCamera(serial_number=args.front_serial)

    # 전체 에피소드 버퍼
    all_obs  = []
    all_nobs = []
    all_act  = []
    all_ims  = []
    all_don  = []
    all_ids  = []
    total_steps = 0

    try:
        for ep in range(args.num_episodes):
            print(f"\n[에피소드 {ep+1}/{args.num_episodes}] 홈 포즈 복귀 중...")
            reset_to_home(robot)

            ep_obs    = []
            ep_acts   = []
            ep_images = []
            ep_dones  = []

            print(f"[에피소드 {ep+1}] 랜덤 제어 시작 (max_steps={args.max_steps})")
            for step in range(args.max_steps):
                t_start = time.time()

                # (1) 현재 관절 상태
                current_qpos = read_joint_state(robot)

                # (2) 이미지 캡처
                frame = camera.get_frame()

                # (3) 랜덤 액션 생성
                action_delta = generate_random_action(args.action_scale)

                # (4) 목표 관절값 계산 및 클리핑
                target_qpos = np.clip(
                    current_qpos + action_delta,
                    JOINT_LIMITS_LOW,
                    JOINT_LIMITS_HIGH
                )

                # (5) 로봇에 서보 명령 (Radian, is_radian=True로 초기화했으므로 그대로 전달)
                target_deg = np.degrees(target_qpos).tolist()
                ret = robot.set_servo_angle_j(angles=target_deg, is_radian=False)
                if ret != 0:
                    print(f"  경고: set_servo_angle_j 실패 ret={ret}, step={step}")
                    break

                # (6) 버퍼 저장
                ep_obs.append(current_qpos)
                ep_acts.append(action_delta)
                ep_images.append(frame)
                ep_dones.append(step == args.max_steps - 1)

                # 50Hz 루프 유지
                elapsed = time.time() - t_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)

            ep_len = len(ep_acts)
            if ep_len == 0:
                print(f"  [에피소드 {ep+1}] 수집 실패, 건너뜀")
                continue

            # nobs: t+1 관측값 (마지막 스텝은 자기 자신)
            ep_obs_arr  = np.array(ep_obs,  dtype=np.float32)   # [T, 6]
            ep_nobs_arr = np.roll(ep_obs_arr, -1, axis=0)
            ep_nobs_arr[-1] = ep_obs_arr[-1]

            # 프레임 스태킹
            ep_stacked_ims = stack_frames(ep_images, PAST_FRAMES)  # [T, 4, 64, 64, 3]

            all_obs.append(ep_obs_arr)
            all_nobs.append(ep_nobs_arr)
            all_act.append(np.array(ep_acts,   dtype=np.float32))
            all_ims.append(ep_stacked_ims.astype(np.uint8))
            all_don.append(np.array(ep_dones,  dtype=bool))
            all_ids.append(np.full(ep_len, ep, dtype=np.int32))
            total_steps += ep_len

            print(f"  [에피소드 {ep+1}] 완료 ({ep_len} steps)")

    except KeyboardInterrupt:
        print("\n[*] Ctrl+C 수신, 수집된 데이터 저장 후 종료합니다.")

    finally:
        print("[*] 로봇 안전 정지 및 연결 해제")
        try:
            robot.set_mode(0)
            robot.set_state(0)
            robot.disconnect()
        except Exception as e:
            print(f"  로봇 종료 중 오류: {e}")
        camera.stop()

    # ---- 저장 ----
    if not all_obs:
        print("[!] 저장할 데이터가 없습니다.")
        return

    print(f"\n[*] d3il 규격 데이터셋 저장 중... (총 {total_steps} steps)")
    obs_cat  = np.concatenate(all_obs,  axis=0)  # [N, 6]
    nobs_cat = np.concatenate(all_nobs, axis=0)  # [N, 6]
    act_cat  = np.concatenate(all_act,  axis=0)  # [N, 6]
    ims_cat  = np.concatenate(all_ims,  axis=0)  # [N, 4, 64, 64, 3]
    don_cat  = np.concatenate(all_don,  axis=0)  # [N]
    ids_cat  = np.concatenate(all_ids,  axis=0)  # [N]
    rew_cat  = np.zeros(total_steps, dtype=np.float32)  # [N] — real robot은 reward 없음

    save_path = os.path.join(args.output_dir, "xarm6_real_random_dataset.npz")
    np.savez_compressed(
        save_path,
        obs=obs_cat,
        nobs=nobs_cat,
        act=act_cat,
        ims=ims_cat,
        rew=rew_cat,
        don=don_cat,
        ids=ids_cat,
        n=total_steps,
    )
    print(f"[완료] 저장: {save_path}")
    print(f"  obs : {obs_cat.shape}")
    print(f"  act : {act_cat.shape}")
    print(f"  ims : {ims_cat.shape}")
    print(f"  총  : {total_steps} steps, {len(all_obs)} episodes")


if __name__ == "__main__":
    main()