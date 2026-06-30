import os
# 🔥 [렌더링 안전장치] headless 서버에서 무조코 렌더링 버퍼 누락 방지용 (오류 방지)
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"  # 서버 환경에 따라 "egl"로 자동 변경 가능

import inspect
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from pathlib import Path
import cv2
import argparse

import xarm_rl  # noqa: F401  registers envs

def run_task_and_collect_data(
        mode="load",                # "train", "random", "load" 중 선택
        model_path="best_model.zip", # 로드하거나 저장할 모델 경로
        env_id="XArm6Reach-v0",
        save_dir="expert_data",
        num_episodes=100,
        past_frames=4,
        img_size=(64, 64),
        total_timesteps=50000       # "train" 모드 시 학습할 총 스텝 수
):
    """
    모드별(학습, 랜덤, 정적 로드)로 policy를 구동하고,
    최종 상태에서 d3il 규격의 [.npz] 이미지 기반 데이터셋을 수집하여 저장합니다.
    """
    # 1. 환경 생성
    print(f"[*] [{env_id}] 환경 생성 중... (렌더링 모드: rgb_array)")
    env = gym.make(env_id, render_mode="rgb_array")

    try:
        actual_file_path = inspect.getfile(env.unwrapped.__class__)
        print("\n" + "=" * 70)
        print("🎯 [경로 검증 결과] 현재 실행 중인 환경 클래스의 실제 소스 파일 위치:")
        print(f"👉 {actual_file_path}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"[오류] 경로 추적 실패: {e}")

    model = None

    # 2. 선택된 모드에 따른 Policy 정의/학습/로드
    print(f"▶ 실행 모드 설정: [{mode.upper()}]")

    if mode == "train":
        print(f"[*] 새로운 SAC 모델을 생성하여 {total_timesteps} 스텝 동안 학습을 시작합니다...")
        # 환경에 최적화된 하이퍼파라미터 구조로 SAC 인스턴스 생성
        model = SAC("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./sac_xarm_tensorboard/")
        model.learn(total_timesteps=total_timesteps)

        # 학습 완료 후 모델 저장
        model.save(model_path)
        print(f"[+] 모델 학습 완료 및 저장 완료: {model_path}")

    elif mode == "load":
        if not os.path.exists(model_path):
            print(f"[❌ 오류] '{mode}' 모드 실패: 학습된 모델 파일({model_path})이 존재하지 않습니다.")
            env.close()
            return
        print(f"[*] 기존 체크포인트 모델 로드 중: {model_path}")
        model = SAC.load(model_path, env=env, device="cpu")

    elif mode == "random":
        print("[*] 학습된 모델 없이, 환경의 Action Space를 무작위로 샘플링하여 궤적을 수집합니다.")
        model = None  # 무작위 행동 모드 표기

    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}. ('train', 'random', 'load' 중 선택)")

    # 3. 데이터셋 수집 루프 진입
    obs_list, nobs_list, act_list, rew_list, don_list = [], [], [], [], []
    ims_list, ids_list = [], []

    total_transitions = 0
    print(f"[*] 데이터셋 수집 시작합니다. (목표 에피소드: {num_episodes}개)")

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        done = False

        ep_obs, ep_nobs, ep_acts, ep_rews, ep_dones, ep_raw_ims = [], [], [], [], [], []

        while not done:
            # --- [모드별 액션 샘플링 제어] ---
            if model is not None:
                # 'train'에서 방금 학습했거나 'load'에서 가져온 모델로 행동 결정
                action, _ = model.predict(obs, deterministic=True)
            else:
                # 'random' 모드일 때 완전히 무작위 액션 추출
                action = env.action_space.sample()

            # --- [렌더링 타이밍 정석 변경] ---
            # 액션을 수행하기 전, 현재 시점(t)의 올바른 관찰 카메라 화면을 캡처합니다.
            frame = env.render()
            if frame is None:
                frame_resized = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            else:
                frame_resized = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_AREA)

            # 환경 step 실행
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 리스트 빌딩
            ep_obs.append(obs)
            ep_nobs.append(next_obs)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_dones.append(done)
            ep_raw_ims.append(frame_resized)

            obs = next_obs
            total_transitions += 1

        # 에피소드 종료 후 프레임 스태킹 윈도우 계산 (d3il 규격화)
        ep_len = len(ep_acts)
        ep_stacked_ims = []
        for t in range(ep_len):
            window = []
            for f in range(past_frames):
                idx = max(0, t - f)
                window.append(ep_raw_ims[idx])
            ep_stacked_ims.append(window)

        # 배치 리스트 추가
        obs_list.append(np.array(ep_obs, dtype=np.float32))
        nobs_list.append(np.array(ep_nobs, dtype=np.float32))
        act_list.append(np.array(ep_acts, dtype=np.float32))
        rew_list.append(np.array(ep_rews, dtype=np.float32))
        don_list.append(np.array(ep_dones, dtype=bool))
        ims_list.append(np.array(ep_stacked_ims, dtype=np.uint8))
        ids_list.append(np.full((ep_len,), ep_idx, dtype=np.int32))

    env.close()

    # 4. 전체 데이터 병합 (d3il 포맷)
    d3il_dataset = {
        'obs': np.concatenate(obs_list, axis=0),
        'nobs': np.concatenate(nobs_list, axis=0),
        'act': np.concatenate(act_list, axis=0),
        'rew': np.concatenate(rew_list, axis=0),
        'don': np.concatenate(don_list, axis=0),
        'ims': np.concatenate(ims_list, axis=0),
        'ids': np.concatenate(ids_list, axis=0),
        'n': total_transitions
    }

    # 5. npz 파일 압축 저장
    target_path = Path(save_dir) / env_id
    target_path.mkdir(parents=True, exist_ok=True)

    # 모드에 따라 데이터셋 파일 이름을 분리해두면 비교 실험 시 매우 편리합니다.
    filename = f"expert_{mode}_compressed.npz" if mode != "load" else "expert_compressed.npz"
    full_filepath = target_path / filename

    np.savez_compressed(
        full_filepath,
        obs=d3il_dataset['obs'],
        nobs=d3il_dataset['nobs'],
        act=d3il_dataset['act'],
        rew=d3il_dataset['rew'],
        don=d3il_dataset['don'],
        ims=d3il_dataset['ims'],
        ids=d3il_dataset['ids'],
        n=d3il_dataset['n']
    )

    print(f"📊 [{mode.upper()} 모드 수집 완료] 최종 데이터셋 파일 저장 위치: {full_filepath} (총 스텝 수: {total_transitions})")

if __name__ == "__main__":
    # 터미널에서 파이썬을 실행할 때 인자를 주입받을 수 있도록 파서 설정
    parser = argparse.ArgumentParser(description="xArm6 Sim2Real 및 d3il 데이터 수집 스크립트 툴킷")
    parser.add_argument("--mode", type=str, default="random", choices=["train", "random", "load"],
                        help="실행 모드 선택: 신규 학습 후 수집(train), 무작위 행동 수집(random), 기존 가중치 로드 후 수집(load)")
    parser.add_argument("--model_path", type=str, default="best_model.zip", help="체크포인트 모델 파일 이름 또는 저장할 이름")
    parser.add_argument("--episodes", type=str, default="2", help="수집할 에피소드 수")
    parser.add_argument("--timesteps", type=int, default=50000, help="train 모드 시 수행할 총 학습 타임스텝 수")

    args = parser.parse_args()

    # XArm6Pusher - v0
    # 실행 환경 스크립트 호출
    run_task_and_collect_data(
        mode=args.mode,
        model_path=args.model_path,
        env_id="XArm6Reach-v0",
        # env_id="XArm6Pusher-v0",
        save_dir="expert_data",
        num_episodes=int(args.episodes),
        past_frames=4,
        img_size=(64, 64),
        total_timesteps=args.timesteps
    )