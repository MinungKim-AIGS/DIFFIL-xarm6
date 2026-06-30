"""Train PPO or SAC on XArm6 Reach/PickPlace via Stable-Baselines3.

Usage:
    python scripts/train.py --task reach --algo ppo
    python scripts/train.py --task reach --algo sac
    python scripts/train.py --task pick_place --algo ppo
"""
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO
from pathlib import Path
import cv2

import xarm_rl  # noqa: F401  registers envs

def collect_and_save_sim2real_dataset(
        model_path,
        env_id="XArm6Reach-v0",
        save_dir="expert_data",
        num_episodes=100,
        past_frames=4,
        img_size=(64, 64)
):
    """
    오른쪽 xArm6 환경과 학습된 모델을 사용하여 데이터를 수집하고,
    왼쪽 d3il(buffers.py) 구조에 맞추어 [.npz] 압축 파일로 저장합니다.
    """
    # 1. 환경 생성 (렌더링 모드를 rgb_array로 설정하여 이미지 추출 가능하게 함)
    import xarm_rl  # 환경 등록을 위해 필요
    env = gym.make(env_id, render_mode="rgb_array")

    # 2. 학습된 전문가 모델 로드 (SAC 예시, PPO일 경우 PPO.load 사용)
    model = SAC.load(model_path, env=env, device="cpu")
    # model = SAC.load(model_path)

    obs_list, nobs_list, act_list, rew_list, don_list = [], [], [], [], []
    ims_list, ids_list = [], []

    total_transitions = 0
    print(f"[{env_id}] 환경에서 {num_episodes}개의 에피소드 수집을 시작합니다.")

    for ep_idx in range(num_episodes):
        obs, info = env.reset()
        done = False

        # 각 에피소드별 임시 저장 버퍼
        ep_obs, ep_nobs, ep_acts, ep_rews, ep_dones, ep_raw_ims = [], [], [], [], [], []

        while not done:
            # 모델로부터 액션 추출
            action, _ = model.predict(obs, deterministic=True)

            # 환경 렌더링을 통해 현재 프레임 이미지 획득 (Sim2Real 및 d3il 비주얼 학습용)
            frame = env.render()  # 기본 픽셀 이미지 [H, W, 3]
            if frame is None:
                # 빈 프레임으로 대체
                frame = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                frame_resized = frame
            else:
                frame_resized = cv2.resize(frame, dsize=img_size, interpolation=cv2.INTER_AREA)

            # 환경 Step 실행
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 데이터 저장
            ep_obs.append(obs)
            ep_nobs.append(next_obs)
            ep_acts.append(action)
            ep_rews.append(reward)
            ep_dones.append(done)
            ep_raw_ims.append(frame_resized)  # [uint8] 타입 유지

            obs = next_obs
            total_transitions += 1

        # 에피소드가 종료되면 d3il의 프레임 스태킹(past_frames) 구조 생성
        ep_len = len(ep_acts)
        ep_stacked_ims = []

        for t in range(ep_len):
            window = []
            for f in range(past_frames):
                # 과거 프레임이 부족한 초기 타임스텝(t - f < 0)은 첫 프레임으로 패딩
                idx = max(0, t - f)
                window.append(ep_raw_ims[idx])
            ep_stacked_ims.append(window)  # [past_frames, H, W, C] 구조 생성

        # 전체 리스트에 병합
        obs_list.append(np.array(ep_obs, dtype=np.float32))
        nobs_list.append(np.array(ep_nobs, dtype=np.float32))
        act_list.append(np.array(ep_acts, dtype=np.float32))
        rew_list.append(np.array(ep_rews, dtype=np.float32))
        don_list.append(np.array(ep_dones, dtype=bool))
        ims_list.append(np.array(ep_stacked_ims, dtype=np.uint8))
        ids_list.append(np.full((ep_len,), ep_idx, dtype=np.int32))

    # 3. 모든 에피소드 데이터를 하나의 큰 어레이로 concatenate (d3il buffers 규격)
    d3il_dataset = {
        'obs': np.concatenate(obs_list, axis=0),
        'nobs': np.concatenate(nobs_list, axis=0),
        'act': np.concatenate(act_list, axis=0),
        'rew': np.concatenate(rew_list, axis=0),
        'don': np.concatenate(don_list, axis=0),
        'ims': np.concatenate(ims_list, axis=0),  # 형태: [N, past_frames, H, W, C]
        'ids': np.concatenate(ids_list, axis=0),  # 형태: [N]
        'n': total_transitions
    }

    # 4. d3il 양식에 맞춰 파일 저장
    target_path = Path(save_dir) / env_id
    target_path.mkdir(parents=True, exist_ok=True)
    full_filepath = target_path / 'expert_compressed.npz'

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

    print(f"데이터셋 저장 완료 ({total_transitions} steps): {full_filepath}")


if __name__ == "__main__":
    # 윈도우 절대 경로를 r을 붙여서 Raw String으로 정확히 지정해줍니다.
    # 파일 확장자(.zip)까지 명시해주어야 stable-baselines3가 올바르게 로드합니다.
    MODEL_PATH = "best_model.zip"

    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 학습된 모델 파일({MODEL_PATH})을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    else:
        collect_and_save_sim2real_dataset(
            model_path=MODEL_PATH,
            env_id="XArm6Reach-v0",
            save_dir="expert_data",  # 데이터셋이 저장될 루팅 폴더
            num_episodes=100,  # 수집할 에피소드 개수
            past_frames=4,  # d3il buffers.py 규격
            img_size=(64, 64)  # d3il 이미지 가공 규격
        )

# 사용 예시:
# collect_and_save_sim2real_dataset(model_path="outputs/reach_sac/ckpts/best_model.zip")