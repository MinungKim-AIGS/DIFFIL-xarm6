import os
import numpy as np

# 🔥 [중요] GUI 창을 띄우지 않고 파일 저장만 가능하도록
# matplotlib 임포트 직후 백엔드를 'Agg'로 고정합니다.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def verify_and_save_plots(npz_path, episode_to_plot=0, output_dir="plots"):
    """
    저장된 .npz 데이터셋을 로드하고, 특정 에피소드의
    이미지 시퀀스, State, Action 그래프를 그린 뒤 서버에 파일로 저장합니다.
    """
    if not os.path.exists(npz_path):
        print(f"[오류] 데이터셋 파일을 찾을 수 없습니다: {npz_path}")
        return

    # 결과물이 저장될 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] 데이터셋 로딩 중: {npz_path}")
    data = np.load(npz_path)

    # 1. 특정 에피소드의 인덱스 추출
    ep_indices = np.where(data['ids'] == episode_to_plot)[0]
    if len(ep_indices) == 0:
        print(f"[경고] 에피소드 ID {episode_to_plot}번 데이터를 찾을 수 없습니다.")
        return

    print(f"[*] {episode_to_plot}번 에피소드 분석 중 (총 {len(ep_indices)} steps)...")

    ep_obs = data['obs'][ep_indices]
    ep_act = data['act'][ep_indices]
    ep_rew = data['rew'][ep_indices]
    ep_ims = data['ims'][ep_indices]

    # --- 시각화 1: State & Action & Reward 트렌드 플롯 ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    timesteps = np.arange(len(ep_indices))

    # (1) State (앞쪽 6개 Joint Position)
    axes[0].plot(timesteps, ep_obs[:, :6])
    axes[0].set_title(f"Episode {episode_to_plot}: Joint Positions (State 0~5)")
    axes[0].set_ylabel("Radians (rad)")
    axes[0].grid(True)

    # (2) Action (Joint Deltas)
    axes[1].plot(timesteps, ep_act)
    axes[1].set_title(f"Episode {episode_to_plot}: Actions (Joint Deltas)")
    axes[1].set_ylabel("Control Scale")
    axes[1].grid(True)

    # (3) Reward (fontweight 오타 수정 및 linewidth 적용)
    axes[2].plot(timesteps, ep_rew, color='crimson', linewidth=2.5)
    axes[2].set_title(f"Episode {episode_to_plot}: Step Rewards")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Reward")
    axes[2].grid(True)

    plt.tight_layout()

    # 💾 플롯을 이미지 파일로 저장 (plt.show() 대신 사용)
    trajectory_plot_path = os.path.join(output_dir, f"episode_{episode_to_plot}_trajectory.png")
    plt.savefig(trajectory_plot_path, dpi=200)
    plt.close(fig) # 메모리 해제
    print(f"[+] 대동적 궤적 그래프 저장 완료: {trajectory_plot_path}")


    # --- 시각화 2: 특정 스텝의 4프레임 과거 스태킹 이미지 상태 확인 ---
    sample_step = len(ep_indices) // 2
    sample_img_stacked = ep_ims[sample_step] # 형태: [4, 64, 64, 3]
    past_frames_n = sample_img_stacked.shape[0]

    fig_img, axes_img = plt.subplots(1, past_frames_n, figsize=(3 * past_frames_n, 3.5))

    for f_idx in range(past_frames_n):
        img_to_show = sample_img_stacked[f_idx]
        axes_img[f_idx].imshow(img_to_show)

        if f_idx == 0:
            axes_img[f_idx].set_title(f"Current (t)", color='blue', fontweight='bold')
        else:
            axes_img[f_idx].set_title(f"Past (t-{f_idx})")
        axes_img[f_idx].axis('off')

    fig_img.suptitle(f"Stacked Image Structure at Episode {episode_to_plot} (Step {sample_step})", fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 💾 스태킹 프레임 이미지를 파일로 저장
    image_plot_path = os.path.join(output_dir, f"episode_{episode_to_plot}_step_{sample_step}_frames.png")
    plt.savefig(image_plot_path, dpi=200)
    plt.close(fig_img) # 메모리 해제
    print(f"[+] 프레임 스태킹 이미지 저장 완료: {image_plot_path}")

if __name__ == "__main__":
    # 데이터 경로 규칙 매칭
    NPZ_PATH = "expert_data/xarm6_real_random_dataset_front.npz"

    # 분석 실행 및 plots/ 폴더 하위에 이미지 생성
    verify_and_save_plots(NPZ_PATH, episode_to_plot=0, output_dir="plots")