export OPENCV_IO_ENABLE_OPENEXR=1
export PYREALSENSE2_IGNORE_UDEV=1  # udev 체크 우회

#!/usr/bin/env bash
set -euo pipefail

# 1. 경로 및 기존 환경 변수 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

# 혹시 카메라 프로세스가 점유되어 있다면 해제
"${SCRIPT_DIR}/05_release_cameras.sh"

# ==========================================
# 2. 파라미터 세팅 (실험 목적에 맞게 수정 가능)
# ==========================================
NUM_EPISODES=20       # 수집할 총 에피소드 수
MAX_STEPS=150         # 에피소드당 최대 타임스텝 (Pusher=150, Reach=50)
ACTION_SCALE="0.03"   # 스텝당 관절 움직임 변화량 한계 (Radian)
OUTPUT_DIR="./data/d3il_random_xarm6" # 데이터셋(.npz)이 저장될 경로

# ==========================================
# 3. 데이터 수집 스크립트 실행
# ==========================================
# 조이스틱 관련 아규먼트(--device, --deadzone 등)와
# LeRobot 관련 아규먼트(--repo-id, --task)를 모두 제거하고 새 파이썬 스크립트를 호출합니다.

python scripts/random_collect_d3il.py \
  --ip "${ROBOT_IP}" \
  --wrist-serial "${WRIST_CAMERA_SERIAL}" \
  --front-serial "${FRONT_CAMERA_SERIAL}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --action-scale "${ACTION_SCALE}" \
  --output-dir "${OUTPUT_DIR}"

echo "🎉 [수집 완료] 모든 무작위 데이터셋이 ${OUTPUT_DIR}에 정상적으로 저장되었습니다."