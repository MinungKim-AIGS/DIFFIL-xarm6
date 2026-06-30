#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

cd "${PROJECT_ROOT}"

"${SCRIPT_DIR}/05_release_cameras.sh"

EXTRA_ARGS=()
EXTRA_ARGS+=(
  --return-home-mode joint-home
  --home-qpos-rad 0.0 -0.3 -1.2 0.0 1.5 0.0
  --return-home-joint-speed 0.35
  --return-home-joint-acc 2.0
  --safe-x-mm 0 570
  --safe-y-mm -540 550
  --safe-z-mm 180 600
)
if [[ "${USE_FRONT_CAMERA}" == "1" ]]; then
  EXTRA_ARGS+=(--front-camera-serial "${FRONT_CAMERA_SERIAL}")
fi

python joy_stick/joy_telecontrol_serial.py \
  --ip "${ROBOT_IP}" \
  --device /dev/input/js0 \
  --fps "${FPS}" \
  --control-hz 100 \
  --record \
  --repo-id "${REPO_ID}" \
  --root "${DATA_ROOT}" \
  --task-id "${TASK_ID}" \
  --task "${TASK}" \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --wrist-camera-serial "${WRIST_CAMERA_SERIAL}" \
  --image-writer-threads 4 \
  --image-writer-processes 0 \
  --action-mode "${DATA_ACTION_MODE}" \
  --deadzone 0.10 \
  --trigger-deadzone 0.05 \
  --pos-gain 80 \
  --rot-gain-deg 25 \
  --servo-speed 35 \
  --servo-acc 300 \
  --gripper-rate 300 \
  --gripper-command-hz 8 \
  --gripper-close-button 3 \
  --gripper-open-button 2 \
  --gripper-min 0 \
  --gripper-max 850 \
  --gripper-speed 5000 \
  --z-up-button 4 \
  --yaw-pos-button 5 \
  --save-button 6 \
  --emergency-button 7 \
  --x-sign -1 \
  --y-sign -1 \
  --z-sign 1 \
  --roll-sign 1 \
  --pitch-sign -1 \
  --yaw-sign 1 \
  --collision-sensitivity 1 \
  --return-home-seconds 5 \
  "${EXTRA_ARGS[@]}"


# TASK_ID=TASK1 \
# TASK="pick up the bottle with the white cap and place it in the dark brown box" \
# FPS=10 \
# ./sh_scripts/10_collect_delta_dataset.sh
