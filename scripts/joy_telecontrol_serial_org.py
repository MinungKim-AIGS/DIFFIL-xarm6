"""Teleoperate xArm6 with a Linux joystick/gamepad and optionally record LeRobot v3 data.

Examples:
    # Teleoperation only
    python joy_stick/joy_telecontrol.py --ip 192.168.1.199 --device /dev/input/js0 --fps 10

    # Teleoperation + LeRobot recording
    python joy_stick/joy_telecontrol.py --ip 192.168.1.199 --device /dev/input/js0 --fps 10 --record \
        --repo-id kangkang9412/xarm6_joystick_demo \
        --root ./data/xarm6_joystick_demo \
        --task "pick up the object"

    # Teleoperation + dual-camera LeRobot recording with fixed RealSense serials
    python joy_stick/joy_telecontrol_serial.py --ip 192.168.1.199 --device /dev/input/js0 --fps 10 --record \
        --repo-id kangkang9412/xarm6_joystick_dualcam_demo \
        --root ./data/xarm6_joystick_dualcam_demo \
        --task "pick up the object" \
        --wrist-camera-serial 817512070394 \
        --front-camera-serial 261222078861
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import select
import shutil
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC = MONOREPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


# =========================
# Xbox 360 / Linux joydev mapping
# =========================

AXIS_LX = 0
AXIS_LY = 1
AXIS_LT = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_RT = 5
AXIS_DPAD_X = 6
AXIS_DPAD_Y = 7

BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
BTN_SELECT = 6
BTN_START = 7
BTN_MODE = 8
BTN_L_THUMB = 9
BTN_R_THUMB = 10

JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

# PPO home joint pose used by xarm_rl/envs/base_env.py and real deploy scripts.
PPO_HOME_QPOS_RAD = np.array([0.0, -0.3, -1.2, 0.0, 1.5, 0.0], dtype=np.float32)
TASK_INSTRUCTION_FILENAME = "task_instruction.txt"


def import_required(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"{module_name} is not installed. {install_hint}") from exc


def realsense_format_name(rs, fmt) -> str:
    if fmt == rs.format.bgr8:
        return "bgr8"
    if fmt == rs.format.rgb8:
        return "rgb8"
    return str(fmt)


def realsense_format_from_name(rs, name: str):
    if name == "bgr8":
        return rs.format.bgr8
    if name == "rgb8":
        return rs.format.rgb8
    raise ValueError(f"Unsupported RealSense color format: {name}")


def get_realsense_device_by_serial(rs, serial: str):
    ctx = rs.context()
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            return dev
    return None


def get_realsense_color_profiles(rs, dev) -> set[tuple[int, int, int, str]]:
    profiles = set()
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() != rs.stream.color:
                continue
            fmt = profile.format()
            if fmt not in (rs.format.bgr8, rs.format.rgb8):
                continue
            video_profile = profile.as_video_stream_profile()
            profiles.add(
                (
                    video_profile.width(),
                    video_profile.height(),
                    profile.fps(),
                    realsense_format_name(rs, fmt),
                )
            )
    return profiles


def format_camera_profile(profile: tuple[int, int, int, str]) -> str:
    width, height, fps, fmt = profile
    return f"{width}x{height}@{fps} {fmt}"


def configure_camera_profile(args: argparse.Namespace) -> None:
    if not args.record:
        return

    rs = import_required("pyrealsense2", "Install Intel RealSense SDK and pyrealsense2.")
    serials = [args.wrist_camera_serial]
    if args.use_front_camera:
        serials.append(args.front_camera_serial)

    supported_by_serial = {}
    for serial in serials:
        dev = get_realsense_device_by_serial(rs, serial)
        if dev is None:
            connected = [
                d.get_info(rs.camera_info.serial_number)
                for d in rs.context().query_devices()
            ]
            raise SystemExit(
                f"RealSense serial {serial} is not connected. "
                f"Connected serials: {connected if connected else 'none'}"
            )
        profiles = get_realsense_color_profiles(rs, dev)
        if not profiles:
            raise SystemExit(f"RealSense serial {serial} has no supported RGB/BGR color profiles.")
        supported_by_serial[serial] = profiles

    common_profiles = set.intersection(*supported_by_serial.values())
    preferred_profiles = [
        (args.width, args.height, args.camera_fps, "bgr8"),
        (args.width, args.height, args.camera_fps, "rgb8"),
        (args.width, args.height, 15, "bgr8"),
        (args.width, args.height, 15, "rgb8"),
        (640, 480, 15, "bgr8"),
        (640, 480, 15, "rgb8"),
        (424, 240, 30, "bgr8"),
        (424, 240, 30, "rgb8"),
        (424, 240, 15, "bgr8"),
        (424, 240, 15, "rgb8"),
        (320, 240, 30, "bgr8"),
        (320, 240, 30, "rgb8"),
        (320, 240, 15, "bgr8"),
        (320, 240, 15, "rgb8"),
    ]
    candidates = [profile for profile in preferred_profiles if profile in common_profiles]
    if not candidates:
        supported = {
            serial: ", ".join(format_camera_profile(profile) for profile in sorted(profiles))
            for serial, profiles in supported_by_serial.items()
        }
        raise SystemExit(f"No common RealSense color profile found for selected cameras: {supported}")

    selected = candidates[0]
    requested = (args.width, args.height, args.camera_fps, "bgr8")
    args.camera_profile_candidates = candidates
    args.width, args.height, args.camera_fps, args.camera_format = selected
    if selected != requested:
        print(
            "[camera] requested profile is not available for all selected cameras; "
            f"using {format_camera_profile(selected)}"
        )


def move_incomplete_lerobot_root(root: Path) -> Path | None:
    """Move aside a LeRobot root created by create() before any episode was saved."""
    if not root.exists():
        return None

    info_path = root / "meta" / "info.json"
    episode_meta_dir = root / "meta" / "episodes"
    tasks_path = root / "meta" / "tasks.parquet"
    data_dir = root / "data"
    videos_dir = root / "videos"
    images_dir = root / "images"
    is_empty = not any(root.iterdir())
    has_saved_episode_metadata = episode_meta_dir.exists() or tasks_path.exists()
    has_saved_episode_payload = data_dir.exists() or videos_dir.exists()
    has_only_temporary_images = images_dir.exists() and not has_saved_episode_metadata and not has_saved_episode_payload
    is_unsaved_lerobot_root = info_path.exists() and not has_saved_episode_metadata and not has_saved_episode_payload
    if not is_empty and not is_unsaved_lerobot_root and not has_only_temporary_images:
        return None

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = root.with_name(f"{root.name}.partial-{stamp}")
    suffix = 1
    while backup.exists():
        backup = root.with_name(f"{root.name}.partial-{stamp}-{suffix}")
        suffix += 1

    shutil.move(str(root), str(backup))
    return backup


def sanitize_task_id(task_id: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in task_id.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError("--task-id must contain at least one letter or number")
    return sanitized


def apply_task_root(args: argparse.Namespace) -> None:
    if not args.task_id:
        return

    task_id = sanitize_task_id(args.task_id)
    base_root = Path(args.root) if args.root is not None else Path("data")
    args.root = str(base_root if base_root.name == task_id else base_root / task_id)
    args.task_id = task_id


def normalize_instruction(text: str) -> str:
    return " ".join(text.strip().split())


def validate_or_write_task_instruction(root: Path | None, task: str) -> None:
    if root is None:
        return

    instruction_path = root / TASK_INSTRUCTION_FILENAME
    current = normalize_instruction(task)
    if not current:
        raise ValueError("task instruction is empty")

    if instruction_path.exists():
        expected = normalize_instruction(instruction_path.read_text(encoding="utf-8"))
        if expected != current:
            raise ValueError(
                "Task instruction mismatch for existing dataset root.\n"
                f"  root: {root}\n"
                f"  saved in {TASK_INSTRUCTION_FILENAME}: {expected!r}\n"
                f"  current --task: {current!r}\n"
                "Use the matching instruction, or choose a new TASK_ID/DATA_ROOT for a different task."
            )
        print(f"[joy-teleop] task instruction matched: {instruction_path}")
        return

    root.mkdir(parents=True, exist_ok=True)
    instruction_path.write_text(current + "\n", encoding="utf-8")
    print(f"[joy-teleop] wrote task instruction: {instruction_path}")


def validate_existing_task_instruction(root: Path | None, task: str) -> None:
    if root is None:
        return
    instruction_path = root / TASK_INSTRUCTION_FILENAME
    if not instruction_path.exists():
        return

    expected = normalize_instruction(instruction_path.read_text(encoding="utf-8"))
    current = normalize_instruction(task)
    if expected != current:
        raise ValueError(
            "Task instruction mismatch for existing dataset root.\n"
            f"  root: {root}\n"
            f"  saved in {TASK_INSTRUCTION_FILENAME}: {expected!r}\n"
            f"  current --task: {current!r}\n"
            "Use the matching instruction, or choose a new TASK_ID/DATA_ROOT for a different task."
        )


def existing_dataset_feature_keys(root: Path | None) -> set[str]:
    if root is None:
        return set()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return set()
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    features = info.get("features")
    return set(features) if isinstance(features, dict) else set()


def validate_existing_action_schema(root: Path | None, action_mode: str) -> None:
    keys = existing_dataset_feature_keys(root)
    if not keys:
        return

    has_absolute_extra = "action.absolute" in keys
    if action_mode == "both" and not has_absolute_extra:
        raise ValueError(
            "Existing dataset root does not contain 'action.absolute', but current --action-mode is 'both'.\n"
            f"  root: {root}\n"
            "Use DATA_ACTION_MODE=delta for this existing dataset, or choose a new TASK_ID/DATA_ROOT."
        )
    if action_mode != "both" and has_absolute_extra:
        raise ValueError(
            "Existing dataset root contains 'action.absolute', so it was created with --action-mode=both.\n"
            f"  root: {root}\n"
            "Use DATA_ACTION_MODE=both for this existing dataset, or choose a new TASK_ID/DATA_ROOT."
        )


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dz(v: float, deadzone: float) -> float:
    return 0.0 if abs(v) < deadzone else float(v)


def norm_axis(value: int) -> float:
    return clip(value / 32767.0, -1.0, 1.0)


def trigger_value(axis_value: float, trigger_deadzone: float) -> float:
    # Linux joydev exposes Xbox triggers at -1.0 when released and +1.0 when fully pressed.
    value = clip((axis_value + 1.0) * 0.5, 0.0, 1.0)
    return 0.0 if value < trigger_deadzone else value


def joystick_name(path: str) -> str:
    name_path = f"/sys/class/input/{os.path.basename(path)}/device/name"
    try:
        with open(name_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return path


class Gamepad:
    """Minimal non-blocking Linux joydev reader.

    This intentionally mirrors the previously verified demo.py reader, so the same
    /dev/input/js0 device and Xbox-style axis/button mapping can be reused.
    """

    def __init__(self, path: str):
        self.path = path
        self.name = joystick_name(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = [0.0] * 8
        self.buttons = [False] * 11

        # Xbox triggers rest at -1.0. Initializing them avoids a false half-press
        # before init events are received.
        self.axes[AXIS_LT] = -1.0
        self.axes[AXIS_RT] = -1.0

    def close(self) -> None:
        os.close(self.fd)

    def poll(self) -> None:
        while True:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise

            if len(data) != JS_EVENT_SIZE:
                break

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            is_init = bool(event_type & JS_EVENT_INIT)
            event_type &= ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS:
                while number >= len(self.axes):
                    self.axes.append(0.0)
                axis_value = norm_axis(value)
                if is_init and number in (AXIS_LT, AXIS_RT) and abs(axis_value) < 0.001:
                    axis_value = -1.0
                self.axes[number] = axis_value
            elif event_type == JS_EVENT_BUTTON:
                while number >= len(self.buttons):
                    self.buttons.append(False)
                self.buttons[number] = bool(value)

    def axis(self, number: int) -> float:
        if number < 0 or number >= len(self.axes):
            return 0.0
        return float(self.axes[number])

    def button(self, number: int) -> bool:
        if number < 0 or number >= len(self.buttons):
            return False
        return bool(self.buttons[number])

    def settle(self, seconds: float = 0.3) -> None:
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            self.poll()
            time.sleep(0.01)


def make_lerobot_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_action_names = [
        "delta_tcp_x_mm",
        "delta_tcp_y_mm",
        "delta_tcp_z_mm",
        "delta_tcp_roll_rad",
        "delta_tcp_pitch_rad",
        "delta_tcp_yaw_rad",
        "delta_gripper_pos",
    ]
    absolute_action_names = [
        "target_tcp_x_mm",
        "target_tcp_y_mm",
        "target_tcp_z_mm",
        "target_tcp_roll_rad",
        "target_tcp_pitch_rad",
        "target_tcp_yaw_rad",
        "target_gripper_pos",
    ]
    primary_action_names = absolute_action_names if args.action_mode == "absolute" else delta_action_names

    features = {
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "tcp_x_mm",
                "tcp_y_mm",
                "tcp_z_mm",
                "tcp_roll_rad",
                "tcp_pitch_rad",
                "tcp_yaw_rad",
                "gripper_pos",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": primary_action_names,
        },
    }

    if args.action_mode == "both":
        features["action.absolute"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": absolute_action_names,
        }

    if args.use_front_camera:
        features["observation.images.front"] = {
            "dtype": "video",
            "shape": (args.height, args.width, 3),
            "names": ["height", "width", "channel"],
        }

    root = Path(args.root) if args.root is not None else None
    if root is not None and root.exists():
        backup = move_incomplete_lerobot_root(root)
        if backup is not None:
            print(f"[joy-teleop] moved incomplete dataset stub to {backup}")
        else:
            validate_or_write_task_instruction(root, args.task)
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=root,
                batch_encoding_size=1,
                streaming_encoding=args.streaming_encoding,
            )
            if args.image_writer_processes or args.image_writer_threads:
                dataset.start_image_writer(
                    num_processes=args.image_writer_processes,
                    num_threads=args.image_writer_threads,
                )
            print(
                "[joy-teleop] resuming dataset "
                f"{root} ({dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames, "
                f"next EP{dataset.meta.total_episodes + 1})"
            )
            return dataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        robot_type="xarm6",
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        streaming_encoding=args.streaming_encoding,
    )
    validate_or_write_task_instruction(Path(dataset.root), args.task)
    print(f"[joy-teleop] created dataset {dataset.root} (next EP1)")
    return dataset


def list_realsense_devices() -> None:
    rs = import_required("pyrealsense2", "Install Intel RealSense SDK and pyrealsense2.")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return
    print("Connected RealSense devices:")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)
        usb_type = "unknown"
        if dev.supports(rs.camera_info.usb_type_descriptor):
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"  [{i}] {name}")
        print(f"      Serial   : {serial}")
        print(f"      Firmware : {firmware}")
        print(f"      USB      : {usb_type}")
        color_profiles = sorted(get_realsense_color_profiles(rs, dev))
        if color_profiles:
            preview = ", ".join(format_camera_profile(profile) for profile in color_profiles[:12])
            if len(color_profiles) > 12:
                preview += f", ... ({len(color_profiles)} total)"
            print(f"      Color    : {preview}")
        else:
            print("      Color    : none for bgr8/rgb8")


class RealSenseCamera:
    """Threaded RealSense RGB reader for wrist/front cameras.

    The robot control loop must not block on camera.wait_for_frames(), especially when
    two cameras are used. This class continuously keeps the latest RGB frame in a
    background thread, and the recorder samples that latest frame at args.fps.
    """

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        fps: int,
        serial: str | None = None,
        color_format: str = "bgr8",
    ):
        self.rs = import_required("pyrealsense2", "Install Intel RealSense SDK and pyrealsense2.")
        self.cv2 = import_required("cv2", "pip install opencv-python")
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.color_format = color_format
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(
            self.rs.stream.color,
            width,
            height,
            realsense_format_from_name(self.rs, color_format),
            fps,
        )
        self.lock = threading.Lock()
        self.latest_rgb: np.ndarray | None = None
        self.running = False
        self.started = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        serial_msg = self.serial if self.serial else "auto"
        try:
            self.pipeline.start(self.config)
            self.started = True
            self.running = True
            self.thread = threading.Thread(target=self._loop, name=f"realsense-{self.name}", daemon=True)
            self.thread.start()
            self.wait_until_ready(timeout_s=5.0)
        except Exception as exc:
            self.running = False
            if self.thread is not None:
                self.thread.join(timeout=1.0)
                self.thread = None
            if self.started:
                self.pipeline.stop()
                self.started = False
            raise RuntimeError(
                f"Failed to start RealSense camera '{self.name}' "
                f"(serial={serial_msg}, stream={self.width}x{self.height}@{self.fps}): {exc}"
            ) from exc
        print(f"[camera:{self.name}] started serial={serial_msg}, {self.width}x{self.height}@{self.fps}")

    def _loop(self) -> None:
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                if self.color_format == "rgb8":
                    rgb = color
                else:
                    rgb = self.cv2.cvtColor(color, self.cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.latest_rgb = rgb.copy()
            except Exception as exc:
                if self.running:
                    print(f"[camera:{self.name}] frame error: {exc}")
                time.sleep(0.01)

    def wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                if self.latest_rgb is not None:
                    return
            time.sleep(0.01)
        raise RuntimeError(f"RealSense camera '{self.name}' did not produce RGB frames within {timeout_s}s")

    def get_latest_rgb(self) -> np.ndarray:
        with self.lock:
            if self.latest_rgb is None:
                raise RuntimeError(f"RealSense camera '{self.name}' RGB frame is not available.")
            return self.latest_rgb.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if not self.started:
            return
        self.pipeline.stop()
        self.started = False
        print(f"[camera:{self.name}] stopped")


def make_recording_cameras(args: argparse.Namespace) -> dict[str, RealSenseCamera]:
    profiles = getattr(args, "camera_profile_candidates", [(args.width, args.height, args.camera_fps, args.camera_format)])
    last_error: Exception | None = None

    for profile in profiles:
        args.width, args.height, args.camera_fps, args.camera_format = profile
        cameras = {
            "wrist": RealSenseCamera(
                "wrist",
                args.width,
                args.height,
                args.camera_fps,
                args.wrist_camera_serial,
                args.camera_format,
            )
        }
        if args.use_front_camera:
            cameras["front"] = RealSenseCamera(
                "front",
                args.width,
                args.height,
                args.camera_fps,
                args.front_camera_serial,
                args.camera_format,
            )

        try:
            for cam in cameras.values():
                cam.start()
            return cameras
        except Exception as exc:
            last_error = exc
            for cam in cameras.values():
                cam.stop()
            print(f"[camera] failed with {format_camera_profile(profile)}: {exc}")

    raise RuntimeError(
        "Failed to start selected RealSense cameras with all common color profiles. "
        "Check USB bandwidth, camera serials, and whether another process is using the cameras. "
        "For errno=16 Device or resource busy, close RealSense Viewer/old robot clients, "
        "or find the owner with: fuser -v /dev/video*"
    ) from last_error


class XArm6ServoCartesian:
    """xArm wrapper for target-pose Cartesian servo teleoperation.

    It uses set_mode(1) + set_servo_cartesian(), matching the joystick demo that
    was already verified on the robot, while storing poses in the same 7D
    LeRobot schema used by space_telecontrol.py.
    """

    def __init__(self, ip: str, collision_sensitivity: int, enable_gripper: bool, gripper_speed: int):
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise SystemExit("xArm-Python-SDK is not installed. pip install xArm-Python-SDK") from exc

        self.arm = XArmAPI(ip, is_radian=True)
        self.gripper_speed = int(gripper_speed)
        time.sleep(0.5)

        if getattr(self.arm, "warn_code", 0) != 0:
            self.arm.clean_warn()
        if getattr(self.arm, "error_code", 0) != 0:
            self.arm.clean_error()

        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.set_collision_sensitivity(collision_sensitivity)
        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.5)

        if enable_gripper:
            self.setup_gripper(gripper_speed)

    def setup_gripper(self, gripper_speed: int) -> None:
        if hasattr(self.arm, "set_gripper_enable"):
            ret = self.arm.set_gripper_enable(True)
            print(f"[joy-teleop] set_gripper_enable ret={ret}")
        if hasattr(self.arm, "set_gripper_mode"):
            ret = self.arm.set_gripper_mode(0)
            print(f"[joy-teleop] set_gripper_mode ret={ret}")
        if hasattr(self.arm, "set_gripper_speed"):
            ret = self.arm.set_gripper_speed(gripper_speed)
            print(f"[joy-teleop] set_gripper_speed ret={ret}, speed={gripper_speed}")

    def read_state(self) -> np.ndarray:
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"get_position failed with code={code}")
        gripper = self.read_gripper()
        return np.array([*pose[:6], gripper], dtype=np.float32)

    def read_gripper(self) -> float:
        if not hasattr(self.arm, "get_gripper_position"):
            return 0.0
        code, pos = self.arm.get_gripper_position()
        return float(pos) if code == 0 else 0.0

    def send_target_pose(self, target_pose7: np.ndarray, speed: float, acc: float) -> int:
        target6 = [float(v) for v in target_pose7[:6]]
        return int(
            self.arm.set_servo_cartesian(
                target6,
                speed=float(speed),
                mvacc=float(acc),
                is_radian=True,
            )
        )

    def return_to_joint_home(
        self,
        home_qpos_rad: np.ndarray,
        gripper_target: float | None,
        speed: float,
        acc: float,
    ) -> np.ndarray:
        home_qpos_rad = np.asarray(home_qpos_rad, dtype=np.float32)
        if home_qpos_rad.shape != (6,):
            raise ValueError(f"home_qpos_rad must have shape (6,), got {home_qpos_rad.shape}")

        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.2)
        ret = int(
            self.arm.set_servo_angle(
                angle=[float(v) for v in home_qpos_rad],
                speed=float(speed),
                mvacc=float(acc),
                is_radian=True,
                wait=True,
            )
        )
        if ret != 0:
            raise RuntimeError(f"return to PPO joint home failed: ret={ret}, qpos={home_qpos_rad.tolist()}")

        if gripper_target is not None:
            self.command_gripper(float(gripper_target))

        self.arm.set_mode(1)
        self.arm.set_state(0)
        time.sleep(0.2)
        return self.read_state()

    def command_gripper(self, target: float) -> int | None:
        if not hasattr(self.arm, "set_gripper_position"):
            print("[joy-teleop] WARNING: xArm API has no set_gripper_position")
            return None

        target = float(target)
        try:
            return int(
                self.arm.set_gripper_position(
                    target,
                    wait=False,
                    speed=self.gripper_speed,
                    auto_enable=True,
                )
            )
        except TypeError:
            return int(self.arm.set_gripper_position(target, wait=False))

    def emergency_stop(self) -> None:
        try:
            self.arm.set_state(4)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self.arm.set_state(4)
        except Exception:
            pass
        self.arm.disconnect()


def clamp_target_pose(pose7: np.ndarray, args) -> np.ndarray:
    pose7 = pose7.copy()
    pose7[0] = np.clip(pose7[0], args.safe_x_mm[0], args.safe_x_mm[1])
    pose7[1] = np.clip(pose7[1], args.safe_y_mm[0], args.safe_y_mm[1])
    pose7[2] = np.clip(pose7[2], args.safe_z_mm[0], args.safe_z_mm[1])
    pose7[3:6] = ((pose7[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    pose7[6] = np.clip(pose7[6], args.gripper_min, args.gripper_max)
    return pose7


def joystick_to_pose_delta(pad: Gamepad, args, dt: float) -> tuple[np.ndarray, dict[str, float]]:
    lx = dz(pad.axis(AXIS_LX), args.deadzone)
    ly = dz(pad.axis(AXIS_LY), args.deadzone)
    lt = trigger_value(pad.axis(AXIS_LT), args.trigger_deadzone)
    rx = dz(pad.axis(AXIS_RX), args.deadzone)
    ry = dz(pad.axis(AXIS_RY), args.deadzone)
    rt = trigger_value(pad.axis(AXIS_RT), args.trigger_deadzone)

    z_input = dz(float(pad.button(args.z_up_button)) - lt, args.deadzone)
    yaw_input = dz(float(pad.button(args.yaw_pos_button)) - rt, args.deadzone)

    delta = np.zeros(7, dtype=np.float32)
    delta[0] = args.x_sign * ly * args.pos_gain * dt
    delta[1] = args.y_sign * lx * args.pos_gain * dt
    delta[2] = args.z_sign * z_input * args.pos_gain * dt
    delta[3] = args.roll_sign * rx * args.rot_gain_rad * dt
    delta[4] = args.pitch_sign * ry * args.rot_gain_rad * dt
    delta[5] = args.yaw_sign * yaw_input * args.rot_gain_rad * dt

    debug = {
        "lx": lx,
        "ly": ly,
        "lt": lt,
        "rx": rx,
        "ry": ry,
        "rt": rt,
        "z_input": z_input,
        "yaw_input": yaw_input,
    }
    return delta, debug


def button_edge(pad: Gamepad, button: int, previous: bool) -> tuple[bool, bool]:
    current = pad.button(button)
    return current and not previous, current


def read_keyboard_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1)


def enter_pressed() -> bool:
    key = read_keyboard_key()
    return key in ("\n", "\r")


def dataset_has_pending_frames(dataset) -> bool:
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if not episode_buffer:
        return False
    return int(episode_buffer.get("size", 0)) > 0


def save_pending_dataset(dataset) -> None:
    if dataset_has_pending_frames(dataset):
        dataset.save_episode()


def return_to_initial_pose(
    robot: XArm6ServoCartesian,
    current_pose7: np.ndarray,
    initial_pose7: np.ndarray,
    args,
) -> np.ndarray:
    return_home_mode = getattr(args, "return_home_mode", "cartesian-start")
    if return_home_mode == "joint-home":
        home_qpos_rad = np.asarray(getattr(args, "home_qpos_rad", PPO_HOME_QPOS_RAD), dtype=np.float32)
        home_gripper = getattr(args, "home_gripper", None)
        gripper_target = float(initial_pose7[6]) if home_gripper is None else float(home_gripper)
        print("[joy-teleop] returning to PPO joint home...")
        state7 = robot.return_to_joint_home(
            home_qpos_rad,
            gripper_target=gripper_target,
            speed=float(getattr(args, "return_home_joint_speed", 0.35)),
            acc=float(getattr(args, "return_home_joint_acc", 2.0)),
        )
        print(f"[joy-teleop] returned to PPO joint home: {np.round(state7, 3).tolist()}")
        return state7

    print("[joy-teleop] returning to initial state...")
    start = current_pose7.copy()
    goal = clamp_target_pose(initial_pose7, args)
    delta = goal - start
    delta[3:6] = ((delta[3:6] + math.pi) % (2.0 * math.pi)) - math.pi

    steps = max(2, int(args.return_home_seconds * args.control_hz))
    dt = 1.0 / args.control_hz
    for i in range(1, steps + 1):
        alpha = i / steps
        pose = start + alpha * delta
        pose = clamp_target_pose(pose, args)
        ret = robot.send_target_pose(pose, speed=args.servo_speed, acc=args.servo_acc)
        if ret != 0:
            raise RuntimeError(f"return to initial state failed: ret={ret}, target={pose.tolist()}")
        time.sleep(dt)

    robot.command_gripper(float(goal[6]))
    print("[joy-teleop] returned to initial state.")
    return goal


def record_delta_action_from_target(state7: np.ndarray, target_pose7: np.ndarray) -> np.ndarray:
    action7 = target_pose7.astype(np.float32) - state7.astype(np.float32)
    action7[3:6] = ((action7[3:6] + math.pi) % (2.0 * math.pi)) - math.pi
    return action7


def record_absolute_action_from_target(target_pose7: np.ndarray) -> np.ndarray:
    return target_pose7.astype(np.float32).copy()


def add_action_fields(frame: dict, state7: np.ndarray, target_pose7: np.ndarray, action_mode: str) -> None:
    if action_mode == "absolute":
        frame["action"] = record_absolute_action_from_target(target_pose7).astype(np.float32)
        return

    frame["action"] = record_delta_action_from_target(state7, target_pose7).astype(np.float32)
    if action_mode == "both":
        frame["action.absolute"] = record_absolute_action_from_target(target_pose7).astype(np.float32)


def teleop_loop(args, robot: XArm6ServoCartesian, cameras: dict[str, RealSenseCamera] | None, dataset) -> None:
    pad = Gamepad(args.device)
    pad.settle()
    old_terminal_attrs = None
    if sys.stdin.isatty():
        old_terminal_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    control_dt = 1.0 / args.control_hz
    record_dt = 1.0 / args.fps
    next_record_t = time.time()

    start_pose7 = robot.read_state()
    initial_pose7 = (
        np.asarray(args.initial_pose7, dtype=np.float32)
        if args.initial_pose7 is not None
        else start_pose7.copy()
    )
    target_pose7 = start_pose7.copy()
    gripper_target = float(target_pose7[6])
    discard_episode = False
    return_home_on_exit = False
    previous_close_held = False
    previous_open_held = False
    last_gripper_command_t = 0.0
    last_gripper_log_t = 0.0
    last_gripper_ret: int | None = None

    print("[joy-teleop] running")
    print(f"  device : {pad.name} ({args.device})")
    print("  left stick  : TCP X / Y")
    print("  LB / LT     : TCP Z up / down")
    print("  right stick : TCP roll / pitch")
    print("  RB / RT     : TCP yaw + / -")
    print("  close btn   : hold to close gripper")
    print("  open btn    : hold to open gripper")
    print("  SELECT      : save and exit")
    print(f"  return mode : {args.return_home_mode}")
    if args.return_home_mode == "joint-home":
        print(f"  home qpos   : {np.round(args.home_qpos_rad, 4).tolist()}")
    print("  Enter       : save, return home, and exit")
    print("  Esc         : discard episode, return home, and exit")
    print("  START       : emergency stop and discard episode")
    print("  Ctrl+C      : save and exit")

    try:
        while True:
            t0 = time.time()
            pad.poll()

            if pad.button(args.save_button):
                print("[joy-teleop] save button pressed.")
                break

            key = read_keyboard_key()
            if key in ("\n", "\r"):
                print("[joy-teleop] Enter pressed; save, return home, and exit.")
                return_home_on_exit = True
                break
            if key == "\x1b":
                print("[joy-teleop] Esc pressed; discard episode, return home, and exit.")
                discard_episode = True
                return_home_on_exit = True
                break

            if pad.button(args.emergency_button):
                print("[joy-teleop] EMERGENCY STOP button pressed.")
                discard_episode = args.discard_on_emergency
                robot.emergency_stop()
                break

            delta7, _ = joystick_to_pose_delta(pad, args, control_dt)
            target_pose7[:6] += delta7[:6]

            close_held = pad.button(args.gripper_close_button)
            open_held = pad.button(args.gripper_open_button)
            if close_held != previous_close_held:
                state = "pressed" if close_held else "released"
                print(f"[joy-teleop] gripper close button {state} (button={args.gripper_close_button})")
                if close_held:
                    gripper_target = float(np.clip(robot.read_gripper(), args.gripper_min, args.gripper_max))
                    target_pose7[6] = gripper_target
                    last_gripper_command_t = 0.0
                    print(f"[joy-teleop] gripper target synced to actual: {gripper_target:.1f}")
                previous_close_held = close_held
            if open_held != previous_open_held:
                state = "pressed" if open_held else "released"
                print(f"[joy-teleop] gripper open button {state} (button={args.gripper_open_button})")
                if open_held:
                    gripper_target = float(np.clip(robot.read_gripper(), args.gripper_min, args.gripper_max))
                    target_pose7[6] = gripper_target
                    last_gripper_command_t = 0.0
                    print(f"[joy-teleop] gripper target synced to actual: {gripper_target:.1f}")
                previous_open_held = open_held

            gripper_delta = 0.0
            if close_held and not open_held:
                gripper_delta = -args.gripper_rate * control_dt
            elif open_held and not close_held:
                gripper_delta = args.gripper_rate * control_dt

            if gripper_delta != 0.0:
                gripper_target += gripper_delta
                gripper_target = float(np.clip(gripper_target, args.gripper_min, args.gripper_max))
                now_t = time.time()
                if now_t - last_gripper_command_t >= 1.0 / args.gripper_command_hz:
                    last_gripper_ret = robot.command_gripper(gripper_target)
                    if last_gripper_ret not in (None, 0):
                        print(
                            "[joy-teleop] WARNING: set_gripper_position failed "
                            f"ret={last_gripper_ret}, target={gripper_target:.1f}"
                        )
                    last_gripper_command_t = now_t
                if now_t - last_gripper_log_t > 0.5:
                    direction = "close" if gripper_delta < 0 else "open"
                    actual_gripper = robot.read_gripper()
                    limit_note = ""
                    if gripper_target <= args.gripper_min + 1e-3:
                        limit_note = " min-limit"
                    elif gripper_target >= args.gripper_max - 1e-3:
                        limit_note = " max-limit"
                    print(
                        f"[joy-teleop] gripper {direction} "
                        f"target={gripper_target:.1f} actual={actual_gripper:.1f} "
                        f"ret={last_gripper_ret}{limit_note}"
                    )
                    last_gripper_log_t = now_t

            target_pose7[6] = gripper_target
            target_pose7 = clamp_target_pose(target_pose7, args)

            ret = robot.send_target_pose(target_pose7, speed=args.servo_speed, acc=args.servo_acc)
            if ret != 0:
                print(f"[joy-teleop] set_servo_cartesian failed: ret={ret}, target={target_pose7.tolist()}")
                discard_episode = args.discard_on_robot_error
                break

            now = time.time()
            if dataset is not None and now >= next_record_t:
                if cameras is None or "wrist" not in cameras:
                    raise RuntimeError("wrist camera is required when dataset is enabled")
                wrist_rgb = cameras["wrist"].get_latest_rgb()
                state7 = robot.read_state()
                frame = {
                    "observation.images.wrist": wrist_rgb,
                    "observation.state": state7.astype(np.float32),
                    "task": args.task,
                }
                add_action_fields(frame, state7, target_pose7, args.action_mode)
                if args.use_front_camera:
                    if "front" not in cameras:
                        raise RuntimeError("front camera is enabled but not available")
                    frame["observation.images.front"] = cameras["front"].get_latest_rgb()
                dataset.add_frame(frame)
                next_record_t += record_dt

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        if return_home_on_exit:
            target_pose7 = return_to_initial_pose(robot, target_pose7, initial_pose7, args)
    finally:
        if old_terminal_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_attrs)
        pad.close()
        if discard_episode and dataset is not None:
            dataset.clear_episode_buffer()
            print("[joy-teleop] episode buffer cleared.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.199", help="xArm controller IP")
    ap.add_argument("--device", default="/dev/input/js0", help="Linux joystick device path")
    ap.add_argument("--fps", type=int, default=10, help="LeRobot recording FPS")
    ap.add_argument("--control-hz", type=float, default=100.0, help="xArm servo Cartesian loop rate")
    ap.add_argument("--record", action="store_true", help="record one episode in LeRobot v3 format")
    ap.add_argument(
        "--action-mode",
        choices=["both", "delta", "absolute"],
        default="both",
        help=(
            "Action saved in dataset. both stores delta in 'action' and absolute target in "
            "'action.absolute'; delta/absolute store only that mode in 'action'."
        ),
    )
    ap.add_argument("--repo-id", default=None, help="Hugging Face dataset repo id")
    ap.add_argument("--root", default=None, help="local LeRobot dataset root")
    ap.add_argument("--task-id", default=None, help="folder name under --root for this task, e.g. TASK1 -> ./data/TASK1")
    ap.add_argument("--task", default="teleoperate xarm6 with joystick", help="task string saved with each frame")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument(
        "--wrist-camera-serial",
        "--camera-serial",
        dest="wrist_camera_serial",
        default=None,
        help="RealSense serial for observation.images.wrist. Required with --record.",
    )
    ap.add_argument(
        "--front-camera-serial",
        default=None,
        help="RealSense serial for observation.images.front. If provided, front camera recording is enabled.",
    )
    ap.add_argument(
        "--use-front-camera",
        action="store_true",
        help="Require/enable observation.images.front. Equivalent to providing --front-camera-serial, but still needs the serial.",
    )
    ap.add_argument("--list-cameras", action="store_true", help="print connected RealSense serial numbers and exit")
    ap.add_argument("--image-writer-threads", type=int, default=4)
    ap.add_argument("--image-writer-processes", type=int, default=0)
    ap.add_argument("--streaming-encoding", action="store_true")

    ap.add_argument("--deadzone", type=float, default=0.10)
    ap.add_argument("--trigger-deadzone", type=float, default=0.05)
    ap.add_argument("--pos-gain", type=float, default=80.0, help="mm/s at full stick deflection")
    ap.add_argument("--rot-gain-deg", type=float, default=25.0, help="deg/s at full stick deflection")
    ap.add_argument("--servo-speed", type=float, default=100.0)
    ap.add_argument("--servo-acc", type=float, default=1000.0)
    ap.add_argument("--return-home-seconds", type=float, default=3.0, help="seconds used for Cartesian return modes")
    ap.add_argument(
        "--return-home-mode",
        choices=["joint-home", "cartesian-start", "cartesian-fixed"],
        default="joint-home",
        help="Enter/Esc return target. joint-home uses the PPO HOME_QPOS joint pose.",
    )
    ap.add_argument(
        "--home-qpos-rad",
        nargs=6,
        type=float,
        default=PPO_HOME_QPOS_RAD.tolist(),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="joint home used by --return-home-mode joint-home, in radians",
    )
    ap.add_argument("--return-home-joint-speed", type=float, default=0.35, help="joint-home return speed in rad/s")
    ap.add_argument("--return-home-joint-acc", type=float, default=2.0, help="joint-home return acceleration in rad/s^2")
    ap.add_argument("--home-gripper", type=float, default=None, help="optional gripper target after joint-home return")
    ap.add_argument(
        "--initial-pose7",
        nargs=7,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW", "GRIPPER"),
        help="fixed TCP xyz/rpy/gripper target used by --return-home-mode cartesian-fixed",
    )

    ap.add_argument("--x-sign", type=float, default=-1.0)
    ap.add_argument("--y-sign", type=float, default=-1.0)
    ap.add_argument("--z-sign", type=float, default=1.0)
    ap.add_argument("--roll-sign", type=float, default=1.0)
    ap.add_argument("--pitch-sign", type=float, default=-1.0)
    ap.add_argument("--yaw-sign", type=float, default=1.0)

    ap.add_argument("--z-up-button", type=int, default=BTN_LB)
    ap.add_argument("--yaw-pos-button", type=int, default=BTN_RB)
    ap.add_argument("--gripper-close-button", type=int, default=BTN_Y)
    ap.add_argument("--gripper-open-button", type=int, default=BTN_X)
    ap.add_argument("--save-button", type=int, default=BTN_SELECT)
    ap.add_argument("--emergency-button", type=int, default=BTN_START)
    ap.add_argument("--discard-on-emergency", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--discard-on-robot-error", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gripper-min", type=float, default=0.0)
    ap.add_argument("--gripper-max", type=float, default=850.0)
    ap.add_argument("--gripper-rate", type=float, default=300.0, help="gripper position units per second while holding open/close buttons")
    ap.add_argument("--gripper-command-hz", type=float, default=15.0, help="max gripper set_position command rate while holding open/close buttons")
    ap.add_argument("--gripper-speed", type=int, default=5000)

    ap.add_argument("--collision-sensitivity", type=int, default=1)
    ap.add_argument("--safe-x-mm", nargs=2, type=float, default=[0.0, 570.0])
    ap.add_argument("--safe-y-mm", nargs=2, type=float, default=[-540.0, 550.0])
    ap.add_argument("--safe-z-mm", nargs=2, type=float, default=[180.0, 600.0])

    args = ap.parse_args()
    args.rot_gain_rad = math.radians(args.rot_gain_deg)
    args.camera_format = "bgr8"
    args.home_qpos_rad = np.asarray(args.home_qpos_rad, dtype=np.float32)
    if args.gripper_command_hz <= 0:
        ap.error("--gripper-command-hz must be positive")
    if args.return_home_mode == "cartesian-fixed" and args.initial_pose7 is None:
        ap.error("--initial-pose7 is required when --return-home-mode=cartesian-fixed")
    if args.initial_pose7 is not None and args.return_home_mode != "cartesian-fixed":
        args.return_home_mode = "cartesian-fixed"
    try:
        apply_task_root(args)
    except ValueError as exc:
        ap.error(str(exc))

    # Serial-fixed recording policy:
    #   --record always requires --wrist-camera-serial.
    #   --front-camera-serial enables dual-camera recording.
    #   --use-front-camera is kept as an explicit compatibility flag, but it
    #   still requires --front-camera-serial to avoid accidental device-index swaps.
    if args.front_camera_serial:
        args.use_front_camera = True

    if args.record and not args.repo_id:
        ap.error("--repo-id is required when --record is set")
    if args.record:
        try:
            dataset_root = Path(args.root) if args.root is not None else None
            validate_existing_task_instruction(dataset_root, args.task)
            validate_existing_action_schema(dataset_root, args.action_mode)
        except ValueError as exc:
            ap.error(str(exc))
    if args.record and not args.wrist_camera_serial:
        ap.error("--wrist-camera-serial/--camera-serial is required when --record is set")
    if args.use_front_camera and not args.front_camera_serial:
        ap.error("--front-camera-serial is required when front camera recording is enabled")
    if args.front_camera_serial and args.wrist_camera_serial and args.front_camera_serial == args.wrist_camera_serial:
        ap.error("wrist and front camera serials must be different")
    return args


def main() -> None:
    args = parse_args()
    dataset = None
    cameras: dict[str, RealSenseCamera] | None = None
    robot = None

    try:
        if args.list_cameras:
            list_realsense_devices()
            return

        if args.record:
            configure_camera_profile(args)
            cameras = make_recording_cameras(args)

        robot = XArm6ServoCartesian(
            ip=args.ip,
            collision_sensitivity=args.collision_sensitivity,
            enable_gripper=args.enable_gripper,
            gripper_speed=args.gripper_speed,
        )
        if args.record:
            dataset = make_lerobot_dataset(args)
        teleop_loop(args, robot, cameras, dataset)
    except KeyboardInterrupt:
        print("\n[joy-teleop] Ctrl+C received.")
    finally:
        if robot is not None:
            robot.stop()
        if cameras is not None:
            for cam in cameras.values():
                cam.stop()
        if dataset is not None:
            save_pending_dataset(dataset)
            dataset.finalize()
            print(f"[joy-teleop] LeRobot dataset finalized at {dataset.root}")


if __name__ == "__main__":
    main()
