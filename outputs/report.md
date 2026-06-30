# xArm6 Reach — 성공한 학습 조합 보고서

> Task: `XArm6Reach-v0` (state-based, MuJoCo)
> Success criterion: end-effector ↔ target 거리 < 3 cm (50 episodes deterministic eval)

---

## 🏆 성공 조합 (success rate ≥ 80%)

| # | 알고리즘 | 학습 step | DR | safe-zone penalty | success | mean reward | final dist |
|---|---|---|---|---|---|---|---|
| 1 | **PPO+DR (배포 1순위)** | 3M | ✅ | ✅ | **86%** | +3.40 | 3.36 cm |
| 2 | PPO v2 (baseline) | 1.5M | ❌ | ❌ | 86% | +3.54 | 3.22 cm |
| 3 | SAC v3 (best @ 200k ckpt) | 200k | ❌ | ❌ | 86% | +3.37 | 3.52 cm |
| 4 | **SAC+DR (best @ 400k ckpt)** | 400k | ✅ | ✅ | **84%** | +2.95 | 3.77 cm |
| 5 | SAC+DR (final, 500k) | 500k | ✅ | ✅ | 82% | +2.55 | 4.05 cm |
| 6 | SAC v3 (final) | 250k | ❌ | ❌ | 80% | +2.73 | 3.58 cm |

모델 파일:
- `outputs/reach_ppo_dr/final_model.zip` — **PPO+DR (실제 배포 1순위)**
- `outputs/reach_sac_dr/best_model.zip` — **SAC+DR (실제 배포 2순위, 400k ckpt copy)**
- `outputs/reach_sac_dr/final_model.zip` — SAC+DR final (500k)
- `outputs/reach_ppo_v2/final_model.zip` — PPO baseline
- `outputs/reach_sac_v3/best_model.zip` — SAC v3 best (200k ckpt copy)
- `outputs/reach_sac_v3/final_model.zip` — SAC v3 final

**모든 6개 모델이 시뮬 grid tour 9/9 달성** ([scripts/demo_grid_tour.py](../scripts/demo_grid_tour.py)).

---

## 🔧 조합 1 — PPO v2 (가장 안정적인 추천)

### 환경
| 항목 | 값 |
|---|---|
| Env | `XArm6Reach-v0` |
| obs dim | 21 (joint_pos 6 + joint_vel 6 + ee_pos 3 + target_pos 3 + diff 3) |
| action dim | 6 (joint deltas, scale 0.05 rad) |
| max_episode_steps | 100 |
| control freq | 50 Hz (frame_skip=10 × 0.002s) |
| reward | `-dist - 0.001·‖a‖² + 10·success` |

### 알고리즘 하이퍼파라미터
```python
PPO(
    "MlpPolicy", vec_env,
    n_steps=256,          # 짧은 episode(100)에 맞춰 작게
    batch_size=256,
    n_epochs=10,
    learning_rate=5e-4,   # 기본 3e-4 → 더 공격적
    gamma=0.98,           # short horizon
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,       # 핵심: 기본 0.0 → 약한 entropy bonus로 탐색 강화
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=7,
)
```

### 실행 환경
- n_envs = **16** (SubprocVecEnv)
- 총 1,500,000 timesteps
- CPU 학습 (~45분)
- seed = 7

### 왜 v1은 실패하고 v2는 성공했나?
| 파라미터 | v1 (실패, 0%) | v2 (성공, 86%) | 이유 |
|---|---|---|---|
| `learning_rate` | 3e-4 | **5e-4** | 더 빠른 정책 갱신 |
| `n_steps` | 1024 | **256** | episode 100보다 짧게 — advantage 추정 정확도 ↑ |
| `ent_coef` | 0.0 | **0.005** | 결정론적 plateau 탈출 (탐색 ↑) |
| `gamma` | 0.99 | **0.98** | short-horizon에 맞춰 effective horizon 단축 |
| `n_envs` | 8 | 16 | data throughput ↑ |

핵심은 **(a) 짧은 horizon에 맞춘 n_steps 축소** + **(b) entropy bonus 추가**. 둘 중 하나만 빠져도 plateau에 갇혔습니다.

---

## 🔧 조합 2 — SAC v3 best (sample-efficient)

### 환경 — PPO v2와 동일

### 알고리즘 하이퍼파라미터
```python
SAC(
    "MlpPolicy", vec_env,
    learning_rate=3e-4,
    buffer_size=500_000,
    batch_size=256,
    tau=0.01,                       # 기본 0.005 → 더 빠른 target update
    gamma=0.95,                     # short horizon
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto_0.1",            # 초기 0.1에서 자동 감소
    target_entropy=-6.0,            # 핵심: = -action_dim, 결정론적 수렴 압력
    learning_starts=10_000,
    policy_kwargs=dict(net_arch=[256, 256]),
    seed=23,
)
```

### 실행 환경
- n_envs = 1 (off-policy, replay buffer)
- 200,000 timesteps에 best 체크포인트 도달 (250k까지 학습은 했으나 best는 중간)
- CPU 학습 (~75분)
- seed = 23

### 왜 v1/v2는 실패하고 v3는 성공했나?
| 버전 | 설정 | 결과 | 문제 |
|---|---|---|---|
| v1 | `ent_coef="auto"` (target=-action_dim=-6, 자동), default 나머지 | 68% (불안정) | entropy가 진동, 정책이 탐색만 함 |
| v2 | `ent_coef=0.05` 고정 | 정체 (중단) | 너무 낮게 잡혀 탐색 부족 → 학습 멈춤 |
| v3 | `ent_coef="auto_0.1"` + `target_entropy=-6.0` 명시 | **86%** | 초기 탐색은 충분, target_entropy 명시로 결정론 압력 |

핵심: SAC는 `target_entropy`를 명시적으로 잡아주는 것이 진동 억제에 결정적.

### SAC 운영 팁
- **final이 항상 best가 아님**. SAC는 진동이 있어 중간 체크포인트가 더 좋을 수 있음.
- → `CheckpointCallback`으로 50k마다 저장 + 평가로 best 선정 (이번에 200k가 best였음)

---

## 📊 PPO v2 vs SAC v3 비교

| 항목 | PPO v2 | SAC v3 best |
|---|---|---|
| Wall-clock 학습 시간 | ~45분 (n_envs 16, CPU) | ~60분 (n_envs 1, 200k 도달까지) |
| Sample efficiency | 1.5M steps 필요 | **200k steps** ← 7.5배 효율적 |
| 안정성 | 매우 안정 | 진동 있음 (best ≠ final) |
| 성공률 | 86% | 86% |
| 평균 도달 시간 | 43.7 step | 45.6 step |
| 추천 용도 | 안정된 baseline | 빠른 실험/iteration |

**결론**: 단일 모델 골라야 하면 **PPO v2** (안정성 ↑, final 그대로 쓸 수 있음). 빠르게 학습 비교 실험하려면 **SAC v3**.

---

## ❌ 실패 조합 (참고용)

| # | 모델 | success | 실패 원인 |
|---|---|---|---|
| - | PPO v1 (default) | 0% | `n_steps=1024` + `ent_coef=0.0` → plateau |
| - | SAC v1 (ent auto only) | 68% | entropy 진동, target_entropy 자동 |
| - | SAC v2 (`ent_coef=0.05` 고정) | (정체, 중단) | entropy 너무 낮음 → 탐색 부족 |

---

## 🔁 재현 방법

```bash
cd xArm-project
source .venv/bin/activate

# PPO v2 재학습
python scripts/train.py --task reach --algo ppo \
    --n_envs 16 --timesteps 1500000 --seed 7 \
    --out outputs/reach_ppo_v2_repro

# SAC v3 재학습 (200k에서 best ckpt가 나오는지 확인하려면 250k)
python scripts/train.py --task reach --algo sac \
    --n_envs 1 --timesteps 250000 --seed 23 \
    --out outputs/reach_sac_v3_repro

# 평가
python scripts/eval_headless.py --task reach --algo ppo \
    --model outputs/reach_ppo_v2_repro/final_model.zip --episodes 50
```

하이퍼파라미터는 [scripts/train.py](../scripts/train.py)의 `build_ppo()` / `build_sac()` 함수에 그대로 코드화되어 있습니다.

---

## 🎞️ GIF 생성 (헤드리스 OK)

학습된 정책의 roll-out을 GIF로 저장합니다. DISPLAY 없는 서버에서도 동작 (MuJoCo `EGL` offscreen backend).

### 한 줄 사용법
```bash
# PPO v2
MUJOCO_GL=egl python scripts/render_gif.py --task reach --algo ppo \
    --model outputs/reach_ppo_v2/final_model.zip \
    --out outputs/reach_ppo_v2/rollout.gif --episodes 3

# SAC v3 best
MUJOCO_GL=egl python scripts/render_gif.py --task reach --algo sac \
    --model outputs/reach_sac_v3/best_model.zip \
    --out outputs/reach_sac_v3/rollout_best.gif --episodes 3
```
(스크립트가 `MUJOCO_GL=egl` 자동 세팅하므로 env var은 생략 가능)

생성된 파일:
- [reach_ppo_v2/rollout.gif](reach_ppo_v2/rollout.gif) — 2.7 MB
- [reach_sac_v3/rollout_best.gif](reach_sac_v3/rollout_best.gif) — 2.4 MB

### 옵션
| 옵션 | 의미 | 기본 |
|---|---|---|
| `--episodes` | 몇 에피소드 녹화 | 3 |
| `--width` / `--height` | 해상도 | 480 × 360 |
| `--fps` | GIF 재생 속도 | 30 |
| `--render-every N` | N step마다 한 프레임 (낮을수록 부드럽지만 큼) | 2 |
| `--camera` | 카메라 (이름/ID, -1=free cam) | -1 |
| `--seed` | 평가 시드 | 2000 |

### 크기 줄이기 (gif가 너무 클 때)
1. `--render-every 4` 로 프레임 절반
2. `--width 320 --height 240` 해상도 ↓
3. 또는 MP4가 더 효율적 — `imageio.mimsave('out.mp4', frames, fps=30)`로 바꾸면 ~10배 작음

### 의존성
- `imageio` + `imageio-ffmpeg` (MP4용)
- `mujoco>=3.x` (EGL backend 내장)
- 시스템에 EGL 라이브러리 필요 (대부분의 Linux에 이미 있음)

---

## 🤖 실제 xArm6 (UFactory) 배포 — 노트북 연결 프로세스

### 전제
- xArm6 컨트롤러 박스 + 노트북을 같은 네트워크에 연결
- xArm6 컨트롤러는 기본 IP `192.168.1.xxx` (UFactory 출고 기본 보통 `192.168.1.185`)
- 노트북에 학습된 정책 파일 + 본 레포 코드 보유

### Step 0 — 사전 점검 (10분)
1. **컨트롤러 펌웨어**: UFactory Studio (제조사 GUI)로 연결 → 펌웨어 최신화
2. **xArm Studio에서 수동 동작 확인**: home position 이동, joint 각각 점프 — 모터 정상 확인
3. **Safe zone 설정**: xArm Studio → Settings → Safety → workspace box를
   `x: 0~570 mm, y: -540~550 mm, z: 180~600 mm`로 등록 (사용자 알려준 값)
4. **E-stop 위치 확보**: 컨트롤러 e-stop 버튼이 손 닿는 거리에

### Step 1 — 노트북 네트워크 설정 (5분)
```bash
# 노트북 유선 LAN을 컨트롤러와 직결 (또는 같은 스위치)
# 노트북 IP를 같은 서브넷에 고정 — 예시:
sudo ip addr add 192.168.1.10/24 dev eth0   # Linux
# Windows: 제어판 → 네트워크 → 어댑터 설정 → IPv4 → 수동 192.168.1.10 / 255.255.255.0

# 통신 확인
ping 192.168.1.185
```

### Step 2 — 노트북에 환경 설치 (10분)
```bash
# 본 레포 clone (또는 USB로 복사)
git clone <this-repo> xArm-project
cd xArm-project

# Python 3.11 venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[real]"     # mujoco + sb3 + xArm-Python-SDK

# 정책 파일 복사
mkdir -p outputs/reach_ppo_v2
# (학습 머신에서 outputs/reach_ppo_v2/final_model.zip 복사)
```

### Step 3 — Dry-run (정책이 멀쩡한지 검증, 5분)
**실제 모터 동작 없음.** 정책 출력만 확인.
```bash
python scripts/deploy_real.py --task reach \
    --model outputs/reach_ppo_v2/final_model.zip \
    --ip 192.168.1.185 \
    --target 0.45 0.0 0.55 \
    --dry-run
```
출력: 매 step마다 `q=[..] a=[..] ee=[..]` 찍힘. action 값이 [-1, 1] 범위 안에 있고, 시간에 따라 합리적으로 변하면 OK.

### Step 4 — 실제 동작 (천천히, 안전 우선)
```bash
# 첫 시도: 30% 속도, 20Hz control
python scripts/deploy_real.py --task reach \
    --model outputs/reach_ppo_v2/final_model.zip \
    --ip 192.168.1.185 \
    --target 0.45 0.0 0.55 \
    --speed 30 --hz 20
```

스크립트가 내부적으로 수행하는 것:
1. `XArmAPI(ip)` 연결
2. `motion_enable(True)` + `set_mode(1)` (servo motion mode) + `set_state(0)`
3. `move_gohome()` 으로 home 자세 이동 (사용자가 등록한 home)
4. **루프 (1/hz Hz)**:
   - `arm.get_servo_angle(is_radian=True)` → joint state
   - `arm.get_position(is_radian=True)` → TCP xyz (mm → m 변환)
   - `obs = [q(6), qd(6), ee(3), target(3), diff(3)]` 구성
   - `policy.predict(obs)` → action ∈ [-1,1]^6
   - **Safe zone 가드**: 현재 TCP가 safe zone 박스 밖이면 `[SAFETY] STOPPING` 출력 후 중단
   - `target_q = q + action * 0.05` (시뮬과 동일 action_scale)
   - `arm.set_servo_angle_j(target_q, is_radian=True, speed=...)` 로 명령
5. target 거리 < 3 cm 도달 시 종료

### Step 5 — 문제 발생 시 즉시 대응
| 증상 | 원인 후보 | 대응 |
|---|---|---|
| 떨림(jitter)/덜컹임 | action_scale 너무 큼, control rate 부족 | `--action-scale 0.03 --hz 30` 시도 |
| 안 움직임 | `set_mode(1)` 실패, 모터 disable | xArm Studio에서 enable 재확인 |
| 잘못된 방향으로 움직임 | 시뮬과 실제의 joint sign/origin 차이 | 좌표계 캘리브레이션 필요 (아래) |
| safe zone 침범 즉시 중단 | 정상 동작 (가드 작동) | target 위치 조정 |
| E-stop 누름 | 안전 우선 | 원인 분석 후 재시작 |

### Step 6 — 좌표계 캘리브레이션 (필요시)
시뮬 MJCF의 joint 0 위치 = xArm 컨트롤러의 home 위치 라고 가정했습니다. 실제로 안 맞을 수 있음:
1. xArm Studio에서 home에 두기 → `arm.get_servo_angle()` 값 기록 (예: `[0, -17.2, -68.8, 0, 86.0, 0]` deg)
2. [base_env.py](xArm_rl/envs/base_env.py:25) 의 `HOME_QPOS` 비교 — 다르면 deploy 스크립트에서 시뮬↔실제 offset 추가
3. 시뮬에서 동일 자세 만든 뒤 TCP xyz와 실제 `arm.get_position()` 비교 — 일치하면 좌표계 OK

### ⚠️ 시뮬과 실제의 차이 (Reality Gap)
현재 본 정책은 **domain randomization 미적용**이라 첫 시도에 잘 안 될 수 있습니다:
- 시뮬 PD gain ≠ 실제 컨트롤러 PD
- 시뮬 timestep 0.002s, 실제 통신 latency 수 ms
- 시뮬 joint friction 가정값

대응:
1. **(쉬움)** 천천히 동작 (`--speed 20`, `--hz 15`) 후 점진적으로 빠르게
2. **(권장)** Domain randomization 추가 후 재학습 (mass ±20%, friction 0.5–1.5x, control delay 1–3 step, PD gain ±30%)
3. **(고급)** 실제에서 소량 데이터 수집 → residual policy로 보정

### 권장 첫 실험 시나리오
```
1. dry-run                                   ─ 정책 sanity (5 min)
2. real, target=home 근처 (5cm 이동)         ─ 안전 검증 (10 min)
3. real, target=workspace 중앙               ─ 보통 케이스 (15 min)
4. real, target=workspace 모서리             ─ 가장자리 케이스 (15 min)
5. 다양한 target 10개 자동 테스트             ─ success rate 측정 (30 min)
```

### 보안 / 운영 팁
- 노트북 ↔ 컨트롤러 직결 (인터넷 노출 X)
- `set_collision_sensitivity(3)` (높을수록 충돌 감지 민감) → 부딪히면 즉시 정지
- 로깅: `arm.get_servo_angle()`, `arm.get_state()`, `info` 매 step 저장 → 사후 분석용
- 컨트롤러 로그도 별도 보관 (xArm Studio → Logs)

---

## 📁 파일 인덱스

```
outputs/
├── report.md                              ← 본 문서
├── reach_ppo_v2/
│   ├── final_model.zip                    ← PPO v2 (86% success)
│   ├── eval.json
│   ├── monitor.csv
│   └── tb/                                ← TensorBoard logs
├── reach_sac_v3/
│   ├── final_model.zip                    ← SAC v3 final (80%)
│   ├── best_model.zip                     ← SAC v3 best (86%, 200k ckpt copy)
│   ├── eval_final.json
│   ├── eval_best.json
│   ├── monitor.csv
│   ├── tb/
│   └── ckpts/sac_*.zip                    ← 50k 간격 체크포인트
└── (실패 폴더: reach_ppo, reach_sac, reach_sac_v2)
```

GIF 파일은 각 모델 폴더에 `rollout*.gif`로 저장됩니다.

