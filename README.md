<<<<<<< HEAD
# xArm6 Sim2Real + DIFF-IL 사용자 가이드

이 문서 하나로 전체 프로젝트를 **이해하고 실행**할 수 있도록 정리했습니다. 처음 보는 사람도
따라올 수 있게 개념 → 환경 설정 → 동작 확인 → 전체 워크플로우 순서로 설명합니다.

> 빠르게 동작만 확인하려면 **[3. 빠른 점검(5분 dry-run)](#3-빠른-점검-5분-dry-run-하드웨어gpu-불필요)** 으로 바로 가세요.

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [전체 구조](#2-전체-구조)
3. [빠른 점검(5분 dry-run)](#3-빠른-점검-5분-dry-run-하드웨어gpu-불필요)
4. [환경 설정 — 3개의 분리된 venv](#4-환경-설정--3개의-분리된-venv)
5. [전체 워크플로우](#5-전체-워크플로우)
6. [레포 구조 / 파일 맵](#6-레포-구조--파일-맵)
7. [핵심 개념 요약](#7-핵심-개념-요약)
8. [안전 체크리스트](#8-안전-체크리스트-실로봇)
9. [트러블슈팅](#9-트러블슈팅)

---

## 1. 프로젝트 개요

**MuJoCo 시뮬레이션에서 학습한 정책을 실제 UFactory xArm6에 배포(sim2real)**하고, 나아가
**cross-domain imitation(DIFF-IL)** 으로 **시뮬(source)의 전문가**를 **실제 로봇(target)에서
온라인 모방**하는 프로젝트입니다.

- 태스크: **Reach** — end-effector를 테이블 위 **고정 goal**로 이동
- 시뮬: MuJoCo 3.x, RL: Stable-Baselines3(PPO/SAC), 실배포: xArm-Python-SDK
- DIFF-IL: 논문 *"Domain-Invariant Per-Frame Feature Extraction for Cross-Domain
  Imitation Learning with Visual Observations"* 의 DisentanGAIL 구조

**모든 것을 좌우하는 두 가지 설계 사실:**

- **정책은 state 기반**입니다. 입력 = 21차원 상태 `[q(6), q̇(6), ee_xyz(3), goal_xyz(3),
  goal-ee(3)]`, 출력 = 6차원 joint delta. 작아서 **노트북에서 TensorFlow 없이 50Hz**로 돕니다.
- **이미지는 보상 계산에만** 쓰입니다. DIFF-IL의 encoder/label/discriminator가 이미지로
  `R̂ = -log(1 - F_s·F_f)` 를 계산하며 이건 **서버에서만** 일어납니다. 카메라는 sim↔real의
  *시각적* 격차를 메우는 용도이고, 정책 입력(state)에는 도메인 격차가 없습니다.

---

## 2. 전체 구조

```
            SOURCE = 시뮬레이션                          TARGET = 실제 xArm6
   ┌───────────────────────────────┐          ┌────────────────────────────────┐
   │ XArm6Reach-v0 (MuJoCo)         │          │ RealRobotEnv (XArmAPI+RealSense)│
   │  전문가 demo  B^SE             │          │  랜덤 demo   B^TR               │
   │  랜덤  demo  B^SR              │          │  온라인 롤아웃 B^TL  ◀──────────┼─ π^TL (numpy actor)
   └───────────────┬───────────────┘          └───────────────┬────────────────┘
                   │ 고정 데이터셋(오프라인)                    │ trajectory (ZeroMQ, 온라인)
                   ▼                                           ▼
         ┌─────────────────────────────────────────────────────────────┐
         │  GPU 서버  —  learner_node.py (DIFF-IL / DisentanGAIL, TF)    │
         │  encoder p · decoder q^S/q^T · F_f · F_s · D_f · D_s · SAC    │
         │  보상 R̂ = -log(1 - F_s·F_f)   →   SAC 정책 학습              │
         └──────────────────────────────┬──────────────────────────────┘
                                        │ actor 가중치 (ZeroMQ PUB, 주기적)
                                        ▼  노트북에서 hot-swap
```

**4개의 데이터셋** (DIFF-IL의 핵심 입력):

| 이름 | 도메인 | 수집 | 만드는 법 |
|---|---|---|---|
| `B^SE` | sim 전문가 | 고정 | `collect_sim_demos.py --mode policy` |
| `B^SR` | sim 랜덤 | 고정 | `collect_sim_demos.py --mode random` |
| `B^TR` | real 랜덤 | 고정 | `run_real_reach_collect.py` |
| `B^TL` | real 학습자 | **온라인** | `actor_node.py`가 실시간 스트리밍 |

**두 개의 label**(논문 기여): `F_f` = frame **시간** label(goal에 가까운 뒤 프레임일수록 높음 →
진행/완수 보상), `F_s` = sequence **전문성** label(전문가 vs 랜덤). 보상은 둘의 곱.

---

## 3. 빠른 점검 (5분 dry-run, 하드웨어/GPU 불필요)

mock learner(TF 없음) + mock 로봇(하드웨어 없음)으로 **분산 루프 전체**를 확인합니다.
터미널 2개:

```bash
pip install pyzmq msgpack lz4 numpy

# 터미널 A — 가짜 learner (데이터 수신, 가짜 actor publish)
cd scripts/diffil
python learner_node.py --mock --publish-every 1

# 터미널 B — mock 팔 위 가짜 actor, 3 에피소드 전송
cd scripts/diffil
python actor_node.py --dry-run --server-host 127.0.0.1 --num-episodes 3 --max-steps 20
```

actor가 trajectory를 보내고, learner가 받고(`ims (N,4,64,64,3)`), actor가 `policy v1, v2…`로
**hot-swap** 되면 전송+actor+가중치 동기화 경로가 정상입니다.

같은 방식으로 개별 부품도 dry-run 가능:
```bash
python scripts/run_real_reach_collect.py --dry-run --num-episodes 2 --max-steps 20   # 실 수집기
python scripts/deploy_real.py --task reach --model <policy.zip> --dry-run             # 실 배포
```

---

## 4. 환경 설정 — 3개의 분리된 venv

DIFF-IL 학습 스택(**TensorFlow, 예: TF 2.5** + 구 `gym` + `mujoco_py`)과 시뮬 스택
(`gymnasium` + 새 `mujoco` + SB3)은 **의존성 핀이 충돌**합니다. **한 venv에 같이 두지 마세요.**
세 환경은 **npz 파일 + ZeroMQ 메시지**로만 통신하므로 서로의 의존성을 알 필요가 없습니다.

| venv | 머신 | 설치 | 실행 |
|---|---|---|---|
| **actor** | 로봇 노트북 | `requirements-actor.txt` (numpy·opencv·pyzmq/msgpack/lz4·pyrealsense2·xArm SDK) — **TF/gym/mujoco 없음** | `actor_node.py`, 실 수집기, `deploy_real.py` |
| **learner** | GPU 서버 | `requirements-learner.txt` (**기존 DIFF-IL TF env**, 예: TF 2.5, + pyzmq/msgpack/lz4) — **gymnasium 없음** | `learner_node.py`(real) |
| **sim** | 서버/워크스테이션 | `requirements-sim.txt` (gymnasium·SB3·mujoco) | `train.py`, `collect_sim_demos.py`, eval, render |

> **핵심:** learner는 기본값이 **gymnasium-free**(`build_diffil(use_source_env=False)`)라서
> TF 2.5 venv가 gymnasium/mujoco를 전혀 import하지 않습니다 → 버전 충돌 없음. 학습은 오직
> 데이터셋(npz) + 스트리밍 trajectory로만 진행됩니다. (소스 롤아웃 wandb GIF가 필요할 때만
> `--use-source-env`를 켜면 되고, 그때만 그 venv에 gymnasium+mujoco가 필요합니다.)

- 헤드리스 렌더링(sim venv 한정): `MUJOCO_GL=egl` (또는 `osmesa`)
- 네트워크: 노트북이 서버의 **5557**(데이터 업)·**5558**(정책 다운) 포트에 접근 가능해야 함

설치 예시:
```bash
# 노트북
python -m venv venv-actor && source venv-actor/bin/activate
pip install -r requirements-actor.txt

# 서버(학습) — 기존 DIFF-IL venv에 전송 라이브러리만 추가
pip install -r requirements-learner.txt   # 사실상 pyzmq msgpack lz4 만

# 시뮬/데이터 준비
python -m venv venv-sim && source venv-sim/bin/activate
pip install -r requirements-sim.txt
```

---

## 5. 전체 워크플로우

### Step A — 시뮬 Reach 전문가 학습 (sim venv)

```bash
python scripts/train.py --task reach --algo ppo --domain_rand
#  -> outputs/... 에 모델 저장 (레포에 예시로 outputs/reach_ppo_dr/final_model.zip 포함)
python scripts/eval_headless.py --task reach --algo ppo --model outputs/reach_ppo_dr/final_model.zip
```

> 현재 reach 설정: 에피소드 **200**스텝, **고정 goal `[0.48, -0.30, 0.42]`**(front 카메라에
> 또렷이 보이며 200스텝 내 도달 가능한 가장 먼 점), action_scale 0.05, 50Hz, DR(질량/마찰/PD/
> 노이즈/지연) 옵션. 카메라는 `front`(실 RealSense 근사)·`ob_b/c/d`·`topdown` 선택 가능.

### Step B — 4개 데이터셋 수집 (sim venv + 노트북)

```bash
# B^SR — 소스 랜덤(sim). 연구할 카메라 선택(front/ob_b/ob_c/ob_d)
python scripts/diffil/collect_sim_demos.py --mode random \
    --render-camera front --num-episodes 50 \
    --name XArm6Reach_random --out-dir prior_data

# B^SE — 소스 전문가(sim), 학습된 정책 롤아웃
python scripts/diffil/collect_sim_demos.py --mode policy --algo ppo \
    --model outputs/reach_ppo_dr/final_model.zip --render-camera front \
    --num-episodes 50 --name XArm6Reach --out-dir expert_data

# B^TR — 타깃 랜덤(실로봇). 먼저 --dry-run, 그다음 실제
python scripts/run_real_reach_collect.py --ip 192.168.1.199 \
    --front-serial <REALSENSE_SERIAL> --num-episodes 30 --max-steps 200
#   -> data/real_reach/xarm6_real_reach_dataset.npz
#   로더가 찾는 위치로 배치/이름변경: 예) prior_data/XArm6Reach_real_random/
```

각 수집기는 끝에 **conform 검사**(DIFF-IL 버퍼 포맷 검증)를 출력합니다. 수동 재검사:
```bash
python scripts/diffil/dataset_conform.py --npz prior_data/XArm6Reach_random/XArm6Reach_random.npz
```

> **데이터셋 역할**: `B^SE`=모방 대상(sim 전문가), `B^SR`=sim 랜덤(전문성 label용),
> `B^TR`=real 랜덤(타깃 prior + B^TL seed), `B^TL`=온라인 real 롤아웃(학습 중 채워짐).

### Step C — (선택) 카메라 시점 선택

sim env는 여러 카메라(`front`≈실 RealSense, 더 위에서 보는 `ob_b/c/d`, `topdown`)를 제공합니다.
`--render-camera` 또는 per-camera env id로 선택하세요. **sim 수집 카메라**를 실제 RealSense
장착 위치와 맞추는 것이 sim↔real 시각 정렬의 핵심 노브입니다.

### Step D — DIFF-IL 루프 실행

```bash
# 1) 서버 (learner venv: TF + DIFF-IL repo + xarm6 repo를 PYTHONPATH에)
cd scripts/diffil
python learner_node.py \
    --env-name XArm6Reach \
    --source-random XArm6Reach_random \
    --target-random XArm6Reach_real_random \
    --target-seed   XArm6Reach_real_random \
    --traj-port 5557 --weight-port 5558
#   (gymnasium 불필요. 소스 GIF가 필요하면 --use-source-env 추가)

# 2) 노트북 (actor venv: TF 없음) — 실 팔에서 정책 실행, trajectory 스트리밍
cd scripts/diffil
python actor_node.py \
    --ip 192.168.1.199 --front-serial <REALSENSE_SERIAL> \
    --server-host <SERVER_IP> --num-episodes 0 \
    --control-hz 50 --action-filter 0.3 --explore-noise 0.1
```

서버는 `B^TL`을 real-랜덤으로 seed하고 `B^SE/B^SR/B^TR`을 로드해 학습하며 매 라운드 새 actor를
publish합니다. 노트북은 그걸 hot-swap하며 계속 수집합니다. **안전(safe-zone·fault·action 필터)은
노트북 로컬**에서 강제되므로 네트워크가 끊겨도 팔은 안전합니다.

### Step E — 단일 정책 배포/평가 (노트북)

온라인 학습과 별개로, 학습된 **state 정책**을 실 팔에서 실행:
```bash
python scripts/deploy_real.py --task reach --model <policy.zip> --dry-run         # 먼저 확인
python scripts/deploy_real.py --task reach --model <policy.zip> --speed 30 --action-filter 0.3
```

---

## 6. 레포 구조 / 파일 맵

| 경로 | 설명 |
|---|---|
| `xarm_rl/envs/reach_env.py` | Reach 태스크(state 21D, 고정 goal, 에피소드 200, 선택적 action-rate 페널티) |
| `xarm_rl/envs/base_env.py`, `assets/scene_reach.xml` | MuJoCo 모델 + 카메라(`front`,`ob_b/c/d`,`topdown`) |
| `xarm_rl/__init__.py` | env id: `XArm6Reach-v0`, per-camera `XArm6Reach-front/obB/obC/obD/topdown-v0` |
| `scripts/train.py` / `eval_headless.py` | sim 전문가 학습 / 평가 |
| `scripts/deploy_real.py` | 실 팔에 state 정책 배포(safe-zone·fault·action 필터) |
| `scripts/run_real_reach_collect.py` + `real_reach_collector.py` | 실 랜덤 데이터 `B^TR` 수집(스레드 카메라·안전) |
| `scripts/diffil/collect_sim_demos.py` | sim `B^SE`/`B^SR` 수집 |
| `scripts/diffil/learner_node.py` | **서버**: DIFF-IL 학습 + actor publish (`--mock`로 TF 없이 e2e) |
| `scripts/diffil/actor_node.py` | **노트북**: 실 팔에서 actor 실행 + 스트리밍 |
| `scripts/diffil/build_diffil.py` | DIFF-IL 그래프/버퍼 구성(gymnasium-free, DIFF-IL repo 드롭인) |
| `scripts/diffil/{comm,weight_io,policy_runtime,real_diffil_env,dataset_conform}.py` | 전송 / 가중치 IO / numpy actor / 실 env / 데이터 검증 |
| `xarm_rl/envs/diffil_adapter.py` | sim env를 DIFF-IL Sampler용으로 래핑(`--use-source-env`일 때만) |
| `requirements-actor/learner/sim.txt` | venv별 의존성 |

---

## 7. 핵심 개념 요약

- **state 정책 / image 보상.** 정책 in=21D state, out=6D joint delta. 이미지(64×64 4프레임)는
  서버에서 보상 `R̂ = -log(1 - F_s·F_f)` 계산에만.
- **2개 label.** `F_f`=frame 시간(goal 근처일수록 ↑ → 진행 보상), `F_s`=sequence 전문성.
- **고정 goal** `[0.48, -0.30, 0.42]` m — front 카메라에 또렷이 보이고 200스텝 내 도달 가능한
  가장 먼 점.
- **카메라 선택 가능** — 정책은 카메라를 보지 않으므로(state 기반), 재학습 없이 IL 이미지 채널만
  시점을 바꿔 실험할 수 있음.
- **안전은 노트북 로컬** — TCP safe-zone hard guard, 컨트롤러 fault 정지, action EMA 필터(급반전
  댐핑 → overspeed/충돌 보호 트립 방지).
- **3-venv 분리** — actor(numpy)·learner(TF)·sim(gymnasium)이 npz+ZeroMQ로만 통신.

---

## 8. 안전 체크리스트 (실로봇)

1. 항상 `--dry-run`(mock 팔+더미 카메라)으로 먼저 흐름·shape 검증.
2. e-stop을 손에 들고, `deploy_real.py`는 `--speed 30`으로 시작.
3. **safe zone**(`x 0–0.57, y -0.54–0.55, z 0.18–0.60` m)이 xArm Studio 설정과 일치하고,
   좌우 작업공간(±0.33 m)에 장애물이 없는지 확인.
4. 온라인 학습 중에는 `--action-filter > 0` 유지(탐험 정책은 jerky함).
5. 제어 루프는 규칙적이어야 함 — RealSense는 **백그라운드 스레드**에서 돌아 50Hz 루프를
   막지 않음.

---

## 9. 트러블슈팅

| 증상 | 해결 |
|---|---|
| actor가 정책을 못 바꿈 | 서버 `--weight-port` 도달 확인. PUB/SUB은 CONFLATE(최신만). |
| learner가 데이터 못 받음 | `--traj-port`/방화벽 확인. actor 로그에 `sent=True` 확인. |
| `conform check ... FAIL` | `ims`가 `[N,4,64,64,3] uint8`+`ids`인지 확인. 수집기 재실행. |
| sim 수집이 검은 화면 | `MUJOCO_GL=egl`(또는 `osmesa`) 설정. 헤드리스 서버는 GL 백엔드 필요. |
| 에피소드 중 팔이 멈춤 | 컨트롤러 fault — env가 clean 후 에피소드 종료. 부하/충돌 민감도 점검. |
| `build_diffil` import 에러 | 서버 PYTHONPATH에 **DIFF-IL repo**(+`--use-source-env`면 xarm6 repo) 추가. |
| TF 2.5 ↔ gymnasium 충돌 | learner는 기본 gymnasium-free. 한 venv에 둘을 같이 깔지 말 것(§4). |

---

### 한 줄 멘탈 모델
> 시뮬에서 전문가를 학습하고 몇 개의 고정 데이터셋을 모은 뒤, **실제 팔**이 작은 state 정책을
> 온라인으로 돌리는 동안 **GPU 서버**가 *이미지*로 모방 점수를 매겨 더 나은 정책을 계속 보내준다 —
> 전부 ZeroMQ로, 안전은 로봇에 고정된 채로.
=======
# DIFFIL-xarm6
test- sim-2-real applications for DIFFIL algorithm
>>>>>>> fcc9d72c8b7c60788b689b07117e46d865521b77
