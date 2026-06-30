# xArm6 Sim2Real + DIFF-IL — Full Pipeline Guide

This guide explains, from scratch, how to run everything we built: the **xArm6 Reach
simulation**, the **data collectors** (sim + real), the **state-based deploy**, and the
**DIFF-IL cross-domain imitation** loop that trains online across a **GPU server** and a
**robot laptop**.

If you only want to confirm the moving parts work, jump to **[2. Quickstart](#2-quickstart-5-minute-dry-run-no-hardware-no-gpu)**.

---

## 1. What this is

We train a robot to do a **Reach** task (move the end-effector to a fixed goal on a table)
and study **cross-domain imitation (DIFF-IL)** where the *expert* lives in **simulation
(source)** and the *learner* runs on the **real xArm6 (target)**.

Two important design facts that shape everything:

- **The control policy is state-based** (21-dim observation → 6-dim joint-delta action).
  It is small and runs on the laptop **without TensorFlow** at 50 Hz.
- **Images are only used to compute the imitation reward** (DIFF-IL encoder + labels +
  discriminators), which happens **on the GPU server**. The camera bridges the sim↔real
  *visual* gap; the policy input itself has no domain gap.

```
            SOURCE = simulation                         TARGET = real xArm6
   ┌───────────────────────────────┐          ┌────────────────────────────────┐
   │ XArm6Reach-v0 (MuJoCo)         │          │ RealRobotEnv (XArmAPI+RealSense)│
   │  expert demos  B^SE            │          │  random demos  B^TR             │
   │  random demos  B^SR            │          │  online rollouts B^TL  ◀────────┼─ π^TL (numpy actor)
   └───────────────┬───────────────┘          └───────────────┬────────────────┘
                   │ fixed datasets (offline)                  │ trajectories (ZeroMQ, online)
                   ▼                                           ▼
         ┌─────────────────────────────────────────────────────────────┐
         │  GPU SERVER  —  learner_node.py (DIFF-IL / DisentanGAIL, TF)  │
         │  encoder p · decoders q^S/q^T · F_f · F_s · D_f · D_s · SAC   │
         │  reward R̂ = -log(1 - F_s·F_f)   →   trains the SAC policy     │
         └──────────────────────────────┬──────────────────────────────┘
                                        │ actor weights (ZeroMQ PUB, periodic)
                                        ▼  hot-swap on the laptop
```

---

## 2. Quickstart (5-minute dry-run, no hardware, no GPU)

This exercises the **whole distributed loop** with a mock learner (no TensorFlow) and a
mock robot (no hardware). Two terminals, same machine:

```bash
pip install pyzmq msgpack lz4 numpy            # only deps needed for the dry-run

# terminal A — fake learner (echoes data, publishes a fake actor)
cd scripts/diffil
python learner_node.py --mock --publish-every 1

# terminal B — fake actor on a mock arm, sends 3 episodes
cd scripts/diffil
python actor_node.py --dry-run --server-host 127.0.0.1 --num-episodes 3 --max-steps 20
```

You should see the actor send trajectories, the learner receive them
(`ims (N,4,64,64,3)`), and the actor **hot-swap** to `policy v1`, `v2`, … That confirms the
transport + actor + weight-sync path end to end.

---

## 3. Repository map (the parts you'll touch)

| Path | What it is |
|---|---|
| `xarm_rl/envs/reach_env.py` | The Reach task (state obs 21-D, fixed goal, episode 200, optional action-rate penalty). |
| `xarm_rl/envs/base_env.py`, `assets/scene_reach.xml` | MuJoCo model + cameras (`front`, `ob_b/c/d`, `topdown`). |
| `xarm_rl/__init__.py` | Env ids: `XArm6Reach-v0` and per-camera `XArm6Reach-front/obB/obC/obD/topdown-v0`. |
| `scripts/train.py` | Train the sim reach **expert** (PPO/SAC, optional domain randomization). |
| `scripts/eval_headless.py` | Evaluate a trained sim policy (success rate). |
| `scripts/deploy_real.py` | Run a trained **state policy** on the real xArm6 (safe-zone guard, fault stop, action filter). |
| `scripts/run_real_reach_collect.py` + `real_reach_collector.py` | Collect **real** random data `B^TR` (threaded camera, safety). |
| `scripts/diffil/collect_sim_demos.py` | Collect **sim** datasets `B^SE` (policy) / `B^SR` (random). |
| `scripts/diffil/learner_node.py` | **Server**: DIFF-IL training + publishes the actor. |
| `scripts/diffil/actor_node.py` | **Laptop**: runs the actor on the real robot, streams data. |
| `scripts/diffil/build_diffil.py` | Builds the DIFF-IL graph/buffers (drop-in for the DIFF-IL repo). |
| `scripts/diffil/{comm,weight_io,policy_runtime,real_diffil_env,dataset_conform}.py` | transport / weight IO / numpy actor / real env / dataset checker. |
| `scripts/diffil/README.md` | Short, code-level notes for the diffil package. |

---

## 4. Prerequisites — three isolated environments

The DIFF-IL training stack (TensorFlow, e.g. **TF 2.5**, with old `gym` + `mujoco_py`) and
the sim stack (`gymnasium` + new `mujoco` + SB3) have **conflicting pins** and must NOT
share a venv. We keep them apart; they only exchange **npz files + ZeroMQ messages**.

| venv | machine | installs | runs |
|---|---|---|---|
| **actor** | robot laptop | `requirements-actor.txt` (numpy, opencv, pyzmq/msgpack/lz4, pyrealsense2, xArm SDK) — **no TF/gym/mujoco** | `actor_node.py`, real collectors, `deploy_real.py` |
| **learner** | GPU server | `requirements-learner.txt` (your **DIFF-IL TF env**, e.g. TF 2.5, + pyzmq/msgpack/lz4) — **gymnasium-free** | `learner_node.py` (real) |
| **sim** | server/workstation | `requirements-sim.txt` (gymnasium + SB3 + mujoco) | `train.py`, `collect_sim_demos.py`, eval, render |

Key point: the **learner is gymnasium-free by default** (`build_diffil(use_source_env=False)`),
so the TF 2.5 venv never imports gymnasium/mujoco — no version conflict. It trains purely
from datasets (npz) + streamed trajectories. (Set `--use-source-env` only if you want the
optional wandb source-rollout GIFs, which then needs gymnasium+mujoco in that venv.)

- Headless rendering (sim venv only): `MUJOCO_GL=egl` (or `osmesa`).
- Network: the laptop must reach the server on **5557** (data-up) and **5558** (policy-down).

---

## 5. Full workflow

### Step A — Train the sim Reach expert (server)

```bash
python scripts/train.py --task reach --algo ppo --domain_rand
# -> saves a model under outputs/...   (the repo already ships e.g. outputs/reach_ppo_dr/final_model.zip)
python scripts/eval_headless.py --task reach --algo ppo --model outputs/reach_ppo_dr/final_model.zip
```

### Step B — Collect the four datasets

DIFF-IL uses four datasets. Three are fixed (collected once); one is online.

```bash
# B^SR — source random (sim).   Pick the camera you want to study (front/ob_b/ob_c/ob_d).
python scripts/diffil/collect_sim_demos.py --mode random \
    --render-camera front --num-episodes 50 \
    --name XArm6Reach_random --out-dir prior_data

# B^SE — source expert (sim), rolled out from the trained policy.
python scripts/diffil/collect_sim_demos.py --mode policy --algo ppo \
    --model outputs/reach_ppo_dr/final_model.zip --render-camera front \
    --num-episodes 50 --name XArm6Reach --out-dir expert_data

# B^TR — target random (REAL robot).  Start with --dry-run, then real.
python scripts/run_real_reach_collect.py --ip 192.168.1.199 \
    --front-serial <REALSENSE_SERIAL> --num-episodes 30 --max-steps 200
#   -> data/real_reach/xarm6_real_reach_dataset.npz
#   place/rename it where the loader expects it, e.g. prior_data/XArm6Reach_real_random/
```

Every collector prints a **conform check** (validates the npz for the DIFF-IL buffers).
You can re-check any file manually:

```bash
python scripts/diffil/dataset_conform.py --npz prior_data/XArm6Reach_random/XArm6Reach_random.npz
```

> **Dataset roles**: `B^SE` = what to imitate (sim expert), `B^SR` = sim random (for the
> expertise label), `B^TR` = real random (target prior + B^TL seed), `B^TL` = online real
> rollouts (filled during training).

### Step C — (optional) choose the camera viewpoint

The sim env exposes several cameras (`front` ≈ real RealSense, plus higher obliques and
top-down). Use the per-camera env ids or `--render-camera`. Keep the **sim collection
camera** consistent with where you physically mount the real RealSense; this is the main
sim↔real *visual* alignment knob.

### Step D — Run the DIFF-IL loop

```bash
# 1) SERVER (needs TF + DIFF-IL repo on PYTHONPATH, plus this repo for xarm_rl)
cd scripts/diffil
python learner_node.py \
    --env-name XArm6Reach \
    --source-random XArm6Reach_random \
    --target-random XArm6Reach_real_random \
    --target-seed   XArm6Reach_real_random \
    --traj-port 5557 --weight-port 5558

# 2) LAPTOP (no TF) — runs the policy on the real arm forever, streams trajectories
cd scripts/diffil
python actor_node.py \
    --ip 192.168.1.199 --front-serial <REALSENSE_SERIAL> \
    --server-host <SERVER_IP> --num-episodes 0 \
    --control-hz 50 --action-filter 0.3 --explore-noise 0.1
```

The server seeds `B^TL` with the real-random data, loads `B^SE/B^SR/B^TR`, trains, and
publishes a new actor each round; the laptop hot-swaps it and keeps collecting. Safety
(safe-zone, fault stop, action filtering) is enforced **locally on the laptop**, so a
network hiccup never makes the arm unsafe.

### Step E — Deploy / evaluate a single policy on the real arm

Independent of online training, you can run any trained **state** policy on the real arm:

```bash
python scripts/deploy_real.py --task reach --model <policy.zip> --dry-run      # check first
python scripts/deploy_real.py --task reach --model <policy.zip> --speed 30 --action-filter 0.3
```

---

## 6. Key concepts (cheat-sheet)

- **State policy, image reward.** Policy in = 21-D state `[q, q̇, ee_xyz, goal_xyz, goal-ee]`,
  out = 6-D joint deltas. Images (`get_ims`, 4-frame 64×64 stack) feed the DIFF-IL reward
  `R̂ = -log(1 - F_s·F_f)` on the server only.
- **Two labels.** `F_f` = frame **time** label (higher near the goal → rewards progress),
  `F_s` = sequence **expertise** label (expert vs random).
- **Fixed goal** `[0.48, -0.30, 0.42]` m — the farthest point that is clearly visible in the
  front camera **and** reachable within the 200-step episode.
- **Cameras** are selectable; the policy never sees them (state-based), so you can re-render
  the IL image channel under different viewpoints without retraining.
- **Safety is local** to the laptop: TCP safe-zone hard guard, controller fault stop, and an
  EMA action filter (damps abrupt reversals that can trip overspeed/collision protection).

---

## 7. Safety checklist (real robot)

1. Always `--dry-run` first (mock arm + dummy camera) to validate shapes and flow.
2. Keep the e-stop in hand; start `deploy_real.py` at `--speed 30`.
3. Confirm the **safe zone** (`x 0–0.57, y -0.54–0.55, z 0.18–0.60` m) matches your xArm
   Studio settings and that the lateral workspace (±0.33 m) is clear of obstacles.
4. Keep `--action-filter` > 0 during online learning (the exploring policy is jittery).
5. The control loop must stay regular — the RealSense runs on a **background thread** so it
   never blocks the 50 Hz loop.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Actor never swaps policy | Check server `--weight-port` reachable; PUB/SUB uses CONFLATE (newest only). |
| Learner gets no data | Check `--traj-port`/firewall; actor `sent=True` in its log. |
| `conform check ... FAIL` | Ensure `ims` is `[N,4,64,64,3] uint8` with `ids`; re-run the collector. |
| Sim collector renders black | Set `MUJOCO_GL=egl` (or `osmesa`); headless servers need a GL backend. |
| Arm freezes mid-episode | Controller fault — the env clears it and ends the episode; check load/collision sensitivity. |
| `build_diffil` import errors | Put the DIFF-IL repo **and** this repo on the server `PYTHONPATH`. |

---

## 9. One-line mental model

> Train an expert in sim, collect a few fixed datasets, then let the **real arm** run a tiny
> state policy online while a **GPU server** watches the *images* to score imitation and
> keeps shipping a better policy back — all over ZeroMQ, with safety pinned to the robot.
