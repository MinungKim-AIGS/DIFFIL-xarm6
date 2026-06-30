# Robot Laptop Guide — xArm6 (Actor Side)

This guide covers **only the work you run on the robot laptop**: the machine physically
connected to the xArm6 and the RealSense camera. The GPU server (DIFF-IL training) is
documented separately in `PIPELINE.md`.

Two facts shape everything the laptop does:

- The control **policy is state-based** (21-D observation → 6-D joint-delta action) and runs
  here as a **pure-numpy forward pass** — no TensorFlow, no gymnasium, no mujoco.
- The **camera images** are collected and streamed up, but they are only used **on the server**
  to compute the DIFF-IL reward. The laptop never trains.

So the laptop has exactly three jobs: **collect real random data** (`B^TR`), **run the online
actor** that rolls out the current policy and streams trajectories, and **deploy** a finished
policy for evaluation.

---

## 0. One-time setup

Create an isolated **actor** venv and install the laptop-only dependencies. This venv must
*not* contain TensorFlow / gymnasium / mujoco — those belong to the server and sim venvs.

```bash
python -m venv .venv-actor
source .venv-actor/bin/activate          # Windows: .venv-actor\Scripts\activate
pip install -r requirements-actor.txt    # numpy, opencv, pyzmq, msgpack, lz4, pyrealsense2, xArm-Python-SDK
```

Things to have ready before touching hardware:

- **xArm6 IP** (default `192.168.1.199`).
- **RealSense serial** for the front camera (`rs-enumerate-devices` lists it).
- **GPU server IP** and that ports **5557** (data up) and **5558** (policy down) are reachable.
- Physical: **e-stop within reach**, table clear, arm in a safe starting area.

---

## 1. Dry-run the loop (no hardware, no GPU)

Always start here. This runs the **entire actor path** — env, numpy actor, ZeroMQ transport,
weight hot-swap — against a mock learner, with a mock arm and a dummy camera. Two terminals
on the laptop:

```bash
# terminal A — fake learner (echoes data, publishes a fake actor); no TensorFlow needed
cd scripts/diffil
python learner_node.py --mock --publish-every 1

# terminal B — fake actor on a mock arm, sends 3 short episodes
cd scripts/diffil
python actor_node.py --dry-run --server-host 127.0.0.1 --num-episodes 3 --max-steps 20
```

You should see the actor send trajectories (`ims (N,4,64,64,3)`) and **hot-swap** to
`policy v1`, `v2`, …. That confirms the transport + actor + weight-sync work end to end before
any robot is involved.

---

## 2. Collect real random data — `B^TR`

This is the **target random** dataset DIFF-IL needs. The collector uses joint-space goal
babbling (sweeps the reachable workspace) plus a **non-terminating** safe-zone guard (if the
TCP nears the boundary it steers home and the episode keeps running, so episodes stay full
length).

```bash
# 2a. dry-run first — verify shapes/format with a mock arm + dummy camera
python scripts/run_real_reach_collect.py --dry-run --num-episodes 2 --max-steps 20

# 2b. real collection (default target: 10,000 transitions)
python scripts/run_real_reach_collect.py \
    --ip 192.168.1.199 --front-serial <REALSENSE_SERIAL> \
    --num-samples 10000 --max-steps 200 --action-scale 0.05
#   -> data/real_reach/xarm6_real_reach_dataset.npz
```

Useful flags: `--num-samples N` (collect by sample count; set `0` to use `--num-episodes`
instead), `--min-steps 50` (discard episodes cut shorter than this), `--waypoints
data/safe_waypoints.npz` (FK-verified safe pool; if missing it falls back to OU noise).

When done, hand the `.npz` to the server and place it where the loader expects (e.g.
`prior_data/XArm6Reach_real_random/`).

---

## 3. Run the online actor — `B^TL`

With the **learner running on the server**, start the actor on the laptop. It rolls out the
**current** policy on the real arm, streams each episode up, and between episodes **pulls and
hot-swaps** to the newest weights the server published.

```bash
cd scripts/diffil
python actor_node.py \
    --ip 192.168.1.199 --front-serial <REALSENSE_SERIAL> \
    --server-host <GPU_SERVER_IP> \
    --init-weights /path/to/actor_v0.npz \
    --num-episodes 0          # 0 = run forever (Ctrl-C to stop)
```

Notes:

- `--init-weights` is the starting policy. Without it the actor begins from a **random** policy
  and waits for the server's first published weights.
- `--action-filter 0.3` (default) is an EMA smoother that limits abrupt action changes — keep it
  on for hardware safety. `--explore-noise 0.1` adds exploration; lower it for cleaner rollouts.
- All safety is enforced **locally** by `RealRobotEnv` (safe-zone guard + fault stop), so the
  arm stays protected even if the network hiccups.

---

## 4. Deploy a finished policy (evaluation)

To just **run a trained state policy** on the arm (no learner, no streaming):

```bash
python scripts/deploy_real.py \
    --ip 192.168.1.199 \
    --model /path/to/policy.npz \
    --target 0.48 -0.30 0.42 \
    --action-filter 0.3
```

This applies the same EMA action filter, safe-zone guard, and fault monitoring (stops on a
controller `error_code`). Start with the arm in a clear area and the e-stop in hand.

---

## Quick reference

| Task | Command (laptop) | Output / effect |
|---|---|---|
| Setup | `pip install -r requirements-actor.txt` | actor venv ready |
| Dry-run loop | `actor_node.py --dry-run` + `learner_node.py --mock` | verify transport + hot-swap |
| Collect `B^TR` | `run_real_reach_collect.py --ip … --front-serial …` | `data/real_reach/…npz` |
| Online actor `B^TL` | `actor_node.py --ip … --server-host …` | streams episodes, hot-swaps policy |
| Deploy | `deploy_real.py --ip … --model …` | runs policy on the arm |

**Safety reminders:** dry-run before every real run · keep the e-stop in hand · start at low
speed in a clear area · the safe-zone guard and fault stop are always on, but they are a backup,
not a substitute for supervision.
