# DIFF-IL sim2real glue for xArm6

Distributed **cross-domain imitation** (DIFF-IL / DisentanGAIL) wiring that connects
the **robot laptop** (target = real xArm6) to a **GPU server** (DIFF-IL training).
Nothing here overwrites existing files — the real env reuses `real_reach_collector`,
the sim env wraps the existing gymnasium `XArm6ReachEnv`.

## Key design facts (from the paper + provided code)

- **Policy is state-based** (21-dim obs → 6-dim action). Images (`get_ims`, 4-frame
  stack) are used **only** by the encoder / labels / discriminators to compute the
  reward `R̂_t = -log(1 - F_s · F_f)`, recomputed **server-side** at train time.
  → the laptop only needs the small actor; no TF, no GAN on the laptop.
- **Two labels**: `F_f` frame *time* label (source-only, higher near the goal →
  rewards task progress), `F_s` sequence *expertise* label (expert vs random).
- **Four datasets**: `B^SE` (sim expert, fixed), `B^SR` (sim random, fixed),
  `B^TR` (real random, fixed ← our collector), `B^TL` (real learner, **online**).
- Transport: **ZeroMQ**. Actor runs the policy **without TensorFlow** (numpy forward).

## Files

| file | runs on | role |
|---|---|---|
| `comm.py` | both | ZeroMQ: trajectories PUSH/PULL, weights PUB/SUB (msgpack+lz4, version tags) |
| `weight_io.py` | server | export TF SAC actor → portable npz; load (numpy) |
| `policy_runtime.py` | laptop | TF-free numpy actor (`tanh(mean)`), hot-swap |
| `real_diffil_env.py` | laptop | `RealRobotEnv` (old-gym + `get_ims`), reuses `real_reach_collector` + safety |
| `../xarm_rl/envs/diffil_adapter.py` | server | `SimDiffilEnv` wraps gymnasium reach env for the Sampler (source / eval) |
| `actor_node.py` | laptop | online actor: env + numpy policy + sender + weight poller |
| `learner_node.py` | server | receiver → `agent_buffer` → `gail.train` → publish actor (`--mock` for TF-free e2e) |
| `dataset_conform.py` | server | validate/convert d3il npz for `DemonstrationsReplayBuffer` |

## Data flow

```
offline (once):  sim expert → B^SE,  sim random → B^SR,  real random → B^TR
online:
  [laptop]  π^TL @50Hz on real xArm6  --(obs,act,ims)--ZMQ PUSH-->  [server]
  [server]  agent_buffer.add(B^TL) → DisentanGAIL.train(enc/dec/F_f/F_s/D_f/D_s + SAC)
            --export actor-- ZMQ PUB --> [laptop hot-swap]
```

## Run

End-to-end **dry-run** (no hardware, no TF) — two terminals:
```
python scripts/diffil/learner_node.py --mock --publish-every 1
python scripts/diffil/actor_node.py --dry-run --server-host 127.0.0.1 --num-episodes 3 --max-steps 20
```

Real:
```
# server
python scripts/diffil/learner_node.py            # real mode (needs TF + DIFF-IL)
# laptop
python scripts/diffil/actor_node.py --ip 192.168.1.199 --front-serial <SERIAL> \
    --server-host <SERVER_IP> --init-weights actor_v0.npz --num-episodes 0
```

## The one seam left to wire (server, real mode)

`learner_node.RealLearner.build()` should construct the DIFF-IL graph + buffers
exactly like `run_experiment_cycle.run_experiment()`. The cleanest way: factor the
model/buffer construction in `run_experiment_cycle` into a `build_diffil(args)`
returning `(gail, agent_buffer, l_agent)`, then import it in `build()`. Everything
else (target feed from the actor, actor export/publish) is already wired and the
SAC actor architecture matches `weight_io.export_actor`.
```
```
