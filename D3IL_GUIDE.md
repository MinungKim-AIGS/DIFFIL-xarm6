# D3IL on xArm6 — Distributed Training + Evaluation

D3IL reuses almost all of the DIFFIL plumbing. The only new pieces are the D3IL
learner construction and an evaluation node. The control policy is the same
state-based SAC actor, so `weight_io` / `policy_runtime` (NumpyActor) / the laptop
`actor_node` are reused unchanged.

## How D3IL differs from DIFFIL

- **Two phases.** Phase 1 pretrains the image-translation / feature model **offline**
  on the fixed datasets (B^SE/B^SN/B^TN) — no robot. Phase 2 freezes that feature
  model and trains the SAC policy **online** with an IRL reward, using target-domain
  rollouts O^TL streamed from the laptop actor (the encoder stays frozen; only the
  expert discriminator + SAC update).
- **Datasets** map 1:1 to ours: B^SE = sim expert, B^SN = sim random (our B^SR),
  B^TN = real random (our B^TR). Convert with `npz_to_npy.py` exactly as for DIFFIL.
- The real robot is in the loop **only for the online policy phase and for eval** —
  the heavy feature model is trained offline first.

## New files (copy into your D3IL repo, next to custom_code/.../d3il.py)

| File | Role |
|---|---|
| `build_d3il.py` | `D3ilConfig` + `build_d3il(cfg)` — constructs D3ILModelwithPolicy + SAC + se/sn/tn buffers + CustomReplayBuffer (gymnasium-free). Plus `pretrain_image_translation()` / `policy_train_step()` helpers. |
| `d3il_learner_node.py` | Server learner: Phase-1 pretrain → publish v0 → Phase-2 online loop (feed O^TL → train → publish). `--mock` for transport tests. |
| `d3il_eval_node.py` | Laptop: load/pull a policy → run on the real xArm6 → report reach success rate. (Works for DIFFIL policies too.) |
| `check_d3il_build.py` | Offline smoke test: build → pretrain step → synthetic O^TL → policy step → export. |

Reused unchanged: `comm.py`, `weight_io.py`, `policy_runtime.py`, `actor_node.py`,
`real_diffil_env.py`, `npz_to_npy.py`.

## Run order

```bash
# 0) copy new files into the D3IL repo + datasets in place (expert_*.npy via npz_to_npy.py)
cp /home/user/Kimiw/xarm6/scripts/diffil/{build_d3il,d3il_learner_node,d3il_eval_node,check_d3il_build}.py  <D3IL_REPO>/

# 1) offline smoke test (no robot, no network)
cd <D3IL_REPO> && python check_d3il_build.py            # expect "ALL OK"

# 2) server: real learner (pretrain then online policy phase)
python d3il_learner_node.py --pretrain-epochs 50000 --min-new-steps 200
#    quick check: --pretrain-epochs 50 --training-starts 20 --min-new-steps 20 --l-batch-size 32 --d-batch 8

# 3) laptop: actor streams real-robot O^TL (SAME node as DIFFIL)
python actor_node.py --ip 192.168.1.199 --front-serial <SERIAL> --server-host <SERVER_IP> --num-episodes 0

# 4) evaluation (laptop): pull or load a policy, measure reach success
python d3il_eval_node.py --ip 192.168.1.199 --front-serial <SERIAL> --weights actor_v50.npz \
    --episodes 20 --success-dist 0.05
#    or pull the live policy:  --server-host <SERVER_IP>
```

## Notes / things to watch

- `build_d3il.py` reproduces `run_d3il`'s factory/hyperparameter setup; it assumes the
  D3IL module signatures match the uploaded repo. Run `check_d3il_build.py` first — it
  surfaces any constructor/loader mismatch offline.
- `c_norm_de` / `c_norm_be` default to 0 (Reacher-like). Tune in `D3ilConfig` if needed.
- Phase-1 pretrain is long (tens of thousands of epochs). It's offline and one-time;
  consider saving/reloading the pretrained feature model if your repo supports it.
- Target = real robot here (a real-robot policy). If you instead want target = sim
  (robot for eval only), the expert/source mapping changes — ask before switching.
