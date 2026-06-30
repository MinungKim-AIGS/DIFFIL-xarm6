#!/usr/bin/env python3
"""Offline smoke test for the DIFF-IL learner — NO actor, NO network, NO robot.

It builds the real DIFF-IL graph from whatever datasets are sitting in the
expected folders, runs a few tiny training steps, and exports the SAC actor.
That exercises the four previously-untested seams at once:

    import resolution  ->  dataset loading  ->  graph build  ->  one train step.

Run it FROM the DIFF-IL repo folder (where sac_models.py and the copied
build_diffil.py / weight_io.py live):

    cd /home/user/degail_code
    python check_build.py

Defaults assume the layout you described (the same npz placed in all three dirs):

    expert_data/XArm6Reach/xarm6_real_reach_dataset.npz                 (B^SE)
    prior_data/XArm6Reach_random/xarm6_real_reach_dataset.npz           (B^SR)
    prior_data/XArm6Reach_real_random/xarm6_real_reach_dataset.npz      (B^TR + B^TL seed)

If your folder names differ, pass them via the flags below.
"""
from __future__ import annotations

import os
import glob
import argparse
import numpy as np


def detect_episode_len(npz_path):
    """Most common per-episode length in a d3il npz (so episode_limit matches data)."""
    try:
        z = np.load(npz_path)
        if "ids" not in z.files:
            return None
        _, counts = np.unique(z["ids"], return_counts=True)
        uniq = set(int(c) for c in counts)
        if len(uniq) == 1:
            return int(counts[0])
        print(f"  [warn] variable episode lengths {sorted(uniq)} in {npz_path}; "
              f"using median")
        return int(np.median(counts))
    except Exception as e:
        print(f"  [warn] could not probe {npz_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="DIFF-IL learner offline smoke test")
    ap.add_argument("--file-location", default="expert_data")
    ap.add_argument("--prior-file-location", default="prior_data")
    ap.add_argument("--env-name", default="XArm6Reach", help="B^SE dir under file-location")
    ap.add_argument("--source-random", default="XArm6Reach_random", help="B^SR dir")
    ap.add_argument("--target-random", default="XArm6Reach_real_random", help="B^TR dir")
    ap.add_argument("--target-seed", default="XArm6Reach_real_random", help="B^TL seed dir")
    ap.add_argument("--episode-limit", type=int, default=0, help="0 = auto-detect from data")
    ap.add_argument("--train-steps", type=int, default=2)
    args = ap.parse_args()

    print("=" * 64)
    print("DIFF-IL OFFLINE SMOKE TEST")
    print("=" * 64)

    # ---- stage 0: auto-detect episode length from a placed npz ----
    probe = None
    for d in (os.path.join(args.prior_file_location, args.target_random),
              os.path.join(args.file_location, args.env_name)):
        g = glob.glob(os.path.join(d, "*.npz"))
        if g:
            probe = g[0]
            break
    epi = args.episode_limit
    if epi == 0:
        epi = (detect_episode_len(probe) if probe else None) or 200
        print(f"[stage0] episode_limit = {epi}"
              + (f"  (auto from {probe})" if probe else "  (default; no npz found to probe)"))
    else:
        print(f"[stage0] episode_limit = {epi}  (user-set)")

    # ---- stage 1: imports ----
    try:
        import tensorflow as tf
        from build_diffil import DiffilConfig, build_diffil
        import weight_io
    except Exception as e:
        print(f"[stage1 FAIL] import error: {type(e).__name__}: {e}")
        print("  -> are you running from the DIFF-IL repo folder, with TF installed and")
        print("     our build_diffil.py / weight_io.py copied next to sac_models.py?")
        raise
    # don't let TF grab the whole GPU for a smoke test
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    print(f"[stage1 OK ] imports fine (TF {tf.__version__})")

    cfg = DiffilConfig(
        episode_limit=epi, random_epi_limit=epi,
        file_location=args.file_location, prior_file_location=args.prior_file_location,
        env_name=args.env_name, source_random_location=args.source_random,
        target_random_location=args.target_random, target_learner_seed=args.target_seed,
        n_expert_demos=2000, n_prior_demos=2000, l_buffer_size=2000,
        use_source_env=False,
    )

    # ---- stage 2: build graph + load datasets ----
    print("[stage2] building graph + loading B^SE / B^SR / B^TR / B^TL ...")
    try:
        gail, agent_buffer, l_agent, sampler = build_diffil(cfg)
    except FileNotFoundError as e:
        print(f"[stage2 FAIL] dataset not found: {e}")
        print("  -> load_expert_trajectories could not find your file. It is named")
        print("     'xarm6_real_reach_dataset.npz'. Check how utils.load_expert_trajectories")
        print("     resolves the path (specific filename vs glob), and rename/adjust to match.")
        raise
    except Exception as e:
        print(f"[stage2 FAIL] {type(e).__name__}: {e}")
        print("  -> if this is a TypeError on a constructor (SAC/DisentanGAIL/buffer/loader),")
        print("     your repo's signature differs from build_diffil.py's assumptions; that is")
        print("     exactly the seam to reconcile.")
        raise
    print("[stage2 OK ] graph built, datasets loaded")

    # ---- stage 3: a few train steps ----
    print(f"[stage3] running {args.train_steps} tiny train step(s) ...")
    try:
        for i in range(args.train_steps):
            gail.train(agent_buffer=agent_buffer, l_batch_size=32, l_updates=2, l_act_delay=1,
                       d_updates=2, mi_updates=2, d_e_batch_size=8, d_l_batch_size=8,
                       sampling_alpha=0.0)
            print(f"   step {i + 1}/{args.train_steps} done")
    except Exception as e:
        print(f"[stage3 FAIL] {type(e).__name__}: {e}")
        print("  -> build/load worked; the train() call differs. Compare gail.train(...) "
              "args with your DisentanGAIL.train signature.")
        raise
    print("[stage3 OK ] training loop ran")

    # ---- stage 4: export actor (what gets published to the laptop) ----
    try:
        w = weight_io.export_actor(l_agent._act, version=0)
        print(f"[stage4 OK ] actor exported: act_dim={w['act_dim']}, layers={len(w['layers'])}")
    except Exception as e:
        print(f"[stage4 FAIL] export_actor: {type(e).__name__}: {e}")
        raise

    print("=" * 64)
    print("ALL OK  —  import -> load -> build -> train -> export ran end to end.")
    print("You can now place the real sim datasets and connect the live actor.")
    print("=" * 64)


if __name__ == "__main__":
    main()
