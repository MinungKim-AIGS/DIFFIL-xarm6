#!/usr/bin/env python3
"""Offline smoke test for the D3IL learner — NO actor, NO network, NO robot.

Builds the D3IL graph from the placed datasets, runs a few Phase-1 pretrain steps,
seeds a synthetic O^TL into the agent buffer, runs one Phase-2 policy step, and
exports the SAC actor. Exercises the same seams as check_build.py but for D3IL:

    import -> dataset load -> build -> pretrain step -> policy step -> export.

Run from the D3IL repo folder (where custom_code/run_imitation/d3il.py, sac_models,
buffers, utils + the copied build_d3il.py / weight_io.py live):

    python check_d3il_build.py
"""
from __future__ import annotations

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="D3IL learner offline smoke test")
    ap.add_argument("--file-location", default="expert_data")
    ap.add_argument("--prior-file-location", default="prior_data")
    ap.add_argument("--se-name", default="XArm6Reach")
    ap.add_argument("--sn-name", default="XArm6Reach_random")
    ap.add_argument("--tn-name", default="XArm6Reach_real_random")
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--obs-dim", type=int, default=21)
    ap.add_argument("--action-dim", type=int, default=6)
    args = ap.parse_args()

    print("=" * 60)
    print("D3IL OFFLINE SMOKE TEST")
    print("=" * 60)

    # ---- stage 1: imports ----
    try:
        import tensorflow as tf
        from build_d3il import (D3ilConfig, build_d3il, pretrain_image_translation,
                                policy_train_step)
        import weight_io
    except Exception as e:
        print(f"[stage1 FAIL] import: {type(e).__name__}: {e}")
        print("  -> run from the D3IL repo folder with TF + build_d3il.py/weight_io.py copied in.")
        raise
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    print(f"[stage1 OK ] imports fine (TF {tf.__version__})")

    cfg = D3ilConfig(file_location=args.file_location, prior_file_location=args.prior_file_location,
                     se_name=args.se_name, sn_name=args.sn_name, tn_name=args.tn_name,
                     obs_dim=args.obs_dim, action_dim=args.action_dim,
                     n_demos=2000, l_buffer_size=5000)

    # ---- stage 2: build graph + load datasets ----
    print("[stage2] building D3IL graph + loading B^SE/B^SN/B^TN ...")
    try:
        model, agent_buffer, l_agent, se_b, sn_b, tn_b = build_d3il(cfg)
    except FileNotFoundError as e:
        print(f"[stage2 FAIL] dataset not found: {e}")
        print("  -> ensure expert_*.npy exist in expert_data/<se>, prior_data/<sn>, prior_data/<tn>.")
        raise
    except Exception as e:
        print(f"[stage2 FAIL] {type(e).__name__}: {e}")
        raise
    print("[stage2 OK ] graph built, datasets loaded")

    # ---- stage 3: a few Phase-1 pretrain steps ----
    print(f"[stage3] {args.pretrain_epochs} feature-model pretrain step(s) ...")
    try:
        pretrain_image_translation(model, se_b, sn_b, tn_b, cfg, args.pretrain_epochs, log_interval=1)
    except Exception as e:
        print(f"[stage3 FAIL] pretrain: {type(e).__name__}: {e}")
        raise
    print("[stage3 OK ] pretrain step ran")

    # ---- stage 4: seed a synthetic O^TL, run one Phase-2 policy step ----
    print("[stage4] seeding synthetic O^TL + one policy train step ...")
    try:
        T = 256
        side = cfg.im_side
        otl = {"obs": np.random.randn(T, cfg.obs_dim).astype(np.float32),
               "nobs": np.random.randn(T, cfg.obs_dim).astype(np.float32),
               "act": np.clip(np.random.randn(T, cfg.action_dim), -1, 1).astype(np.float32),
               "rew": np.zeros(T, np.float32),
               "don": np.zeros(T, bool),
               "ims": np.random.randint(0, 255, (T, cfg.past_frames, side, side, 3), np.uint8),
               "n": T}
        agent_buffer.add(otl)
        policy_train_step(model, se_b, sn_b, tn_b, agent_buffer, cfg, n_new=T)
    except Exception as e:
        print(f"[stage4 FAIL] policy step: {type(e).__name__}: {e}")
        raise
    print("[stage4 OK ] policy train step ran")

    # ---- stage 5: export actor ----
    try:
        w = weight_io.export_actor(l_agent._act, version=0)
        print(f"[stage5 OK ] actor exported: act_dim={w['act_dim']}, layers={len(w['layers'])}")
    except Exception as e:
        print(f"[stage5 FAIL] export: {type(e).__name__}: {e}")
        raise

    print("=" * 60)
    print("ALL OK — D3IL build -> pretrain -> policy step -> export ran end to end.")
    print("=" * 60)


if __name__ == "__main__":
    main()
