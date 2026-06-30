#!/usr/bin/env python3
"""Split our single bundled .npz dataset into the per-field .npy files that the
DIFF-IL loader (utils.load_expert_trajectories / load_learner_trajectories) expects.

The reference layout looks like:
    <dir>/expert_obs.npy   <dir>/expert_nobs.npy   <dir>/expert_ims.npy
    <dir>/expert_ids.npy   <dir>/expert_don.npy    <dir>/expert_acs.npy

Note the field name mapping  act -> acs  (the loader calls actions "acs"); rew/step
are NOT written (rewards are recomputed inside the buffers at sample time).

Usage — convert in place (writes the .npy files next to the .npz):
    python npz_to_npy.py --npz prior_data/XArm6Reach_real_random/xarm6_real_reach_dataset.npz

Convert and write to a specific folder / prefix:
    python npz_to_npy.py --npz <file>.npz --out-dir expert_data/XArm6Reach --prefix expert

Tip: B^SE / B^SR / B^TR are loaded with load_expert_trajectories (prefix usually
"expert"). B^TL seed is loaded with load_learner_trajectories — if that one uses a
different prefix in your utils.py (e.g. "learner"), just run this again with
--prefix learner into the same seed folder.
"""
from __future__ import annotations

import os
import argparse
import numpy as np

# our npz key  ->  output field name (the loader's naming).
# Full set the reference saves (8 files). Buffers actually USE:
#   B^SE/B^SR/B^TR (DemonstrationsReplayBuffer): ims, ids, acs, don
#   B^TL seed (LearnerAgentReplayBuffer):        obs, nobs, acs, rew, don, ids, ims, n
# We write all 8 anyway, because load_expert/learner_trajectories np.load each
# prefix file and will error if one is missing. ('step' is unused -> not written.)
FIELD_MAP = {
    "obs":  "obs",
    "nobs": "nobs",
    "ims":  "ims",
    "ids":  "ids",
    "don":  "don",
    "rew":  "rew",
    "n":    "n",
    "act":  "acs",     # <-- renamed (disk 'acs' -> memory 'act')
}


def main():
    ap = argparse.ArgumentParser(description="Split bundled .npz into per-field .npy for DIFF-IL")
    ap.add_argument("--npz", required=True, help="bundled dataset (obs,nobs,act,rew,don,ims,ids,...)")
    ap.add_argument("--out-dir", default=None, help="output folder (default: same dir as the .npz)")
    ap.add_argument("--prefix", default="expert", help='filename prefix (default "expert")')
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(f"not found: {args.npz}")
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(args.npz)
    print(f"[*] loaded {args.npz}  keys={list(z.files)}")
    print(f"[*] writing -> {out_dir}  (prefix '{args.prefix}')")

    missing = []
    for src_key, field in FIELD_MAP.items():
        if src_key not in z.files:
            missing.append(src_key)
            continue
        arr = z[src_key]
        path = os.path.join(out_dir, f"{args.prefix}_{field}.npy")
        np.save(path, arr)
        print(f"    {args.prefix}_{field}.npy   <- '{src_key}'   {arr.shape} {arr.dtype}")

    if missing:
        print(f"[!] WARNING: keys not in npz, skipped: {missing}")
    print("[done]")


if __name__ == "__main__":
    main()
