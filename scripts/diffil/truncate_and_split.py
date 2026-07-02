#!/usr/bin/env python3
"""Truncate every episode in a bundled DIFF-IL .npz to the first --max-step
transitions, then split into the per-field .npy files the loader expects.

Why: if the expert reaches the goal at ~75 and then HOLDS to 200, ~60% of frames
are the same static goal pose. That over-represents "sitting at the goal" and lets
the cross-domain encoder/discriminator/reward degenerate. Cutting each episode to
100 steps (~75 reach + ~25 hold) rebalances toward the reaching MOTION.

Episodes are delimited by the 'ids' field (episode id per transition). For each
episode we keep the first max_step transitions (in order), drop episodes shorter
than max_step so the episode length stays UNIFORM (the DIFF-IL n-step buffers index
by a fixed episode length -> a ragged length triggers IndexError), then rebuild:
    ids  -> 0..K-1                (contiguous, one id per episode)
    step -> 1..max_step
    don  -> True only on each episode's last kept transition (clean boundaries)
    n    -> K * max_step

Writes 8 files:  <prefix>_{obs,nobs,ims,ids,don,rew,n,acs}.npy   (act -> "acs")

Usage (one buffer):
    python truncate_and_split.py \
        --npz expert_data/XArm6Reach/XArm6Reach.npz \
        --out-dir data100/XArm6Reach --max-step 100 --prefix expert
"""
from __future__ import annotations

import os
import argparse
import numpy as np

# bundled npz key -> loader field name ('step' is unused -> not written).
FIELD_MAP = {
    "obs":  "obs",
    "nobs": "nobs",
    "ims":  "ims",
    "ids":  "ids",
    "don":  "don",
    "rew":  "rew",
    "n":    "n",
    "act":  "acs",     # disk 'acs' <- memory 'act'
}


def episode_slices(ids):
    """Contiguous (start, stop) index ranges, one per episode run in `ids`."""
    ids = np.asarray(ids)
    change = np.nonzero(np.diff(ids) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    stops  = np.concatenate([change, [len(ids)]])
    return list(zip(starts.tolist(), stops.tolist()))


def main():
    ap = argparse.ArgumentParser(description="Truncate episodes to max_step + split to per-field .npy")
    ap.add_argument("--npz", required=True, help="bundled dataset (obs,nobs,act,rew,don,ims,ids,step,n)")
    ap.add_argument("--out-dir", default=None, help="output folder (default: <npz_dir>_trunc<max_step>)")
    ap.add_argument("--max-step", type=int, default=100, help="keep first N transitions per episode")
    ap.add_argument("--prefix", default="expert", help='filename prefix (default "expert")')
    ap.add_argument("--save-npz", action="store_true", help="also write a truncated bundle .npz")
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        raise SystemExit(f"not found: {args.npz}")
    out_dir = args.out_dir or (os.path.dirname(os.path.abspath(args.npz)) + f"_trunc{args.max_step}")
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(args.npz)
    keys = list(z.files)
    n_in = int(z["n"]) if "n" in keys else len(z["ids"])
    print(f"[*] loaded {args.npz}\n    keys={keys}  n={n_in}")

    m = args.max_step
    slices = episode_slices(z["ids"])
    keep_idx, kept_ep, dropped = [], 0, 0
    for (s, e) in slices:
        if (e - s) < m:
            dropped += 1
            continue
        keep_idx.append(np.arange(s, s + m))
        kept_ep += 1
    if kept_ep == 0:
        raise SystemExit(f"no episode has >= {m} steps; nothing to keep")
    keep_idx = np.concatenate(keep_idx)

    # gather truncated per-transition arrays
    out = {}
    for k in keys:
        if k == "n":
            continue
        out[k] = np.asarray(z[k])[keep_idx]

    # rebuild bookkeeping fields for uniform, self-consistent episodes
    out["ids"] = np.repeat(np.arange(kept_ep, dtype=np.int32), m)
    if "step" in out:
        out["step"] = np.tile(np.arange(1, m + 1, dtype=np.int32), kept_ep)
    don = np.zeros(kept_ep * m, dtype=bool)
    don[m - 1::m] = True                       # last transition of each episode
    out["don"] = don
    n_new = int(kept_ep * m)

    print(f"[*] episodes: kept {kept_ep}, dropped {dropped} (< {m} steps)")
    print(f"[*] transitions: {n_in} -> {n_new}   (episode length fixed = {m})")
    print(f"[*] writing -> {out_dir}  (prefix '{args.prefix}')")

    # write the 8 per-field .npy
    for src_key, field in FIELD_MAP.items():
        path = os.path.join(out_dir, f"{args.prefix}_{field}.npy")
        if src_key == "n":
            np.save(path, np.asarray(n_new, np.int64))
            print(f"    {args.prefix}_n.npy    <- scalar {n_new}")
            continue
        if src_key not in out:
            print(f"[!] key '{src_key}' not in npz -> skipped")
            continue
        arr = out[src_key]
        np.save(path, arr)
        print(f"    {args.prefix}_{field}.npy   <- '{src_key}'   {arr.shape} {arr.dtype}")

    if args.save_npz:
        bundle = {k: out[k] for k in out}
        bundle["n"] = n_new
        p = os.path.join(out_dir, os.path.basename(args.npz).replace(".npz", f"_trunc{m}.npz"))
        np.savez_compressed(p, **bundle)
        print(f"[*] bundle -> {p}")
    print("[done]")


if __name__ == "__main__":
    main()
