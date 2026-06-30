#!/usr/bin/env python3
"""Check / conform a d3il-format .npz to what DIFF-IL's DemonstrationsReplayBuffer
expects (used for the fixed datasets B^SE, B^SR, B^TR).

DemonstrationsReplayBuffer relies on:
    ims : [N, past_frames, H, W, C] uint8     (uses ims[:,0] and ims[:,1])
    ids : [N] int                             (np.unique -> per-episode first_indices)
    act : [N, A] float
    don : [N] bool
    past_frames = ims.shape[1]

Our sim/real collectors already emit obs/nobs/act/rew/don/ims/ids/n, so this is
mostly a validator; it can also synthesize `ids` from episode boundaries if absent.

    python scripts/diffil/dataset_conform.py --npz data/real_reach/xarm6_real_reach_dataset.npz
"""
from __future__ import annotations

import argparse
import numpy as np


def check_demo_npz(path: str, past_frames: int = 4, epi_len: int = 200, fix_out: str | None = None):
    z = dict(np.load(path, allow_pickle=False))
    report, ok = [], True

    def need(cond, msg):
        nonlocal ok
        report.append(("OK  " if cond else "FAIL") + " " + msg)
        ok = ok and cond

    need("ims" in z, "has 'ims'")
    if "ims" in z:
        ims = z["ims"]
        need(ims.ndim == 5, f"ims is 5-D [N,past,H,W,C] (got {ims.shape})")
        need(ims.shape[1] == past_frames, f"ims past_frames == {past_frames} (got {ims.shape[1]})")
        need(ims.dtype == np.uint8, f"ims dtype uint8 (got {ims.dtype})")
    need("act" in z and z["act"].ndim == 2, "has 2-D 'act' [N,A]")
    need("don" in z, "has 'don'")

    # ids: required for episode grouping; synthesize if missing
    if "ids" not in z:
        report.append("WARN no 'ids' — synthesizing from 'don'/epi_len")
        N = z["ims"].shape[0]
        if "don" in z and z["don"].sum() > 0:
            ids = np.zeros(N, np.int32); cur = 0
            for i in range(N):
                ids[i] = cur
                if z["don"][i]:
                    cur += 1
        else:
            ids = (np.arange(N) // epi_len).astype(np.int32)
        z["ids"] = ids
    else:
        need(z["ids"].ndim == 1, "ids is 1-D")

    # episode lengths from ids
    if "ids" in z:
        _, counts = np.unique(z["ids"], return_counts=True)
        report.append(f"INFO episodes={len(counts)}, len min/mean/max="
                      f"{counts.min()}/{counts.mean():.0f}/{counts.max()}, N={z['ims'].shape[0]}")
        if not np.all(counts == counts[0]):
            report.append("WARN variable episode lengths — fine (DIFF-IL uses ids first_indices), "
                          "but keep buffer epi_len consistent with the dominant length")

    print(f"=== conform check: {path} ===")
    for r in report:
        print("  " + r)
    print("  RESULT:", "PASS" if ok else "FAIL")

    if fix_out and ok:
        np.savez_compressed(fix_out, **z)
        print(f"  wrote conformed copy (with ids) -> {fix_out}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--past-frames", type=int, default=4)
    ap.add_argument("--epi-len", type=int, default=200)
    ap.add_argument("--fix-out", default=None, help="write conformed copy (adds ids if missing)")
    args = ap.parse_args()
    check_demo_npz(args.npz, args.past_frames, args.epi_len, args.fix_out)


if __name__ == "__main__":
    main()
