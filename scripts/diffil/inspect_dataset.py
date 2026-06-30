#!/usr/bin/env python3
"""Inspect a collected DIFF-IL dataset (.npz) — quick sanity view of the
preprocessed data, so you can confirm a collection (B^TR / B^SR / B^SE) looks right
before feeding it to training.

It prints a stats summary and writes three image files:
  - montage.png : the 4-frame stack for several samples (oldest -> newest), so you
                  can see exactly what the encoder will receive.
  - episode.png : action curves (6 joints) + reward for one episode.
  - episode.gif : that episode played back (newest frame of each step).

No training deps — just numpy + matplotlib + Pillow. Runs anywhere.

    python scripts/diffil/inspect_dataset.py \
        --npz data/real_reach/xarm6_real_reach_dataset.npz \
        --episode 0 --num-samples 8 --gif
"""
from __future__ import annotations

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(npz_path):
    if not os.path.exists(npz_path):
        raise SystemExit(f"not found: {npz_path}")
    z = np.load(npz_path)
    d = {k: z[k] for k in z.files}
    if "ids" not in d:
        d["ids"] = np.zeros(len(d["act"]), np.int32)   # treat as a single episode
    return d


def summary(d):
    act, rew, ids = d["act"], d.get("rew"), d["ids"]
    n = int(d["n"]) if "n" in d and np.ndim(d["n"]) == 0 else len(act)
    eps = np.unique(ids)
    lens = [int((ids == e).sum()) for e in eps]
    print("=== DATASET SUMMARY ===")
    print(f"  samples N      : {n}")
    print(f"  episodes       : {len(eps)}   ep-len min/mean/max = "
          f"{min(lens)}/{np.mean(lens):.0f}/{max(lens)}")
    for k in ("obs", "nobs", "act", "ims"):
        if k in d:
            a = d[k]
            print(f"  {k:5s} {str(a.shape):20s} {str(a.dtype):8s} "
                  f"range[{a.min():.2f},{a.max():.2f}]")
    print(f"  act per-dim |mean| : {np.abs(act).mean(0).round(2)}")
    if rew is not None:
        print(f"  reward         : range[{rew.min():.2f},{rew.max():.2f}] mean={rew.mean():.2f}")
    ims = d.get("ims")
    if ims is not None and int(ims.max()) == 0:
        print("  [!] WARNING: all image pixels are 0 — looks like a dry-run / "
              "DummyCamera collection, or the camera produced black frames.")
    return eps, lens


def save_montage(d, out_path, num_samples):
    ims = d.get("ims")
    if ims is None:
        print("  (no 'ims' in dataset — skipping montage)")
        return
    N, PF = ims.shape[0], ims.shape[1]
    idx = np.linspace(0, N - 1, min(num_samples, N)).astype(int)
    fig, ax = plt.subplots(len(idx), PF, figsize=(PF * 1.6, len(idx) * 1.6),
                           squeeze=False)
    for r, i in enumerate(idx):
        for c in range(PF):
            ax[r, c].imshow(ims[i, c]); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(f"t-{PF - 1 - c}", fontsize=8)
        ax[r, 0].set_ylabel(f"#{i}", fontsize=8, rotation=0, labelpad=14)
    fig.suptitle("4-frame stacks (oldest → newest)")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    print(f"  saved {out_path}")


def save_episode(d, out_png, out_gif, ep_id, make_gif):
    act, ids = d["act"], d["ids"]
    rew = d.get("rew")
    m = ids == ep_id
    if not m.any():
        print(f"  (episode {ep_id} not found — skipping episode plot)")
        return
    nrow = 2 if rew is not None else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(8, 2.6 * nrow), sharex=True, squeeze=False)
    a1 = axes[0, 0]
    for j in range(act.shape[1]):
        a1.plot(act[m][:, j], label=f"a{j}")
    a1.set_ylabel("action [-1,1]"); a1.set_ylim(-1.05, 1.05)
    a1.legend(ncol=act.shape[1], fontsize=7)
    if rew is not None:
        a2 = axes[1, 0]
        a2.plot(rew[m], color="crimson"); a2.set_ylabel("reward")
    axes[-1, 0].set_xlabel("step")
    fig.suptitle(f"episode {ep_id}  ({int(m.sum())} steps)")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"  saved {out_png}")

    if make_gif and "ims" in d:
        from PIL import Image
        frames = [Image.fromarray(f) for f in d["ims"][m][:, -1]]   # newest frame/step
        if frames:
            frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                           duration=40, loop=0)
            print(f"  saved {out_gif}")


def main():
    ap = argparse.ArgumentParser(description="Inspect a collected DIFF-IL .npz dataset")
    ap.add_argument("--npz", required=True, help="path to the dataset .npz")
    ap.add_argument("--out-dir", default=None,
                    help="where to write images (default: <npz_dir>/inspect)")
    ap.add_argument("--episode", type=int, default=0, help="episode id for the plot/gif")
    ap.add_argument("--num-samples", type=int, default=8,
                    help="how many samples to show in the montage")
    ap.add_argument("--gif", action="store_true", help="also write episode.gif")
    args = ap.parse_args()

    d = load(args.npz)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.npz)), "inspect")
    os.makedirs(out_dir, exist_ok=True)

    summary(d)
    print("=== WRITING VIEWS ===")
    save_montage(d, os.path.join(out_dir, "montage.png"), args.num_samples)
    save_episode(d, os.path.join(out_dir, "episode.png"),
                 os.path.join(out_dir, "episode.gif"), args.episode, args.gif)
    print(f"[done] inspect outputs -> {out_dir}")


if __name__ == "__main__":
    main()
