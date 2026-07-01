#!/usr/bin/env python3
"""Make a GIF from a collected dataset's frames — quick visual check of the sim
(or real) render: camera viewpoint, goal marker, and the motion in an episode.

Works on any d3il-format .npz (B^SE/B^SR/B^TR or a real collection): it uses the
newest frame of each step's stack (ims[:, -1]) so you see the actual latest view.

    # one episode, 4x upscaled
    python scripts/diffil/dataset_to_gif.py --npz expert_data/XArm6Reach/XArm6Reach.npz --episode 0

    # first 3 episodes concatenated into one GIF
    python scripts/diffil/dataset_to_gif.py --npz prior_data/XArm6Reach_random/XArm6Reach_random.npz \
        --max-episodes 3 --scale 4 --fps 25
"""
from __future__ import annotations

import os
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Render a collected dataset to a GIF")
    ap.add_argument("--npz", required=True, help="d3il-format dataset (.npz with 'ims','ids')")
    ap.add_argument("--episode", type=int, default=None, help="single episode id to render")
    ap.add_argument("--max-episodes", type=int, default=1,
                    help="if --episode not set, concatenate this many episodes (in order)")
    ap.add_argument("--scale", type=int, default=4, help="upscale factor (NEAREST)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--frame", choices=["newest", "oldest"], default="newest",
                    help="which frame of each 4-stack to show")
    ap.add_argument("--out", default=None, help="output .gif (default: next to the npz)")
    args = ap.parse_args()

    from PIL import Image

    if not os.path.exists(args.npz):
        raise SystemExit(f"not found: {args.npz}")
    z = np.load(args.npz)
    if "ims" not in z.files:
        raise SystemExit("no 'ims' in this npz")
    ims = z["ims"]                                   # [N, past_frames, H, W, C]
    ids = z["ids"] if "ids" in z.files else np.zeros(len(ims), np.int32)
    fidx = -1 if args.frame == "newest" else 0

    eps = np.unique(ids)
    chosen = [args.episode] if args.episode is not None else list(eps[:args.max_episodes])
    if ims.max() == 0:
        print("[!] all frames are 0 (dry-run / black camera) — the GIF will be blank.")

    frames = []
    for e in chosen:
        m = ids == e
        seq = ims[m][:, fidx]                        # [T, H, W, C]
        for f in seq:
            img = Image.fromarray(f.astype(np.uint8))
            if args.scale != 1:
                img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
            frames.append(img)
    if not frames:
        raise SystemExit(f"no frames for episode(s) {chosen}")

    out = args.out or os.path.splitext(args.npz)[0] + (
        f"_ep{args.episode}.gif" if args.episode is not None else f"_ep0-{len(chosen)-1}.gif")
    dur = max(1, int(1000.0 / args.fps))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0)
    print(f"[saved] {out}  ({len(frames)} frames, {chosen if len(chosen)<=10 else len(chosen)} episode(s), "
          f"{frames[0].size[0]}x{frames[0].size[1]})")


if __name__ == "__main__":
    main()
