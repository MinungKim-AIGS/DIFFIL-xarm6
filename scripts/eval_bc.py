"""Evaluate a BC policy (trained with train_bc.py) on XArm6 environments.

Usage:
    python eval_bc.py --task reach --model outputs/bc_reach/bc_policy_best.pt
    python eval_bc.py --task reach --model outputs/bc_reach/bc_policy_best.pt --episodes 100
    python eval_bc.py --task pick_place --model outputs/bc_pick/bc_policy_best.pt --out_json results/bc_reach.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import xarm_rl  # noqa: F401  registers envs


TASK_TO_ENV = {"reach": "XArm6Reach-v0", "pick_place": "XArm6PickPlace-v0"}


# ─────────────────────────── model (must match train_bc.py) ───────────────── #
class BCPolicy(nn.Module):
    """MLP policy — architecture must match the checkpoint being loaded."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: list[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ─────────────────────────── checkpoint loading ───────────────────────────── #
def load_bc_policy(model_path: str, device: torch.device) -> BCPolicy:
    """Load a checkpoint saved by train_bc.py.

    Supports two formats:
      - dict checkpoint (bc_policy_best.pt):  contains obs_dim, action_dim, hidden, model_state
      - bare state_dict (bc_policy_final.pt): falls back to default arch [256, 256]
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # rich checkpoint saved at best val epoch
        obs_dim    = ckpt.get("obs_dim",    21)
        action_dim = ckpt.get("action_dim",  6)
        hidden     = ckpt.get("hidden",     [256, 256])
        state_dict = ckpt["model_state"]
        epoch_info = f"  (saved at epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.6f})"
    else:
        # bare state_dict — infer full architecture from weight shapes
        state_dict  = ckpt
        weight_keys = [k for k in ckpt if k.endswith(".weight")]
        obs_dim     = ckpt[weight_keys[0]].shape[1]          # first layer input
        action_dim  = ckpt[weight_keys[-1]].shape[0]         # last layer output
        hidden      = [ckpt[k].shape[0] for k in weight_keys[:-1]]  # intermediate outputs
        epoch_info  = "  (bare state_dict, arch inferred from weight shapes)"

    model = BCPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[eval_bc] loaded {model_path}{epoch_info}")
    print(f"[eval_bc] arch: obs_dim={obs_dim}  action_dim={action_dim}  hidden={hidden}")
    return model


# ─────────────────────────── inference helper ─────────────────────────────── #
@torch.no_grad()
def predict(model: BCPolicy, obs: np.ndarray, device: torch.device) -> np.ndarray:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action = model(obs_t).squeeze(0).cpu().numpy()
    return action


# ─────────────────────────── main ─────────────────────────────────────────── #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",     choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--model",    required=True,
                    help="Path to .pt checkpoint from train_bc.py")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed",     type=int, default=1000)
    ap.add_argument("--out_json", type=str, default=None)
    ap.add_argument("--device",   type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--render",   action="store_true",
                    help="Render the environment (requires display)")
    args = ap.parse_args()

    # ── device ──
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    # ── env & model ──
    render_mode = "human" if args.render else None
    env   = gym.make(TASK_TO_ENV[args.task], render_mode=render_mode)
    model = load_bc_policy(args.model, device)

    # ── rollout ──
    rewards, successes, ep_lens, last_dists = [], [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, ep_r, t = False, 0.0, 0
        info = {}

        while not done:
            action          = predict(model, obs, device)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            t    += 1
            done  = term or trunc

        rewards.append(ep_r)
        successes.append(float(info.get("is_success", 0.0)))
        ep_lens.append(t)

        dist_key = "distance" if args.task == "reach" else "d_cube_target"
        last_dists.append(float(info.get(dist_key, -1)))

        print(f"  ep {ep+1:3d}/{args.episodes}  "
              f"reward={ep_r:7.3f}  success={successes[-1]:.0f}  "
              f"len={t:3d}  dist={last_dists[-1]:.4f}")

    env.close()

    # ── summary ──
    succ_rate = float(np.mean(successes))
    summary = {
        "task":             args.task,
        "algo":             "bc",
        "model":            args.model,
        "episodes":         args.episodes,
        "success_rate":     succ_rate,
        "mean_reward":      float(np.mean(rewards)),
        "std_reward":       float(np.std(rewards)),
        "mean_ep_len":      float(np.mean(ep_lens)),
        "mean_final_dist":  float(np.mean(last_dists)),
    }
    print("\n" + json.dumps(summary, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval_bc] results saved → {args.out_json}")

    return succ_rate


if __name__ == "__main__":
    main()
