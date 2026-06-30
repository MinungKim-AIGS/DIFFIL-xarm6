"""Behavioural Cloning (BC) for XArm6 Reach from offline trajectory data.

Usage:
    python train_bc.py --data trajectory_to_goals.csv
    python train_bc.py --data trajectory_to_goals.csv --epochs 100 --lr 3e-4
    python train_bc.py --data trajectory_to_goals.csv --out outputs/bc_reach

Observation : input_obs0 ~ input_obs20  (21-dim)
Action target: output_action_clip1 ~ output_action_clip6  (6-dim clipped joint velocity)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ─────────────────────────────── constants ────────────────────────────────── #
OBS_COLS    = [f"input_obs{i}"          for i in range(21)]   # 21-dim
ACTION_COLS = [f"output_action_clip{i}" for i in range(1, 7)] # 6-dim
OBS_DIM     = len(OBS_COLS)    # 21
ACTION_DIM  = len(ACTION_COLS) # 6


# ─────────────────────────────── model ────────────────────────────────────── #
class BCPolicy(nn.Module):
    """Simple MLP policy: obs → action."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden is None:
            hidden = [256, 256]

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


# ─────────────────────────────── data ─────────────────────────────────────── #
def load_dataset(csv_path: str) -> TensorDataset:
    df = pd.read_csv(csv_path)

    missing_obs = [c for c in OBS_COLS    if c not in df.columns]
    missing_act = [c for c in ACTION_COLS if c not in df.columns]
    if missing_obs or missing_act:
        raise ValueError(f"Missing columns – obs: {missing_obs}, action: {missing_act}")

    n_before = len(df)
    df = df.dropna(subset=OBS_COLS + ACTION_COLS).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[data] dropped {n_dropped} rows with NaN values")

    obs    = torch.tensor(df[OBS_COLS].values,    dtype=torch.float32)
    action = torch.tensor(df[ACTION_COLS].values, dtype=torch.float32)

    print(f"[data] rows={len(df):,}  obs={obs.shape}  action={action.shape}")
    return TensorDataset(obs, action)


# ─────────────────────────────── train / eval ─────────────────────────────── #
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """One forward pass over `loader`. Backprop only when optimizer is given."""
    model.train(optimizer is not None)
    total_loss = 0.0

    with torch.set_grad_enabled(optimizer is not None):
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(obs_batch)
            loss = criterion(pred, act_batch)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(obs_batch)

    return total_loss / len(loader.dataset)


# ─────────────────────────────── main ─────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",    type=str, required=True,
                    help="Path to trajectory CSV")
    ap.add_argument("--out",     type=str, default="outputs/bc_reach",
                    help="Directory for checkpoints and logs")
    ap.add_argument("--epochs",  type=int, default=50)
    ap.add_argument("--lr",      type=float, default=3e-4)
    ap.add_argument("--batch",   type=int, default=256)
    ap.add_argument("--val_frac",type=float, default=0.1,
                    help="Fraction of data held out for validation")
    ap.add_argument("--hidden",  type=int, nargs="+", default=[256, 256],
                    help="Hidden layer sizes, e.g. --hidden 256 256")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--device",  type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
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
    print(f"[train_bc] device={device}")

    # ── reproducibility ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── output dir ──
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── dataset / split ──
    dataset = load_dataset(args.data)
    n_val   = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)
    print(f"[data] train={n_train:,}  val={n_val:,}")

    # ── model / optimiser / loss ──
    model     = BCPolicy(hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"[model] {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] total params: {total_params:,}")

    # ── training loop ──
    best_val_loss = float("inf")
    log_rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = run_epoch(model, val_loader,   None,      criterion, device)
        scheduler.step()

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out / "bc_policy_best.pt"
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "obs_dim":    OBS_DIM,
                "action_dim": ACTION_DIM,
                "hidden":     args.hidden,
            }, ckpt_path)
            print(f"  → best checkpoint saved ({ckpt_path})")

    # save final model
    final_path = out / "bc_policy_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[train_bc] final model saved → {final_path}")

    # save training log
    log_path = out / "train_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"[train_bc] training log saved → {log_path}")


if __name__ == "__main__":
    main()