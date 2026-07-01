"""Train PPO or SAC on XArm6 Reach/PickPlace via Stable-Baselines3.

Usage:
    python scripts/train.py --task reach --algo ppo
    python scripts/train.py --task reach --algo sac
    python scripts/train.py --task pick_place --algo ppo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback

import xarm_rl  # noqa: F401  registers envs


TASK_TO_ENV = {
    "reach":      "XArm6Reach-v0",
    "pick_place": "XArm6PickPlace-v0",
}

DEFAULT_STEPS = {
    ("ppo", "reach"):      500_000,
    ("sac", "reach"):      200_000,   # SAC is more sample-efficient
    ("ppo", "pick_place"): 3_000_000,
    ("sac", "pick_place"): 1_000_000,
}


def make_env(env_id: str, seed: int, domain_rand: bool = False):
    def _thunk():
        kwargs = {"domain_rand": domain_rand} if env_id.startswith("XArm6Reach") else {}
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed)
        return env
    return _thunk


def build_ppo(vec_env, out: Path, seed: int):
    # Tuned for short-horizon (100-step) reach task:
    # - higher lr + entropy bonus for more exploration
    # - smaller n_steps to update more often
    return PPO(
        "MlpPolicy", vec_env,
        n_steps=256, batch_size=256, n_epochs=10,
        learning_rate=5e-4, gamma=0.98, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(out / "tb"), verbose=1, seed=seed,
        device="auto",
    )


def build_sac(vec_env, out: Path, seed: int):
    # v3: lower target entropy (auto with lower target encourages less exploration
    # at convergence than default). Lower gamma for short 100-step task.
    return SAC(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, buffer_size=500_000,
        batch_size=256, tau=0.01, gamma=0.95,
        train_freq=1, gradient_steps=1,
        ent_coef="auto_0.1",       # auto-tuned but starting lower
        target_entropy=-6.0,       # = -action_dim/2, stronger pressure toward determinism
        learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(out / "tb"), verbose=1, seed=seed,
        device="auto",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASK_TO_ENV), required=True)
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--n_envs", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--domain_rand", action="store_true",
                    help="enable domain randomization (Reach only)")
    ap.add_argument("--eval-freq", type=int, default=25_000,
                    help="eval (and best-model save) every N total timesteps")
    ap.add_argument("--n-eval-episodes", type=int, default=10)
    args = ap.parse_args()

    env_id = TASK_TO_ENV[args.task]
    timesteps = args.timesteps or DEFAULT_STEPS[(args.algo, args.task)]
    out = Path(args.out or f"outputs/{args.task}_{args.algo}")
    out.mkdir(parents=True, exist_ok=True)

    # PPO scales with parallel envs; SAC uses replay buffer (1 env is fine)
    n_envs = args.n_envs or (8 if args.algo == "ppo" else 1)

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(env_id, args.seed, args.domain_rand)])
    else:
        vec_env = SubprocVecEnv([make_env(env_id, args.seed + i, args.domain_rand)
                                 for i in range(n_envs)])
    vec_env = VecMonitor(vec_env, filename=str(out / "monitor.csv"))

    model = build_ppo(vec_env, out, args.seed) if args.algo == "ppo" else build_sac(vec_env, out, args.seed)

    # Evaluate on the NOMINAL task (no domain randomization) and keep only the
    # BEST policy by eval reward -> guards against overfitting / late collapse.
    eval_env = DummyVecEnv([make_env(env_id, args.seed + 1000, domain_rand=False)])
    eval_env = VecMonitor(eval_env)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out),                 # writes <out>/best_model.zip
        log_path=str(out / "eval"),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True, render=False,
    )

    print(f"[train] task={args.task} algo={args.algo} n_envs={n_envs} "
          f"timesteps={timesteps:,} domain_rand={args.domain_rand}")
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    final_path = out / "final_model"                   # the LAST policy
    model.save(final_path)
    print(f"[train] saved BEST -> {out / 'best_model'}.zip   LAST -> {final_path}.zip")


if __name__ == "__main__":
    main()
