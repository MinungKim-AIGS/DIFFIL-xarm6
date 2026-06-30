#!/usr/bin/env python3
"""DIFF-IL LEARNER node (GPU server).

Receives target (real-robot) trajectories from the actor, feeds them into the
online learner buffer (B^TL), runs DIFF-IL training (encoder/decoders/F_f/F_s/
D_f/D_s + SAC on images), and publishes the small SAC actor weights back.

Two modes:
  --mock   : NO TensorFlow. Echoes received trajectories, publishes a (randomly
             perturbed) actor every --publish-every rounds. Use this to test the
             whole comm + actor hot-swap loop end-to-end without the GPU stack.
  (default): real DIFF-IL. Wires the receiver into the existing TF training from
             run_experiment_cycle (see RealLearner below). Requires TF + the
             uploaded DIFF-IL modules on the server.

Mock (pairs with actor_node.py --dry-run):
    python scripts/diffil/learner_node.py --mock --publish-every 1
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from comm import TrajectoryReceiver, WeightPublisher
from policy_runtime import random_actor


# ----------------------------------------------------------------------------
# Mock learner — no TF; for end-to-end transport/actor testing
# ----------------------------------------------------------------------------
def run_mock(args):
    rx = TrajectoryReceiver(args.traj_port)
    pub = WeightPublisher(args.weight_port)
    weights = random_actor(version=0, seed=0)
    time.sleep(0.3)                      # let SUB connect
    pub.publish(weights)
    print("[learner/mock] published actor v0; waiting for trajectories...")

    total_steps, rounds = 0, 0
    try:
        while True:
            trajs = rx.drain(max_msgs=64)
            if trajs:
                for t in trajs:
                    total_steps += t["n"]
                print(f"[learner/mock] +{len(trajs)} traj ({total_steps} steps total); "
                      f"last ims {trajs[-1]['ims'].shape}, obs {trajs[-1]['obs'].shape}")
                rounds += 1
                if rounds % args.publish_every == 0:
                    # pretend we trained: bump version, perturb weights, publish
                    weights["version"] += 1
                    for L in weights["layers"]:
                        L["W"] = L["W"] + np.random.randn(*L["W"].shape).astype(np.float32) * 1e-3
                    pub.publish(weights)
                    print(f"[learner/mock] published actor v{weights['version']}")
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[learner/mock] stopped")
    finally:
        rx.close(); pub.close()


# ----------------------------------------------------------------------------
# Real learner — TF + DIFF-IL (structured seam around run_experiment_cycle)
# ----------------------------------------------------------------------------
class RealLearner:
    """Wires the network receiver into the existing DIFF-IL training.

    Construction of the DIFF-IL graph (encoder p, decoders q^S/q^T, F_f, F_s, D_f,
    D_s, SAC l_agent, buffers B^SE/B^SR/B^TR) is identical to run_experiment_cycle;
    `build()` should reuse that code. Only three seams differ from the offline
    reference:

      1. target collection: instead of `sampler.sample_trajectory -> agent_buffer.add`,
         pull trajectories from the actor and add them (`feed_target`).
      2. after each train round, export the SAC actor and publish it.
      3. source datasets B^SE/B^SR (+ B^TR) are loaded once (fixed), as in the paper.
    """

    def __init__(self, args):
        self.args = args
        self.rx = TrajectoryReceiver(args.traj_port)
        self.pub = WeightPublisher(args.weight_port)
        self.gail = None
        self.agent_buffer = None
        self.l_agent = None
        self.version = 0

    def build(self):
        # Lazy, server-only imports (TF + DIFF-IL).
        import weight_io
        from build_diffil import DiffilConfig, build_diffil
        self._weight_io = weight_io
        a = self.args
        cfg = DiffilConfig(
            env_id=a.env_id, render_camera=a.render_camera, episode_limit=a.episode_limit,
            random_epi_limit=a.episode_limit, l_batch_size=a.l_batch_size, RL_num=a.rl_updates,
            model_num_per_epoch=a.model_updates, d_e_batch_size=a.d_batch, d_l_batch_size=a.d_batch,
            sampling_alpha=a.sampling_alpha, file_location=a.file_location,
            prior_file_location=a.prior_file_location, env_name=a.env_name,
            source_random_location=a.source_random, target_random_location=a.target_random,
            target_learner_seed=a.target_seed, action_dim=a.action_dim,
            use_source_env=a.use_source_env)
        self.gail, self.agent_buffer, self.l_agent, self.sampler = build_diffil(cfg)
        print("[learner] DIFF-IL graph + buffers built (B^SE/B^SR/B^TR loaded, B^TL seeded)")

    def feed_target(self, max_msgs: int = 256) -> int:
        """Drain actor trajectories into the online learner buffer B^TL."""
        n = 0
        for t in self.rx.drain(max_msgs=max_msgs):
            # agent_buffer is a LearnerAgentReplayBuffer (VisualReplayBuffer.add
            # consumes obs/nobs/act/rew/don/ims/n). Rewards are recomputed inside
            # the buffer via the label nets at sample time, so the actor's rew is
            # only a placeholder.
            self.agent_buffer.add({k: t[k] for k in ("obs", "nobs", "act", "rew", "don", "ims", "n")})
            n += t["n"]
        return n

    def publish_actor(self):
        w = self._weight_io.export_actor(self.l_agent._act, version=self.version)
        self.pub.publish(w)

    def run(self):
        self.build()
        time.sleep(0.3)
        self.publish_actor()                     # send v0 so the actor can start
        a = self.args
        while True:
            added = self.feed_target()
            if added < a.min_new_steps:           # wait for enough fresh target data
                time.sleep(0.1); continue
            # one DIFF-IL training round (model + RL), identical call to the reference
            self.gail.train(agent_buffer=self.agent_buffer,
                            l_batch_size=a.l_batch_size, l_updates=a.rl_updates, l_act_delay=1,
                            d_updates=a.model_updates, mi_updates=10,
                            d_e_batch_size=a.d_batch, d_l_batch_size=a.d_batch,
                            sampling_alpha=a.sampling_alpha)
            self.version += 1
            self.publish_actor()
            print(f"[learner] published actor v{self.version} (+{added} new target steps)")

    def close(self):
        self.rx.close(); self.pub.close()


def main():
    ap = argparse.ArgumentParser(description="DIFF-IL learner (GPU server)")
    ap.add_argument("--traj-port", type=int, default=5557)
    ap.add_argument("--weight-port", type=int, default=5558)
    ap.add_argument("--mock", action="store_true", help="no-TF echo learner for e2e tests")
    ap.add_argument("--publish-every", type=int, default=1, help="[mock] rounds between publishes")
    # real-mode DIFF-IL knobs (passed through to gail.train / build)
    ap.add_argument("--min-new-steps", type=int, default=200)
    ap.add_argument("--l-batch-size", type=int, default=256)
    ap.add_argument("--rl-updates", type=int, default=1000)
    ap.add_argument("--model-updates", type=int, default=500)
    ap.add_argument("--d-batch", type=int, default=64)
    ap.add_argument("--sampling-alpha", type=float, default=0.0)
    # source env + dataset locations (real mode)
    ap.add_argument("--env-id", default="XArm6Reach-v0")
    ap.add_argument("--render-camera", default="front")
    ap.add_argument("--episode-limit", type=int, default=200)
    ap.add_argument("--action-dim", type=int, default=6, help="reach action size (no env needed)")
    ap.add_argument("--use-source-env", action="store_true", help="build sim source env for wandb viz (needs gymnasium+mujoco)")
    ap.add_argument("--file-location", default="expert_data")
    ap.add_argument("--prior-file-location", default="prior_data")
    ap.add_argument("--env-name", default="XArm6Reach", help="B^SE dir under file-location")
    ap.add_argument("--source-random", default="XArm6Reach_random", help="B^SR dir under prior-file-location")
    ap.add_argument("--target-random", default="XArm6Reach_real_random", help="B^TR dir")
    ap.add_argument("--target-seed", default="XArm6Reach_real_random", help="B^TL seed dir")
    args = ap.parse_args()

    if args.mock:
        run_mock(args)
    else:
        learner = RealLearner(args)
        try:
            learner.run()
        except KeyboardInterrupt:
            print("\n[learner] stopped")
        finally:
            learner.close()


if __name__ == "__main__":
    main()
