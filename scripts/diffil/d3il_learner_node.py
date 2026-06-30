#!/usr/bin/env python3
"""D3IL LEARNER node (GPU server).

Trains D3IL across the distributed setup, reusing the SAME transport as DIFFIL:
the laptop actor performs the online TARGET (real-robot) rollouts O^TL and streams
them here; this node trains and publishes the SAC actor back.

Two phases (see build_d3il / the D3IL paper):
  Phase 1 (OFFLINE): pretrain the image-translation / feature model on the fixed
                     datasets B^SE/B^SN/B^TN. No actor / robot needed.
  Phase 2 (ONLINE) : feature model FROZEN (it_updates=0); each round = drain O^TL
                     from the actor -> agent_buffer.add -> model.train (expert disc
                     + SAC) -> export & publish the actor.

Mock mode (no TF; same as diffil) is provided for transport tests.

Real run (after `cp build_d3il.py d3il_learner_node.py` into the D3IL repo):
    python d3il_learner_node.py --pretrain-epochs 50000 --min-new-steps 200
Quick smoke (tiny pretrain):
    python d3il_learner_node.py --pretrain-epochs 50 --training-starts 20 \
        --min-new-steps 20 --l-batch-size 32 --d-batch 8
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from comm import TrajectoryReceiver, WeightPublisher
from policy_runtime import random_actor


# ----------------------------------------------------------------------------
# Mock learner — no TF; transport/actor hot-swap test
# ----------------------------------------------------------------------------
def run_mock(args):
    rx = TrajectoryReceiver(args.traj_port)
    pub = WeightPublisher(args.weight_port)
    weights = random_actor(version=0, seed=0)
    time.sleep(0.3)
    pub.publish(weights)
    print("[d3il/mock] published actor v0; waiting for trajectories...")
    total, rounds = 0, 0
    try:
        while True:
            trajs = rx.drain(max_msgs=64)
            if trajs:
                for t in trajs:
                    total += t["n"]
                print(f"[d3il/mock] +{len(trajs)} traj ({total} steps); last ims {trajs[-1]['ims'].shape}")
                rounds += 1
                if rounds % args.publish_every == 0:
                    weights["version"] += 1
                    for L in weights["layers"]:
                        L["W"] = L["W"] + np.random.randn(*L["W"].shape).astype(np.float32) * 1e-3
                    pub.publish(weights)
                    print(f"[d3il/mock] published actor v{weights['version']}")
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[d3il/mock] stopped")
    finally:
        rx.close(); pub.close()


# ----------------------------------------------------------------------------
# Real D3IL learner
# ----------------------------------------------------------------------------
class D3ILLearner:
    def __init__(self, args):
        self.args = args
        self.rx = TrajectoryReceiver(args.traj_port)
        self.pub = WeightPublisher(args.weight_port)
        self.version = 0
        self.step_counter = 0

    def build(self):
        import weight_io
        from build_d3il import D3ilConfig, build_d3il, pretrain_image_translation, policy_train_step
        self._weight_io = weight_io
        self._pretrain_fn = pretrain_image_translation
        self._policy_step = policy_train_step
        a = self.args
        self.cfg = D3ilConfig(
            im_side=a.im_side, action_dim=a.action_dim, obs_dim=a.obs_dim,
            episode_limit=a.episode_limit, l_buffer_size=a.l_buffer_size,
            l_batch_size=a.l_batch_size, d_batch_size=a.d_batch,
            file_location=a.file_location, prior_file_location=a.prior_file_location,
            se_name=a.se_name, sn_name=a.sn_name, tn_name=a.tn_name, n_demos=a.n_demos)
        (self.model, self.agent_buffer, self.l_agent,
         self.se_buffer, self.sn_buffer, self.tn_buffer) = build_d3il(self.cfg)
        print("[d3il] graph + buffers built (B^SE/B^SN/B^TN loaded)")

    def pretrain(self):
        a = self.args
        if a.pretrain_epochs > 0:
            print(f"[d3il] Phase 1: pretraining feature model for {a.pretrain_epochs} epochs (offline)")
            self._pretrain_fn(self.model, self.se_buffer, self.sn_buffer, self.tn_buffer,
                              self.cfg, a.pretrain_epochs, log_interval=a.pretrain_log_interval)
            print("[d3il] Phase 1 done (feature model now frozen for policy phase)")
        else:
            print("[d3il] Phase 1 skipped (--pretrain-epochs 0); feature model is NOT pretrained")

    def feed_target(self, max_msgs: int = 256) -> int:
        n = 0
        for t in self.rx.drain(max_msgs=max_msgs):
            self.agent_buffer.add({k: t[k] for k in ("obs", "nobs", "act", "rew", "don", "ims", "n")})
            n += t["n"]
        return n

    def publish_actor(self):
        import tensorflow as tf
        act = self.l_agent._act
        if not act._act_layers[-1].get_weights():
            act.get_action(tf.zeros([1, self.cfg.obs_dim], tf.float32), 0.0)
        self.pub.publish(self._weight_io.export_actor(act, version=self.version))

    def run(self):
        self.build()
        self.pretrain()
        time.sleep(0.3)
        self.publish_actor()                     # v0 so the actor can start collecting
        a = self.args
        print("[d3il] Phase 2: online policy RL (waiting for O^TL from the actor)")
        while True:
            added = self.feed_target()
            self.step_counter += added
            if added < a.min_new_steps or self.step_counter < a.training_starts:
                time.sleep(0.1); continue
            self._policy_step(self.model, self.se_buffer, self.sn_buffer, self.tn_buffer,
                              self.agent_buffer, self.cfg, n_new=added)
            self.version += 1
            self.publish_actor()
            print(f"[d3il] published actor v{self.version} (+{added} new target steps, "
                  f"total {self.step_counter})")

    def close(self):
        self.rx.close(); self.pub.close()


def main():
    ap = argparse.ArgumentParser(description="D3IL learner (GPU server)")
    ap.add_argument("--traj-port", type=int, default=5557)
    ap.add_argument("--weight-port", type=int, default=5558)
    ap.add_argument("--mock", action="store_true", help="no-TF echo learner for transport tests")
    ap.add_argument("--publish-every", type=int, default=1, help="[mock] rounds between publishes")
    # phases
    ap.add_argument("--pretrain-epochs", type=int, default=50000, help="Phase-1 feature-model epochs")
    ap.add_argument("--pretrain-log-interval", type=int, default=100)
    ap.add_argument("--training-starts", type=int, default=512, help="min O^TL steps before policy RL")
    ap.add_argument("--min-new-steps", type=int, default=200, help="fresh O^TL steps per train round")
    # model / learner
    ap.add_argument("--im-side", type=int, default=64)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--obs-dim", type=int, default=21)
    ap.add_argument("--episode-limit", type=int, default=200)
    ap.add_argument("--l-buffer-size", type=int, default=100000)
    ap.add_argument("--l-batch-size", type=int, default=256)
    ap.add_argument("--d-batch", type=int, default=128)
    ap.add_argument("--n-demos", type=int, default=10000)
    # dataset locations
    ap.add_argument("--file-location", default="expert_data")
    ap.add_argument("--prior-file-location", default="prior_data")
    ap.add_argument("--se-name", default="XArm6Reach", help="B^SE dir")
    ap.add_argument("--sn-name", default="XArm6Reach_random", help="B^SN dir")
    ap.add_argument("--tn-name", default="XArm6Reach_real_random", help="B^TN dir")
    args = ap.parse_args()

    if args.mock:
        run_mock(args)
    else:
        learner = D3ILLearner(args)
        try:
            learner.run()
        except KeyboardInterrupt:
            print("\n[d3il] stopped")
        finally:
            learner.close()


if __name__ == "__main__":
    main()
