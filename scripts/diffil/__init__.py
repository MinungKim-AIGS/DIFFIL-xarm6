"""DIFF-IL sim2real glue for xArm6.

Distributed cross-domain imitation (DIFF-IL / DisentanGAIL):
  - learner_node.py : GPU server. Trains encoder/decoders/F_f/F_s/D_f/D_s + SAC
                      on images; publishes the (small) SAC actor weights.
  - actor_node.py   : robot laptop. Runs the actor (TF-free numpy forward) on the
                      real xArm6 at control rate, ships trajectories to the server.

Transport = ZeroMQ (comm.py). Policy on the laptop runs WITHOUT TensorFlow
(weight_io.py exports the actor, policy_runtime.py does a numpy forward).

Nothing here overwrites existing files; the real env reuses real_reach_collector
and the sim env wraps the existing gymnasium XArm6ReachEnv.
"""
