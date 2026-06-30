#!/usr/bin/env python3
"""build_diffil(cfg) -> (gail, agent_buffer, l_agent, sampler)

A faithful, condensed port of the DIFF-IL graph/buffer construction in
``run_experiment_cycle.run_experiment()`` for the **xArm6 reach** task. This is a
DROP-IN for the DIFF-IL repo (server): it imports the DIFF-IL modules
(gail_models_cycle, disentangail_models_cycle, sac_models, buffers_cycle,
samplers, utils) and the xArm6 source env adapter.

learner_node.RealLearner.build() calls this; everything else (target feed from the
actor, actor export/publish, the train loop) is already wired in learner_node.py.

Server PYTHONPATH must include BOTH the DIFF-IL repo and the xarm6 repo
(`scripts/` for diffil_adapter via xarm_rl). TF + the DIFF-IL deps are imported
lazily inside build_diffil() so this file can be parsed without them.

Dataset layout (paths are cfg fields; format == our d3il npz, see dataset_conform):
    expert_data/<env_name>/...                 -> B^SE  (sim expert,  fixed)
    prior_data/<source_random_location>/...    -> B^SR  (sim random,  fixed)
    prior_data/<target_random_location>/...    -> B^TR  (real random, fixed)
    prior_data/<target_learner_seed>/...       -> B^TL  seed (real random rollouts)
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Config — defaults mirror the reference manipulation/Reach hyperparameters.
# ----------------------------------------------------------------------------
@dataclass
class DiffilConfig:
    # env / source adapter
    env_id: str = "XArm6Reach-v0"
    render_camera: str = "front"
    im_side: int = 64
    episode_limit: int = 200
    random_epi_limit: int = 200
    past_frames: int = 4
    feature_size: int = 32
    init_random_samples: int = 1000
    action_dim: int = 6              # xArm6 reach action size (no env needed)
    use_source_env: bool = False    # True -> build SimDiffilEnv (gymnasium+mujoco) for wandb viz only

    # learner (SAC)
    l_type: str = "SAC"
    l_buffer_size: int = 50000
    l_batch_size: int = 256
    l_learning_rate: float = 1e-3
    l_gamma: float = 0.99
    l_polyak: float = 0.995
    l_entropy_coefficient: float = 0.2
    l_tune_entropy_coefficient: bool = True
    l_target_entropy: float | None = None
    l_clip_actor_gradients: bool = False
    l_exploration_noise: float = 0.2

    # discriminator / labels / losses
    d_type: str = "latent"
    d_rew_noise: bool = True
    d_e_batch_size: int = 64
    d_l_batch_size: int = 64
    n_expert_demos: int = 50000
    n_prior_demos: int = 50000
    alpha: float = 0.9
    gen_loss: float = 0.1
    disc_loss: float = 50.0
    recon_scale: float = 1.0
    feature_fake_scale: float = 1.0
    label_loss_se: float = 0.1
    label_loss_sr: float = 0.1
    label_loss_tl: float = 1e-5
    label_loss_tl_relabel: float = 1e-5
    label_loss_tr: float = 1e-4
    percentage: int = 10
    sehat: int = 100
    sampling_alpha: float = 0.0
    model_num_per_epoch: int = 500
    RL_num: int = 1000

    # dataset locations (directories under file_location / prior_file_location)
    file_location: str = "expert_data"
    prior_file_location: str = "prior_data"
    env_name: str = "XArm6Reach"             # B^SE dir
    source_random_location: str = "XArm6Reach_random"     # B^SR
    target_random_location: str = "XArm6Reach_real_random"  # B^TR
    target_learner_seed: str = "XArm6Reach_real_random"     # B^TL seed

    # logging
    log_dir: str = "experiments_data/xarm6_reach"
    run_wandb: object = None                 # pass a wandb run, or None -> no-op


class _NullWandb:
    """Drop-in for a wandb run when none is provided."""
    def log(self, *args, **kwargs):
        pass


class _NullSampler:
    """Learner runs gymnasium-free. DIFF-IL calls the sampler ONLY inside optional
    wandb-visualization methods (plot_img_exec / plot_img task 'tl_exec'); the
    train() path never touches it. Calling it raises a clear, actionable error."""
    def sample_trajectory(self, *a, **k):
        raise RuntimeError("Source-env sampling requested but use_source_env=False. "
                           "Set DiffilConfig.use_source_env=True (needs gymnasium+mujoco) "
                           "to enable source-rollout wandb visualizations.")
    def evaluate(self, *a, **k):
        raise RuntimeError("Source-env evaluation disabled (use_source_env=False).")


def build_diffil(cfg: DiffilConfig):
    """Construct (gail, agent_buffer, l_agent, sampler) for xArm6 reach.

    Mirrors run_experiment_cycle: make_* factories, dataset buffers, SAC agent,
    Sampler over the source sim env, and the DisentanGAIL orchestrator.
    """
    import os
    import numpy as np
    import tensorflow as tf

    # --- DIFF-IL repo modules (must be importable on the server) ---
    from sac_models import StochasticActor, SAC, Critic
    from samplers import Sampler
    from buffers_cycle import LearnerAgentReplayBuffer, DemonstrationsReplayBuffer
    from gail_models_cycle import (Featurediscriminator, Fakefeaturediscriminator,
                                   Feature_generator, GRUPreprocessor, Reconstruct,
                                   Labelnet, Labelnet_frame)
    from disentangail_models_cycle import DisentanGAIL
    from utils import load_expert_trajectories, load_learner_trajectories

    run_wandb = cfg.run_wandb if cfg.run_wandb is not None else _NullWandb()
    os.makedirs(cfg.log_dir, exist_ok=True)

    # ---- source env: OPTIONAL (gymnasium-free by default) ----
    # The learner only needs the source env for wandb-viz GIFs, never for training.
    # Keeping it off means this whole node runs on a pure TF stack (e.g. TF 2.5)
    # with NO gymnasium / mujoco / SB3 -> no version conflicts.
    action_size = cfg.action_dim
    if cfg.use_source_env:
        from xarm_rl.envs.diffil_adapter import SimDiffilEnv   # needs gymnasium+mujoco
        env = SimDiffilEnv(env_id=cfg.env_id, render_camera=cfg.render_camera)
        sampler = Sampler(env, cfg.episode_limit, cfg.init_random_samples, visual_env=True)
    else:
        sampler = _NullSampler()

    # ---- datasets (fixed) ----
    expert_buffer = DemonstrationsReplayBuffer(
        load_expert_trajectories(cfg.env_name, cfg.file_location, visual_data=True,
                                 load_ids=True, max_demos=cfg.n_expert_demos),
        cfg.episode_limit)                                            # B^SE
    expert_shape = expert_buffer.get_random_batch(1)['ims'][0].shape
    past_frames = expert_shape[0]
    prior_expert_buffer = DemonstrationsReplayBuffer(
        load_expert_trajectories(cfg.source_random_location, cfg.prior_file_location,
                                 visual_data=True, load_ids=True, max_demos=cfg.n_prior_demos),
        cfg.random_epi_limit)                                        # B^SR
    prior_agent_buffer = DemonstrationsReplayBuffer(
        load_expert_trajectories(cfg.target_random_location, cfg.prior_file_location,
                                 visual_data=True, load_ids=True, max_demos=cfg.n_prior_demos),
        cfg.random_epi_limit)                                        # B^TR

    im_shape = [cfg.im_side, cfg.im_side]
    im_shape += [3] if cfg.d_type == "latent" else [3 * past_frames]

    # ---- SAC factories ----
    target_entropy = cfg.l_target_entropy
    if target_entropy is None:
        target_entropy = -1.0 * float(action_size)

    def make_actor():
        return StochasticActor([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                                tf.keras.layers.Dense(action_size * 2,
                                                      kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])

    def make_critic():
        return Critic([tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                       tf.keras.layers.Dense(256, 'relu', kernel_initializer='orthogonal'),
                       tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])

    feature_size = cfg.feature_size

    def make_pre():
        myim = [None, 4, im_shape[0], im_shape[0], 3]
        myact = [None, action_size]
        non_im = [4, im_shape[0], im_shape[0], 3]
        non_act = [action_size]
        return GRUPreprocessor(feature_size, myim, myact, non_im, non_act)

    def make_fwgan():        return Featurediscriminator(feature_size, cfg.alpha)
    def make_labelnet():     return Labelnet(feature_size)
    def make_labelnet_frame():return Labelnet_frame(feature_size)
    def make_fake_fwgan():   return Fakefeaturediscriminator(feature_size)
    def make_feature_gen():  return Feature_generator(feature_size)

    def make_recon_layer():
        img_shape = [None, feature_size]
        recon_shape = [None, im_shape[0]]
        return Reconstruct(img_shape, recon_shape), Reconstruct(img_shape, recon_shape)

    # ---- SAC agent ----
    l_opt = tf.keras.optimizers.Adam(cfg.l_learning_rate)
    l_agent = SAC(make_actor=make_actor, make_critic=make_critic, make_critic2=make_critic,
                  actor_optimizer=l_opt, critic_optimizer=l_opt, gamma=cfg.l_gamma,
                  polyak=cfg.l_polyak, entropy_coefficient=cfg.l_entropy_coefficient,
                  tune_entropy_coefficient=cfg.l_tune_entropy_coefficient,
                  target_entropy=target_entropy,
                  clip_actor_gradients=cfg.l_clip_actor_gradients, run_wandb=run_wandb)

    # ---- DIFF-IL orchestrator ----
    gail = DisentanGAIL(agent=l_agent, make_recon=make_recon_layer, make_preprocessing=make_pre,
                        make_label=make_labelnet, make_label_frame=make_labelnet_frame,
                        make_fwgan=make_fwgan, make_fake_fwgan=make_fake_fwgan,
                        make_feature_gen=make_feature_gen, expert_buffer=expert_buffer,
                        log_dir=cfg.log_dir, run_wandb=run_wandb,
                        prior_expert_buffer=prior_expert_buffer, prior_agent_buffer=prior_agent_buffer,
                        past_frames=past_frames, im_shape=im_shape, feature_size=feature_size,
                        recon_loss=cfg.recon_scale, feature_fake_loss=cfg.feature_fake_scale,
                        disc_loss=cfg.disc_loss, gen_loss=cfg.gen_loss,
                        label_loss_se=cfg.label_loss_se, label_loss_sr=cfg.label_loss_sr,
                        label_loss_tl=cfg.label_loss_tl, label_tl_relabel=cfg.label_loss_tl_relabel,
                        label_loss_tr=cfg.label_loss_tr, percentage=cfg.percentage, sehat=cfg.sehat,
                        pol_update=cfg.RL_num, sampler=sampler, epi_limit=cfg.episode_limit,
                        random_epi=cfg.random_epi_limit)

    # ---- online target buffer B^TL, seeded with real-random rollouts ----
    # IMPORTANT: LearnerAgentReplayBuffer uses ReplayBuffer.get_balance_batch_
    # nsteps_with_step, which builds indices from self.buffer_size and ASSUMES the
    # buffer is completely full (the reference pre-fills 50000 samples). If
    # buffer_size > the number of samples actually loaded, those indices run off the
    # end of self.obs -> "IndexError: index N out of bounds". So we size the B^TL
    # buffer to the seed we really have (it then stays full as the actor adds data,
    # since add() evicts oldest beyond buffer_size).
    tl_seed = load_learner_trajectories(cfg.target_learner_seed, cfg.prior_file_location,
                                        visual_data=True, load_ids=True,
                                        max_demos=cfg.l_buffer_size)
    tl_n = int(tl_seed["obs"].shape[0])
    eff_buffer = min(int(cfg.l_buffer_size), tl_n)
    if eff_buffer != int(cfg.l_buffer_size):
        print(f"[build_diffil] B^TL buffer_size {cfg.l_buffer_size} -> {eff_buffer} "
              f"(matched to seed samples; the base sampler assumes a full buffer)")
    agent_buffer = LearnerAgentReplayBuffer(
        gail, eff_buffer, cfg.episode_limit, reward_noise=cfg.d_rew_noise,
        initial_data=tl_seed)

    # ---- force-build the SAC actor so it can be exported BEFORE training ----
    # make_actor's Dense layers are lazy: kernels are created only on the first
    # call. learner_node publishes actor v0 before any train step, so without this
    # warm call export_actor() sees empty get_weights() and crashes. The policy is
    # state-based, so we build it on a zero state vector (obs dim taken from B^TL).
    try:
        obs_dim = int(agent_buffer.obs.shape[1])
    except Exception:
        obs_dim = int(cfg.action_dim)            # last-resort fallback; B^TL always has obs
    l_agent._act.get_action(tf.zeros([1, obs_dim], tf.float32), 0.0)

    return gail, agent_buffer, l_agent, sampler
