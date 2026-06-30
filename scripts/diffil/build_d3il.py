#!/usr/bin/env python3
"""build_d3il(cfg) -> (model, agent_buffer, l_agent, se_buffer, sn_buffer, tn_buffer)

A faithful, condensed port of the D3IL graph/buffer construction in
``run_d3il.run_experiment()`` for the **xArm6 reach** task, as a DROP-IN for the
D3IL repo (server). It imports the D3IL modules (custom_code.run_imitation.d3il,
sac_models, buffers, utils) and is **gymnasium-free**: no source/target env is
constructed here (action size + a zero state vector are used instead), because in
our distributed setup the laptop actor performs the online TARGET (real-robot)
rollouts (O^TL) and streams them; the server only trains.

Two D3IL phases (see the D3IL paper / d3il.D3ILModelwithPolicy.train):
  Phase 1 (OFFLINE): pretrain the image-translation / feature model on the FIXED
                     datasets B^SE / B^SN / B^TN. No robot interaction.
  Phase 2 (ONLINE) : freeze the encoder (it_updates=0); loop = collect O^TL ->
                     update expert discriminator (IRL reward) -> SAC policy update.

Helpers `pretrain_image_translation()` and `policy_train_step()` wrap the two
phases so d3il_learner_node.py stays thin.

Dataset layout (== our per-field .npy via npz_to_npy.py; loaded by utils):
    expert_data/<se_name>/expert_*.npy   -> B^SE  (source/sim expert)
    prior_data/<sn_name>/expert_*.npy    -> B^SN  (source/sim non-expert)
    prior_data/<tn_name>/expert_*.npy    -> B^TN  (target/real non-expert)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class D3ilConfig:
    # task / shapes
    im_side: int = 64
    past_frames: int = 4
    action_dim: int = 6              # xArm6 reach action size (no env needed)
    obs_dim: int = 21                # reach state dim (for the SAC actor input)
    episode_limit: int = 200

    # image-translation (feature) model coefficients  (== run_d3il defaults)
    c_gan_trans: float = 1.0
    c_gan_feat: float = 0.01
    c_recon: float = 100000.0
    c_cycle: float = 100000.0
    c_feat_mean: float = 1000.0
    c_feat_recon: float = 1000.0
    c_feat_reg: float = 0.1
    c_feat_cycle: float = 0.0
    type_recon_loss: str = "l2"
    eg_update_interval: int = 1
    it_max_grad_norm = None
    it_lr: float = 3e-4
    it_batch_size: int = 8
    c_norm_de: float = 0.0           # task-specific normalization (Reacher-like -> 0)
    c_norm_be: float = 0.0

    # discriminator (IRL reward)
    d_rew: str = "mixed"
    d_max_grad_norm = None
    d_learning_rate: float = 1e-3
    d_batch_size: int = 128
    d_updates_per_step: float = 0.02

    # learner (SAC)  (== run_d3il SAC defaults)
    l_buffer_size: int = 100000
    l_learning_rate: float = 1e-3
    l_batch_size: int = 256
    l_updates_per_step: float = 1.0
    l_act_delay: int = 1
    l_gamma: float = 0.99
    l_polyak: float = 0.995
    l_entropy_coefficient: float = 0.1
    l_tune_entropy_coefficient: bool = True
    l_target_entropy = None
    l_clip_actor_gradients: bool = False
    l_exploration_noise: float = 0.2

    # datasets
    file_location: str = "expert_data"
    prior_file_location: str = "prior_data"
    se_name: str = "XArm6Reach"               # B^SE dir under file_location
    sn_name: str = "XArm6Reach_random"        # B^SN dir under prior_file_location
    tn_name: str = "XArm6Reach_real_random"   # B^TN dir under prior_file_location
    n_demos: int = 10000


def build_d3il(cfg: D3ilConfig):
    """Construct (model, agent_buffer, l_agent, se_buffer, sn_buffer, tn_buffer).

    Mirrors run_d3il: make_* factories, se/sn/tn DemonstrationsReplayBuffers, SAC
    agent, the D3ILModelwithPolicy orchestrator, and the reward-computing
    CustomReplayBuffer (B^TL). gymnasium-free (no env)."""
    import numpy as np
    import tensorflow as tf

    # --- D3IL repo modules (must be importable on the server) ---
    # NOTE: import paths mirror run_d3il EXACTLY. In the D3IL repo, Critic lives in
    # td3_models (NOT sac_models, unlike the DIFFIL repo).
    from sac_models import StochasticActor, SAC
    from td3_models import Critic
    from buffers import DemonstrationsReplayBuffer
    from utils import load_expert_trajectories
    from custom_code.run_imitation.d3il import (
        Encoder, Generator, InvariantDiscriminator, TranslatedImageDiscriminator,
        ExpertFeatureDiscriminator, CustomReplayBuffer, D3ILModelwithPolicy)

    # ---- fixed datasets (Phase-1 + reward) ----
    se_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        cfg.se_name, cfg.file_location, visual_data=True, load_ids=True, max_demos=cfg.n_demos))     # B^SE
    sn_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        cfg.sn_name, cfg.prior_file_location, visual_data=True, load_ids=True, max_demos=cfg.n_demos))  # B^SN
    tn_buffer = DemonstrationsReplayBuffer(load_expert_trajectories(
        cfg.tn_name, cfg.prior_file_location, visual_data=True, load_ids=True, max_demos=cfg.n_demos))  # B^TN

    expert_visual_data_shape = se_buffer.get_random_batch(1)["ims"][0].shape
    past_frames = int(expert_visual_data_shape[0])

    # ---- image-translation / feature model architecture (== run_d3il) ----
    im_side = cfg.im_side
    im_shape4 = [im_side, im_side, 3 * past_frames]
    im_shape1 = [im_side, im_side, 3]
    enc_e_filters = [16, 16, 32, 32, 64, 64]
    enc_d_filters = [16, 16, 32, 32, 64, 64]
    gen_filters = [64, 64, 32, 32, 16, 16, 3 * past_frames]
    dom_disc_hidden_units = [32, 32]
    cls_disc_hidden_units = [32, 32]
    trans_disc_hidden_units = [16, 16, 32, 32, 64, 64]
    expert_disc_hidden_units = [100, 100]
    enc_d_final_kernel_size = im_side // 4

    def make_encoder_d():
        layers = [tf.keras.layers.Reshape(im_shape1)]
        for i, f in enumerate(enc_d_filters, start=1):
            s = 2 if (i > 2 and i % 2 == 1) else 1
            layers += [tf.keras.layers.Conv2D(f, 3, strides=s, activation="relu", padding="same")]
        layers += [tf.keras.layers.AveragePooling2D(enc_d_final_kernel_size),
                   tf.keras.layers.Reshape([-1]), tf.keras.layers.Dense(8)]
        return Encoder(layers)

    def make_encoder_e():
        layers = [tf.keras.layers.Reshape(im_shape4)]
        for i, f in enumerate(enc_e_filters, start=1):
            s = 2 if (i > 2 and i % 2 == 1) else 1
            layers += [tf.keras.layers.Conv2D(f, 3, strides=s, activation="relu", padding="same")]
        layers += [tf.keras.layers.Reshape([-1])]
        return Encoder(layers)

    def make_generator():
        layers = []
        for i, f in enumerate(gen_filters[:-1], start=1):
            s = 2 if (i > 2 and i % 2 == 1) else 1
            layers += [tf.keras.layers.Conv2DTranspose(f, 3, strides=s, activation="relu", padding="same")]
        layers += [tf.keras.layers.Conv2DTranspose(gen_filters[-1], 1, padding="same")]
        return Generator(layers, past_frames, n_input_channels=enc_e_filters[-1])

    def make_dom_disc():
        layers = [tf.keras.layers.Dense(u, activation="relu") for u in dom_disc_hidden_units]
        layers.append(tf.keras.layers.Dense(1))
        return InvariantDiscriminator(layers, stab_const=1e-7)

    def make_cls_disc():
        layers = [tf.keras.layers.Dense(u, activation="relu") for u in cls_disc_hidden_units]
        layers.append(tf.keras.layers.Dense(1))
        return InvariantDiscriminator(layers, stab_const=1e-7)

    def make_trans_disc():
        layers = [tf.keras.layers.Reshape(im_shape4)]
        for i, f in enumerate(trans_disc_hidden_units, start=1):
            s = 2 if (i > 2 and i % 2 == 1) else 1
            layers += [tf.keras.layers.Conv2D(f, 3, strides=s, activation="relu", padding="same")]
        layers += [tf.keras.layers.Reshape([-1]), tf.keras.layers.Dense(1)]
        return TranslatedImageDiscriminator(layers, stab_const=1e-7)

    def make_expert_disc():
        layers = [tf.keras.layers.Dense(u, activation="relu") for u in expert_disc_hidden_units]
        layers.append(tf.keras.layers.Dense(1))
        return ExpertFeatureDiscriminator(layers, stab_const=1e-7)

    # ---- SAC learner (identical architecture to DIFFIL; exportable to NumpyActor) ----
    action_size = int(cfg.action_dim)
    target_entropy = cfg.l_target_entropy
    if target_entropy is None:
        target_entropy = -1.0 * float(action_size)

    def make_actor():
        return StochasticActor([tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
                                tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
                                tf.keras.layers.Dense(action_size * 2,
                                                      kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])

    def make_critic():
        return Critic([tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
                       tf.keras.layers.Dense(256, "relu", kernel_initializer="orthogonal"),
                       tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Orthogonal(0.01))])

    l_opt = tf.keras.optimizers.Adam(cfg.l_learning_rate)
    l_agent = SAC(make_actor=make_actor, make_critic=make_critic, make_critic2=make_critic,
                  actor_optimizer=l_opt, critic_optimizer=l_opt, gamma=cfg.l_gamma,
                  polyak=cfg.l_polyak, entropy_coefficient=cfg.l_entropy_coefficient,
                  tune_entropy_coefficient=cfg.l_tune_entropy_coefficient,
                  target_entropy=target_entropy, clip_actor_gradients=cfg.l_clip_actor_gradients)

    # ---- D3IL orchestrator (ver_train_policy) ----
    model = D3ILModelwithPolicy(l_agent, make_encoder_d, make_encoder_e, make_generator,
                                make_dom_disc, make_cls_disc, make_trans_disc, make_expert_disc,
                                cfg.c_gan_trans, cfg.c_gan_feat, cfg.c_recon, cfg.c_cycle,
                                cfg.c_feat_mean, cfg.c_feat_recon, cfg.c_feat_reg, cfg.c_feat_cycle,
                                cfg.c_norm_de, cfg.c_norm_be, cfg.type_recon_loss, cfg.eg_update_interval,
                                cfg.it_max_grad_norm, cfg.it_lr, cfg.d_rew, cfg.d_max_grad_norm,
                                cfg.d_learning_rate, past_frames=past_frames)

    # ---- build the graph (env.reset() replaced by a zero state vector) ----
    zero_state = np.zeros((1, cfg.obs_dim), np.float32)
    model(model.reshape_input_images(se_buffer.get_random_batch(1)["ims"]),
          model.reshape_input_images(sn_buffer.get_random_batch(1)["ims"]),
          model.reshape_input_images(tn_buffer.get_random_batch(1)["ims"]),
          model.reshape_input_images(tn_buffer.get_random_batch(1)["ims"]),   # tl slot = tn (no learner yet)
          zero_state)

    # ---- online target buffer B^TL (recomputes reward via model.get_reward) ----
    agent_buffer = CustomReplayBuffer(model, cfg.l_buffer_size)

    # ---- force-build the SAC actor so it can be exported before training ----
    l_agent._act.get_action(tf.zeros([1, cfg.obs_dim], tf.float32), 0.0)

    return model, agent_buffer, l_agent, se_buffer, sn_buffer, tn_buffer


# ---------------------------------------------------------------------------
# Phase helpers (keep d3il_learner_node.py thin)
# ---------------------------------------------------------------------------
def pretrain_image_translation(model, se_buffer, sn_buffer, tn_buffer, cfg,
                               epochs: int, log_interval: int = 100):
    """Phase 1 (OFFLINE): train the image-translation/feature model on B^SE/B^SN/B^TN.
    tl images are drawn from B^TN (no learner rollouts exist yet)."""
    for e in range(epochs):
        se = se_buffer.get_random_batch(cfg.it_batch_size)["ims"]
        sn = sn_buffer.get_random_batch(cfg.it_batch_size)["ims"]
        tn = tn_buffer.get_random_batch(cfg.it_batch_size)["ims"]
        tl = tn_buffer.get_random_batch(cfg.it_batch_size)["ims"]
        model.train_image_translation(se, sn, tn, tl, e)
        if e == 0 or (e + 1) % log_interval == 0 or (e + 1) == epochs:
            print(f"[d3il/pretrain] epoch {e + 1}/{epochs}")


def policy_train_step(model, se_buffer, sn_buffer, tn_buffer, agent_buffer, cfg, n_new: int):
    """Phase 2 (ONLINE): one D3IL training round on newly added O^TL.
    it_updates=0 keeps the feature model FROZEN (only expert disc + SAC update)."""
    model.train(se_buffer=se_buffer, sn_buffer=sn_buffer, tn_buffer=tn_buffer,
                agent_buffer=agent_buffer,
                l_batch_size=cfg.l_batch_size,
                l_updates=int(cfg.l_updates_per_step * n_new),
                l_act_delay=cfg.l_act_delay,
                d_updates=max(1, int(cfg.d_updates_per_step * n_new)),
                d_batch_size=cfg.d_batch_size,
                it_updates=0, it_batch_size=cfg.it_batch_size,
                epoch=0, pretrain_epochs=0, nn_updates=0, step_counter=0, save_final_path=None)
