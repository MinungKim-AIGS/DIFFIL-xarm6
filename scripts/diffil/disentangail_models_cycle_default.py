import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import sklearn
import matplotlib.pyplot as plt
import random
import datetime
import umap
import os
# import warnings filter
from warnings import simplefilter
import time
import wandb
from sklearn.cluster import KMeans
from collections import Counter
import gc
from PIL import Image
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

class DisentanGAIL(tf.keras.Model):

    def __init__(self,
                 agent,
                 make_recon,
                 make_preprocessing,
                 make_label,
                 make_label_frame,
                 make_fwgan,
                 make_fake_fwgan,
                 make_feature_gen,
                 expert_buffer,
                 log_dir,
                 run_wandb,
                 prior_expert_buffer=None,
                 prior_agent_buffer=None,
                 past_frames=4,
                 im_shape=32,
                 feature_size=10,
                 recon_loss = 1,
                 feature_fake_loss = 1,
                 disc_loss = 50,
                 gen_loss = 10,
                 label_loss_se = 10,
                 label_loss_sr = 10,
                 label_loss_tl = 10,
                 label_tl_relabel = 10,
                 label_loss_tr = 10,
                 percentage = 10,
                 sehat = 100,
                 pol_update = 1000,
                 sampler = None,
                 epi_limit = 200,
                 random_epi = 200):

        super(DisentanGAIL, self).__init__()
        self._recon_layer_s, self._recon_layer_t = make_recon()
        self._pre_s = make_preprocessing()
        self._fake_fwgan = make_fake_fwgan()
        self._fake_gen = make_feature_gen()
        self._feature_wgan = make_fwgan()
        self._label_net = make_label()
        self._label_net_frame = make_label_frame()

        self._epi_len = epi_limit
        self._random_epi_len = random_epi
        self._policy_update = pol_update
        self.int_cnt = 0
        self._tr_cnt = 0
        self.agent = agent
        self._past_frames = past_frames
        self._exp_buff = expert_buffer
        self._log_dir = log_dir

        self.init = True
        self._im_shape = im_shape[0]
        self.frozen_se = []
        self.se_sample_prob = []
        self.se_sample_sehat_prob = []

        self.frozen_source = []
        self.source_prob = []
        self.expert_len = 0
        self.frozen_tl = []
        self.tl_sample_prob = []
        self.tl_sample_sehat_prob = []

        self.frozen_target = []
        self.target_prob = []

        self.sampler = sampler
        # self.source_img = []
        # self.target_img = []
        # self.source_init_img = []
        # self.target_ini_img = []

        # with tf.device('/CPU:0'):
        self._frozen_net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                                   input_shape=(
                                       self._im_shape, self._im_shape, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                                   strides=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=8)
        ])

        self._run_wandb = run_wandb
        self._pr_exp_buff = prior_expert_buffer
        self._pr_age_buff = prior_agent_buffer

        self._update_target = True
        self.feature_size = feature_size

        self.cnt = 0

        self.discriminator_optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.discriminator_optimizer_fake = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.recon_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.next_state_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.mse_loss_batch = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.recon_loss = recon_loss
        self.feature_fake_loss = feature_fake_loss
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss
        self.label_loss_se = label_loss_se
        self.label_loss_sr = label_loss_sr
        self.label_loss_tl = label_loss_tl
        self.label_tl_relabel = label_tl_relabel
        self.percentage = percentage

        self.label_loss_tr = label_loss_tr
        self.sehat = sehat

        self.nstep_list = []
        self._label_se = []
        self._label_tl = []
        self._label_source = []
        self._label_target = []


        self._batch_size = 128
        print("Initialize complete. it should be one-time called.")

    def call(self, inputs):
        out = inputs.get_random_batch(1, False)['ims']
        pre_s = self._pre_s(out)
        recon_s = self._recon_layer_s(pre_s)
        recon_t = self._recon_layer_t(pre_s)
        label = self._label_net(pre_s)

        featurewgan = self._feature_wgan(pre_s)
        fakegen = self._fake_gen(pre_s)
        fakewgan = self._fake_fwgan(fakegen)
        frozen_net = self._frozen_net(out[:, 1])

        self.agent.train(inputs, 1, 1, 1)
        return out

    @staticmethod
    def _ce_gan_loss_custom(labels, probs, lb):
        return tf.losses.binary_crossentropy(labels, probs, label_smoothing=lb)

    @staticmethod
    def _ce_gan_loss(l_disc_prob, e_disc_prob, lb):
        labels = tf.concat([tf.ones_like(l_disc_prob),
                            tf.zeros_like(e_disc_prob)], axis=0)
        probs = tf.concat([l_disc_prob, e_disc_prob], axis=0)
        return tf.losses.binary_crossentropy(labels, probs, label_smoothing=lb)

    def training_fakefwgan(self, data_combine):
        # ================================================================

        LAMBDA = 10
        batch = self._batch_size

        epsilon = tf.random.uniform(shape=[6*batch,1], minval=0, maxval=1, dtype='float32')

        feature = self._pre_s(data_combine)

        feature_gen = self._fake_gen(feature)

        batch_indices = tf.range(batch)
        shuffled_indices_s = tf.random.shuffle(batch_indices)
        shuffled_indices_t = tf.random.shuffle(batch_indices)

        feature_source = tf.concat([feature[:3*batch],feature[:3*batch]],axis=0)
        feature_target = feature_gen

        with tf.GradientTape() as d1_tape:

            interp = epsilon * feature_source + (1 - epsilon) * feature_target

            with tf.GradientTape() as tape:
                tape.watch(interp)
                predictions = self._fake_fwgan(interp)

                gradients = tape.gradient(predictions, [interp])[0]
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-9)
                gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

            true_preds = self._fake_fwgan(feature_source)
            fake_preds = self._fake_fwgan(feature_target)

            true_disc = true_preds
            fake_disc = fake_preds

            discriminator_loss = -true_disc + fake_disc
            disc_loss = 50*(discriminator_loss) + LAMBDA * gradient_penalty


        d1_gradients = d1_tape.gradient(disc_loss, self._fake_fwgan.trainable_variables)
        del d1_tape
        self.discriminator_optimizer_fake.apply_gradients(zip(d1_gradients, self._fake_fwgan.trainable_variables))

        return disc_loss,grad_norms,gradient_penalty

    def training_featurewgan(self, data_combine):

        LAMBDA = 10
        batch = self._batch_size

        epsilon = tf.random.uniform(shape=[2*batch,1], minval=0, maxval=1, dtype='float32')

        feature = self._pre_s(data_combine)

        feature_source = feature[:2 * batch, ]
        feature_target = feature[2 * batch:, ]

    #===================여기서부터 학습=====================================
        with tf.GradientTape() as d1_tape:

            interp = epsilon * feature_source + (1 - epsilon) * feature_target

            with tf.GradientTape() as tape:
                tape.watch(interp)
                predictions = self._feature_wgan(interp)

                gradients = tape.gradient(predictions, [interp])[0]
                grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1e-9)
                gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))

            true_preds = self._feature_wgan(feature_source)
            fake_preds = self._feature_wgan(feature_target)

            true_disc = tf.reduce_mean(true_preds)
            fake_disc = tf.reduce_mean(fake_preds)

            discriminator_loss = -true_disc + fake_disc
            disc_loss = self.disc_loss*(discriminator_loss) + LAMBDA * gradient_penalty

            true_preds_single, true_preds_seq = self._feature_wgan.chk_loss(feature_source)
            fake_preds_single, fake_preds_seq = self._feature_wgan.chk_loss(feature_target)

            disc_loss_single = -true_preds_single + fake_preds_single
            disc_loss_seq = -true_preds_seq + fake_preds_seq

        d1_gradients = d1_tape.gradient(disc_loss, self._feature_wgan.trainable_variables)
        del d1_tape
        self.discriminator_optimizer1.apply_gradients(zip(d1_gradients, self._feature_wgan.trainable_variables))

        return disc_loss,grad_norms,gradient_penalty, disc_loss_single, disc_loss_seq

    def SA_training(self, data_combine, dense, dense_timesteps):

        batch = self._batch_size

        timestep_double =  np.array(dense_timesteps[:batch], dtype=np.float64)
        timestep_double_reshape = timestep_double.reshape(batch, 1)
        modif_label = timestep_double_reshape*tf.ones([batch, 1])
        recon_True_label = np.concatenate([tf.ones([batch, 1]), tf.zeros([5*batch, 1])],axis=0)
        recon_True_label_frame = np.concatenate([modif_label, tf.zeros([3*batch, 1])],axis=0)

        with tf.GradientTape() as g_tape, tf.GradientTape() as gen_tape:

            feature = self._pre_s(data_combine)
            feature_dense = self._pre_s(dense)

            recon_by_target = self._recon_layer_t(feature)
            recon_by_source = self._recon_layer_s(feature)

            recon_by_target_sg = self._recon_layer_t(tf.stop_gradient(feature))
            recon_by_source_sg = self._recon_layer_s(tf.stop_gradient(feature))

            feature_recon_by_source_sg = self._pre_s(recon_by_source_sg)
            feature_recon_by_target_sg = self._pre_s(recon_by_target_sg)

            feature_recon_loss = self.mse_loss_batch(tf.stop_gradient(feature), tf.concat([feature_recon_by_source_sg[:2*batch],feature_recon_by_target_sg[2*batch:]],axis=0))

            feature_fake_loss = self.mse_loss_batch(tf.stop_gradient(feature), tf.concat([feature_recon_by_target_sg[:2*batch],feature_recon_by_source_sg[2*batch:]],axis=0))

            feature_from_target_source = self._recon_layer_s(feature_dense[2*batch:])
            feature_fake_from_target = self._pre_s(tf.stop_gradient(feature_from_target_source))

            feature_bundle = tf.concat([feature_dense[:batch], feature_dense[batch:2*batch],feature_dense[2*batch:], feature_fake_from_target ],axis=0)
            recon_sehat_label = self._label_net(feature_bundle)

            recon_loss = self.mse_loss_batch(data_combine, tf.concat([recon_by_source[:2*self._batch_size], recon_by_target[2*self._batch_size:]],axis=0))
            recon_label_loss = self._ce_gan_loss_custom(recon_True_label,recon_sehat_label, 0)

            frame_label = self._label_net_frame(tf.stop_gradient(feature_dense))
            frame_label_loss = self._ce_gan_loss_custom(recon_True_label_frame, frame_label, 0)

            total_fwgan = self._feature_wgan(feature)

            source_preds_feat = total_fwgan[:2 * batch]

            target_preds_feat = total_fwgan[2 * batch:]

            source_sum = tf.reduce_mean(feature[:2*batch])
            target_sum = tf.reduce_mean(feature[2*batch:])

            se_dist = self.mse_loss_batch(feature[:batch],target_sum)
            sr_dist = self.mse_loss_batch(feature[batch:2*batch], target_sum)
            tl_dist = self.mse_loss_batch(feature[2*batch:3*batch], source_sum)
            tr_dist = self.mse_loss_batch(feature[3*batch:], source_sum)

            true_disc = tf.reduce_mean(source_preds_feat)
            fake_disc = tf.reduce_mean(target_preds_feat)
            recon_scale = 65536 * self.recon_loss
            cycle_scale = self.feature_fake_loss
            generator_scale = self.gen_loss
            dom_div = 2 * self._batch_size

            # total_recon_loss = tf.reduce_sum(recon_loss)
            total_recon_loss = tf.reduce_mean(recon_loss)
            cyclic_loss = tf.reduce_mean(feature_fake_loss)
            gen_loss = true_disc - fake_disc

            label_loss_combine = recon_label_loss

            # if use_sehat:
            #     label_loss = self.label_loss_se *tf.reduce_mean(label_loss_combine[:self._batch_size])\
            #                  + self.label_loss_sr *tf.reduce_mean(label_loss_combine[self._batch_size:2 * self._batch_size]) \
            #                  + self.label_loss_tr *tf.reduce_mean(label_loss_combine[3 * self._batch_size:4*self._batch_size])\
            #                  + self.label_tl_relabel * tf.reduce_mean(recon_label_loss_expert_like)\
            #                  + self.label_tl_relabel * tf.reduce_mean(recon_label_loss_no_expert)
            # else:

            label_loss = self.label_loss_se *tf.reduce_mean(label_loss_combine[:self._batch_size])\
                         + self.label_loss_sr *tf.reduce_mean(label_loss_combine[self._batch_size:2 * self._batch_size]) \
                         + self.label_loss_tr *tf.reduce_mean(label_loss_combine[3 * self._batch_size:4*self._batch_size])\
                         + self.label_loss_tl *tf.reduce_mean(label_loss_combine[2 * self._batch_size:3*self._batch_size])\
                         + self.label_loss_se *tf.reduce_mean(frame_label_loss[:self._batch_size])\
                         + self.label_loss_sr *tf.reduce_mean(frame_label_loss[self._batch_size:2 * self._batch_size])

            g_loss = recon_scale *total_recon_loss + cycle_scale * cyclic_loss + generator_scale * gen_loss + label_loss

        g_gradients_next = g_tape.gradient(g_loss,
                                           self._pre_s.trainable_variables
                                           + self._recon_layer_s.trainable_variables + self._recon_layer_t.trainable_variables
                                           + self._label_net.trainable_variables + self._label_net_frame.trainable_variables)

        del g_tape

        self.next_state_optimizer.apply_gradients(
            zip(g_gradients_next, self._pre_s.trainable_variables
                + self._recon_layer_s.trainable_variables + self._recon_layer_t.trainable_variables+ self._label_net.trainable_variables+ self._label_net_frame.trainable_variables))


        return 65536*recon_loss,feature_recon_loss,feature_fake_loss, recon_label_loss, \
                gen_loss, se_dist, sr_dist, tl_dist, tr_dist, frame_label_loss

    def plot_subplot(self, ax, data, title, description=None, xlabel=None):
        ax.imshow(data)
        if description:
            # ax.set_title('Normalized occupied \n Neighbors')
            ax.set_title(title, fontsize=9)
            ax.text(-1.0, 1.5, description, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=9)

        else:
            ax.set_title(title, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.axis("off")

    def plot_subplot_cluster(self, ax, data, title, description=None, xlabel=None):
        ax.imshow(data)
        if description:
            # ax.set_title('Normalized occupied \n Neighbors')
            ax.set_title(title, fontsize=9)
            ax.text(-1.0, 1.0, description, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=9)

        else:
            ax.set_title(title, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.axis("off")

    def get_10000_feature(self, my_buffer, domain, task):

        # if domain:
        #     buffer_index = random.randint(0, 39)
        #
        # else:
        #     buffer_index = random.randint(0, 15)
        if task == 'tl':
            tl_10000_data = my_buffer.get_random_batch(5000, re_eval_rw = False)
        else:
            tl_10000_data = my_buffer.get_random_batch(5000)
        tl_img = tl_10000_data['ims']

        x = 500
        total_data = len(tl_img)

        tl_feature_array = []
        recon_feature_array = []
        fake_feature_array = []

        tl_feature_sehat_array = []
        recon_feature_array = []
        fake_feature_sehat_array = []
        recon_feature_sehat_array = []
        for i in range(0, total_data, x):

            tl_img_chunk = tl_img[i:i + x]
            tl_feature = self._pre_s.get_state_feature_plot(tl_img_chunk)

            if domain:
                tl_1000_recon= self._recon_layer_t(tl_feature)
                tl_1000_recon_fake = self._recon_layer_s(tl_feature)

                recon_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon)
                fake_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon_fake)

            else:
                tl_1000_recon = self._recon_layer_s(tl_feature)
                tl_1000_recon_fake = self._recon_layer_t(tl_feature)

                recon_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon)
                fake_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon_fake)

            tl_feature_array.append(tl_feature)
            recon_feature_array.append(recon_tl_1000_feature)
            fake_feature_array.append(fake_tl_1000_feature)

        tl_feature = np.concatenate(tl_feature_array, axis=0)
        recon_tl_1000_feature = np.concatenate(recon_feature_array, axis=0)
        fake_tl_1000_feature = np.concatenate(fake_feature_array, axis=0)

        return tl_img, tl_feature, fake_tl_1000_feature, recon_tl_1000_feature


    def plot_cluster_samplee(self, anchor, pos1,pos2, neg1,neg2, task):
        Anchor_feature = self._pre_s(anchor)
        pos1_data = self._pre_s(pos1)
        pos2_data = self._pre_s(pos2)
        neg1_data = self._pre_s(neg1)
        neg2_data = self._pre_s(neg2)

        anchor_label = self._label_net(Anchor_feature)
        pos1_label = self._label_net(pos1_data)
        pos2_label = self._label_net(pos2_data)
        neg1_label = self._label_net(neg1_data)
        neg2_label = self._label_net(neg2_data)

        diff_feature_pos1 = self.mse_loss_batch(Anchor_feature, pos1_label)
        diff_feature_pos2 = self.mse_loss_batch(Anchor_feature, pos2_label)

        diff_feature_neg1 = self.mse_loss_batch(Anchor_feature, neg1_label)
        diff_feature_neg2 = self.mse_loss_batch(Anchor_feature, neg2_label)

        save_tl_dir = self._log_dir + '/samples/cluster_samples/{}_In_{}'.format(task,self.cnt)
        os.makedirs(save_tl_dir)

        for i in range(10):

            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)
            for j in range(3):
                if j ==0:
                    description = "Label[Expert=1]"
                else:
                    description = "Anchor_feature_diff\n Label[Expert=1]"

                for k in range(4):
                    index = i
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}'.format(anchor_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, anchor[index][3-k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, anchor[index][3-k, :], title=None)
                    elif j ==1:
                        title = '{:.5e}\n{}'.format(diff_feature_pos1[index],pos1_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, pos1[index][3-k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, pos1[index][3-k, :], title=None)
                    else:
                        title = '{:.5e}\n{}'.format(diff_feature_pos2[index],pos2_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, pos2[index][3-k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, pos2[index][3-k, :], title=None)

            plt.savefig('{}/pos_{}_{}'.format(save_tl_dir, task, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "cluster_samples/{}_epoch/cluster_pos_{}_{}".format(self.cnt, task,str(i).zfill(3)): [wandb.Image(
                    '{}/pos_{}_{}'.format(save_tl_dir, task, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        for i in range(10):

            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)
            for j in range(3):
                if j == 0:
                    description = "Label[Expert=1]"
                else:
                    description = "Anchor_feature_diff\n Label[Expert=1]"

                for k in range(4):
                    index = i
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}'.format(anchor_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, anchor[index][3 - k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, anchor[index][3 - k, :], title=None)
                    elif j == 1:
                        title = '{:.5e}\n{}'.format(diff_feature_neg1[index], neg1_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, neg1[index][3 - k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, neg1[index][3 - k, :], title=None)
                    else:
                        title = '{:.5e}\n{}'.format(diff_feature_neg2[index], neg2_label[index])
                        if k == 0:
                            self.plot_subplot_cluster(ax, neg2[index][3 - k, :], title, description)
                        else:
                            self.plot_subplot_cluster(ax, neg2[index][3 - k, :], title=None)

            plt.savefig('{}/neg_{}_{}'.format(save_tl_dir, task, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "cluster_samples/{}_epoch/cluster_neg_{}_{}".format(self.cnt, task, str(i).zfill(3)): [wandb.Image(
                    '{}/neg_{}_{}'.format(save_tl_dir, task, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        return 0


    def plot_genfeature(self, tl_buffer, se_buffer, tr_buffer, sr_buffer):


        buffer_index = random.randint(0, 9)
        # buffer_index = random.randint(40, 48)

        sr_1000_data, sr_1000_data_next = sr_buffer.get_det_batch_index_double(1000, buffer_index * 1000)
        sr_img = sr_1000_data['ims']

        tr_1000_data, tr_1000_data_next = tr_buffer.get_det_batch_index_double(1000, buffer_index * 1000)
        tr_img = tr_1000_data['ims']

        tl_1000_data, tl_1000_data_next = tl_buffer.get_det_batch_index_double(1000, buffer_index * 1000)
        tl_img = tl_1000_data['ims']

        se_1000_data, se_1000_data_next = se_buffer.get_det_batch_index_double(1000, buffer_index * 1000)
        se_img = se_1000_data['ims']

        x = 100
        total_data = len(tl_img)

        recon_img_array = []
        cycle_img_array = []
        fake_img_array = []
        true_label_array = []
        fake_label_array = []
        fake_label_array_from_source = []
        feature_true_array = []
        feature_fake_source_array = []
        feature_true_array_comp = []

        recon_img_array_se = []
        recon_img_tl_fake = []
        fake_img_array_se = []
        true_label_se_array = []
        fake_label_se_array = []
        feature_true_array_se = []

        recon_img_array_tr = []
        fake_img_array_tr = []
        true_label_array_tr = []
        fake_label_array_tr = []
        feature_true_array_tr = []

        feature_sehat_flip_array_se = []
        feature_sehat_flip_array_sr = []
        feature_sehat_flip_array_tl = []
        feature_sehat_flip_array_tr = []

        feature_true_flip_array_se = []
        feature_true_flip_array_sr = []
        feature_true_flip_array_tl = []
        feature_true_flip_array_tr = []

        recon_img_array_sr = []
        fake_img_array_sr = []
        true_label_array_sr = []
        fake_label_array_sr = []
        feature_true_array_sr = []

        feat_diff_array = []
        true_sehat_fwgan_array = []
        true_sehat_label_array = []

        se_true_fwgan_array = []
        se_tl_img_array = []
        se_tl_label_array = []
        se_tl_fwgan_array = []

        for i in range(0, total_data, x):
            tl_img_chunk = tl_img[i:i + x]
            se_img_chunk = se_img[i:i + x]
            tr_img_chunk = tr_img[i:i + x]
            sr_img_chunk = sr_img[i:i + x]

            tl_feature_org = self._pre_s.get_state_feature_plot(tl_img_chunk)
            source_from_tl = self._recon_layer_s(tl_feature_org)

            source_feature_from_tl = self._pre_s.get_state_feature_plot(source_from_tl)
            true_sourcehat_fwgan = self._fake_fwgan(source_feature_from_tl)
            true_sourcehat_label = self._label_net(source_feature_from_tl)

            tl_feature = self._fake_gen(tl_feature_org)
            source_feature = self._fake_gen(source_feature_from_tl)

            tl_flip_feature_org = np.concatenate([tl_feature_org[:, 3 * self.feature_size:],
                                              tl_feature_org[:, 2 * self.feature_size:3 * self.feature_size],
                                              tl_feature_org[:, self.feature_size:2 * self.feature_size],
                                              tl_feature_org[:, :self.feature_size]], axis=1)

            tl_flip_feature = np.concatenate([tl_feature[:, 3 * self.feature_size:],
                                              tl_feature[:, 2 * self.feature_size:3 * self.feature_size],
                                              tl_feature[:, self.feature_size:2 * self.feature_size],
                                              tl_feature[:, :self.feature_size]], axis=1)

            hat_preds = self._fake_fwgan(tl_feature)
            hat_preds_true = self._fake_fwgan(tl_feature_org)
            diff_feature = hat_preds_true
            # diff_feature = self.mse_loss_batch(hat_preds_true, hat_preds)
            fake_source_preds = self._fake_fwgan(source_feature)
            # diff_feature = self.mse_loss_batch(tl_feature_org, tl_feature)

            tl_gen_label = self._label_net(tl_feature)
            tl_true_label = self._label_net(tl_feature_org)
            tl_flip_label = self._label_net(tl_flip_feature)
            tl_true_flip_label = self._label_net(tl_flip_feature_org)
            tl_gen_label_from_source = self._label_net(source_feature)

            se_feature_org = self._pre_s.get_state_feature_plot(se_img_chunk)
            se_feature = self._fake_gen(se_feature_org)
            hat_preds_se = self._fake_fwgan(se_feature)

            se_flip_feature = np.concatenate([se_feature[:, 3 * self.feature_size:],
                                              se_feature[:, 2 * self.feature_size:3 * self.feature_size],
                                              se_feature[:, self.feature_size:2 * self.feature_size],
                                              se_feature[:, :self.feature_size]], axis=1)

            se_flip_feature_org = np.concatenate([se_feature_org[:, 3 * self.feature_size:],
                                              se_feature_org[:, 2 * self.feature_size:3 * self.feature_size],
                                              se_feature_org[:, self.feature_size:2 * self.feature_size],
                                              se_feature_org[:, :self.feature_size]], axis=1)

            se_gen_label = self._label_net(se_feature)
            se_true_fwgan = self._fake_fwgan(se_feature_org)
            se_true_label = self._label_net(se_feature_org)
            se_flip_label = self._label_net(se_flip_feature)
            se_true_flip_label = self._label_net(se_flip_feature_org)

            se_tl_img = self._recon_layer_t(se_feature_org)
            se_tl_feature = self._pre_s.get_state_feature_plot(se_tl_img)
            se_tl_label = self._label_net(se_tl_feature)
            se_tl_fwgan = self._fake_fwgan(se_tl_feature)

            tr_feature_org = self._pre_s.get_state_feature_plot(tr_img_chunk)
            tr_feature = self._fake_gen(tr_feature_org)
            hat_preds_tr = self._fake_fwgan(tr_feature)
            tr_flip_feature = np.concatenate([tr_feature[:, 3 * self.feature_size:],
                                              tr_feature[:, 2 * self.feature_size:3 * self.feature_size],
                                              tr_feature[:, self.feature_size:2 * self.feature_size],
                                              tr_feature[:, :self.feature_size]], axis=1)

            tr_flip_feature_org = np.concatenate([tr_feature_org[:, 3 * self.feature_size:],
                                              tr_feature_org[:, 2 * self.feature_size:3 * self.feature_size],
                                              tr_feature_org[:, self.feature_size:2 * self.feature_size],
                                              tr_feature_org[:, :self.feature_size]], axis=1)

            tr_gen_label = self._label_net(tr_feature)
            tr_true_label = self._label_net(tr_feature_org)
            tr_flip_label = self._label_net(tr_flip_feature)
            tr_true_flip_label = self._label_net(tr_flip_feature_org)

            sr_feature_org = self._pre_s.get_state_feature_plot(sr_img_chunk)
            sr_feature = self._fake_gen(sr_feature_org)
            hat_preds_sr = self._fake_fwgan(sr_feature)
            sr_flip_feature = np.concatenate([sr_feature[:, 3 * self.feature_size:],
                                              sr_feature[:, 2 * self.feature_size:3 * self.feature_size],
                                              sr_feature[:, self.feature_size:2 * self.feature_size],
                                              sr_feature[:, :self.feature_size]], axis=1)

            sr_flip_feature_org = np.concatenate([sr_feature_org[:, 3 * self.feature_size:],
                                              sr_feature_org[:, 2 * self.feature_size:3 * self.feature_size],
                                              sr_feature_org[:, self.feature_size:2 * self.feature_size],
                                              sr_feature_org[:, :self.feature_size]], axis=1)
            sr_gen_label = self._label_net(sr_feature)
            sr_true_label = self._label_net(sr_feature_org)
            sr_flip_label = self._label_net(sr_flip_feature)
            sr_true_flip_label = self._label_net(sr_flip_feature_org)

            tl_1000_recon = self._recon_layer_t(tl_feature)
            source_tl_1000_recon = self._recon_layer_s(source_feature)

            tr_1000_recon = self._recon_layer_t(tr_feature)

            se_1000_recon = self._recon_layer_s(se_feature)

            sr_1000_recon = self._recon_layer_s(sr_feature)

            tl_1000_recon_fake = self._recon_layer_s(tl_feature)

            se_1000_recon_fake = self._recon_layer_t(se_feature)

            tr_1000_recon_fake = self._recon_layer_s(tr_feature)

            sr_1000_recon_fake = self._recon_layer_t(sr_feature)

            recon_img_array.append(tl_1000_recon)
            cycle_img_array.append(source_tl_1000_recon)
            fake_img_array.append(tl_1000_recon_fake)
            recon_img_tl_fake.append(source_from_tl)

            fake_label_array.append(tl_gen_label)
            fake_label_array_from_source.append(tl_gen_label_from_source)
            fake_label_se_array.append(se_gen_label)
            true_label_array.append(tl_true_label)
            true_label_se_array.append(se_true_label)

            true_sehat_fwgan_array.append(true_sourcehat_fwgan)
            true_sehat_label_array.append(true_sourcehat_label)

            recon_img_array_se.append(se_1000_recon)
            fake_img_array_se.append(se_1000_recon_fake)

            feature_true_array.append(hat_preds)
            feature_fake_source_array.append(fake_source_preds)
            feature_true_array_comp.append(hat_preds_true)
            feature_true_array_se.append(hat_preds_se)

            recon_img_array_tr.append(tr_1000_recon)
            fake_img_array_tr.append(tr_1000_recon_fake)
            recon_img_array_sr.append(sr_1000_recon)
            fake_img_array_sr.append(sr_1000_recon_fake)

            fake_label_array_tr.append(tr_gen_label)
            fake_label_array_sr.append(sr_gen_label)
            true_label_array_tr.append(tr_true_label)
            true_label_array_sr.append(sr_true_label)

            feature_true_array_tr.append(hat_preds_tr)
            feature_true_array_sr.append(hat_preds_sr)

            feat_diff_array.append(diff_feature)

            feature_sehat_flip_array_se.append(se_flip_label)
            feature_sehat_flip_array_sr.append(sr_flip_label)
            feature_sehat_flip_array_tl.append(tl_flip_label)
            feature_sehat_flip_array_tr.append(tr_flip_label)

            feature_true_flip_array_se.append(se_true_flip_label)
            feature_true_flip_array_sr.append(sr_true_flip_label)
            feature_true_flip_array_tl.append(tl_true_flip_label)
            feature_true_flip_array_tr.append(tr_true_flip_label)

            se_true_fwgan_array.append(se_true_fwgan)
            se_tl_img_array.append(se_tl_img)
            se_tl_label_array.append(se_tl_label)
            se_tl_fwgan_array.append(se_tl_fwgan)

        tl_1000_recon = np.concatenate(recon_img_array, axis=0)
        tl_1000_true_to_source = np.concatenate(recon_img_tl_fake, axis=0)
        tl_1000_recon_from_source = np.concatenate(cycle_img_array, axis=0)
        tl_1000_recon_fake = np.concatenate(fake_img_array, axis=0)

        tl_fake_label = np.concatenate(fake_label_array, axis=0)
        tl_fake_label_from_source = np.concatenate(fake_label_array_from_source, axis=0)
        se_fake_label = np.concatenate(fake_label_se_array, axis=0)

        tl_true_label = np.concatenate(true_label_array, axis=0)
        se_true_label = np.concatenate(true_label_se_array, axis=0)

        se_1000_recon = np.concatenate(recon_img_array_se, axis=0)
        se_1000_recon_fake = np.concatenate(fake_img_array_se, axis=0)

        true_sehat_fwgan = np.concatenate(true_sehat_fwgan_array, axis=0)
        true_sehat_label = np.concatenate(true_sehat_label_array, axis=0)

        tr_1000_recon = np.concatenate(recon_img_array_tr, axis=0)
        tr_1000_recon_fake = np.concatenate(fake_img_array_tr, axis=0)

        tr_fake_label = np.concatenate(fake_label_array_tr, axis=0)
        sr_fake_label = np.concatenate(fake_label_array_sr, axis=0)

        tr_true_label = np.concatenate(true_label_array_tr, axis=0)
        sr_true_label = np.concatenate(true_label_array_sr, axis=0)

        sr_1000_recon = np.concatenate(recon_img_array_sr, axis=0)
        sr_1000_recon_fake = np.concatenate(fake_img_array_sr, axis=0)

        feature_tl = np.concatenate(feature_true_array, axis=0)
        feature_fake_source = np.concatenate(feature_fake_source_array, axis=0)
        feature_tl_true = np.concatenate(feature_true_array_comp, axis=0)
        feature_se = np.concatenate(feature_true_array_se, axis=0)
        feature_tr = np.concatenate(feature_true_array_tr, axis=0)
        feature_sr = np.concatenate(feature_true_array_sr, axis=0)

        feature_flip_se = np.concatenate(feature_sehat_flip_array_se, axis=0)
        feature_flip_sr = np.concatenate(feature_sehat_flip_array_sr, axis=0)
        feature_flip_tl = np.concatenate(feature_sehat_flip_array_tl, axis=0)
        feature_flip_tr = np.concatenate(feature_sehat_flip_array_tr, axis=0)

        feature_flip_se_true = np.concatenate(feature_true_flip_array_se, axis=0)
        feature_flip_sr_true = np.concatenate(feature_true_flip_array_sr, axis=0)
        feature_flip_tl_true = np.concatenate(feature_true_flip_array_tl, axis=0)
        feature_flip_tr_true = np.concatenate(feature_true_flip_array_tr, axis=0)

        true_se_fwgan = np.concatenate(se_true_fwgan_array, axis=0)
        se_tl_img = np.concatenate(se_tl_img_array, axis=0)
        se_tl_label = np.concatenate(se_tl_label_array, axis=0)
        se_tl_fwgan = np.concatenate(se_tl_fwgan_array, axis=0)


        feature_difference = np.concatenate(feat_diff_array, axis=0)

        save_tl_dir = self._log_dir + '/samples/fake_generation/In_{}'.format(self.cnt)
        os.makedirs(save_tl_dir)

        plot_img = 5
        jump_idx = 40

        for i in range(plot_img):
            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)

            description = "fake_fwgan[positive= realistic]\nTrue_label[Expert = 1]\n sehat_label[Expert=1]\nflip_label[True]\nflip_label[sehat]"

            for j in range(3):
                for k in range(4):
                    index = jump_idx * i + k
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}\n{}\n{}\n{}\n{}'.format(feature_tl[index], tl_true_label[index],tl_fake_label[index],feature_flip_tl_true[index], feature_flip_tl[index])

                        if k == 0:
                            self.plot_subplot(ax, tl_img[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tl_img[index][3-k, :], title=None)

                    if j == 1:
                        title = 'Target_dom_recon'
                        if k == 0:
                            self.plot_subplot(ax, tl_1000_recon[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tl_1000_recon[index][3-k, :], title=None)

                    elif j == 2:
                        title = 'Source_dom_recon'
                        if k == 0:
                            self.plot_subplot(ax, tl_1000_recon_fake[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tl_1000_recon_fake[index][3-k, :], title=None)


            plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/fake_samples_tl_{}".format(self.cnt, str(i).zfill(3)): [wandb.Image(
                    '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        for i in range(plot_img):
            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)

            description = "fake_fwgan[positive= realistic]\nTrue_label[Expert = 1]\n sehat_label[Expert=1]\nflip_label[True]\nflip_label[sehat]"

            for j in range(3):
                for k in range(4):
                    index = jump_idx * i + k
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}\n{}\n{}\n{}\n{}'.format(feature_se[index], se_true_label[index],se_fake_label[index],feature_flip_se_true[index], feature_flip_se[index])

                        if k == 0:
                            self.plot_subplot(ax, se_img[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, se_img[index][3-k, :], title=None)

                    if j == 1:
                        title = 'source_dom_recon'

                        if k == 0:
                            self.plot_subplot(ax, se_1000_recon[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, se_1000_recon[index][3-k, :], title=None)

                    elif j == 2:
                        title = 'target_dom_recon'
                        if k == 0:
                            self.plot_subplot(ax, se_1000_recon_fake[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, se_1000_recon_fake[index][3-k, :], title=None)

            plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/fake_samples_se_{}".format(self.cnt, str(i).zfill(3)): [wandb.Image(
                    '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        for i in range(plot_img):
            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)

            description = "fake_fwgan[positive= realistic]\nTrue_label[Expert = 1]\n sehat_label[Expert=1]\nflip_label[True]\nflip_label[sehat]"

            for j in range(3):
                for k in range(4):
                    index = jump_idx * i + k
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}\n{}\n{}\n{}\n{}'.format(feature_tr[index], tr_true_label[index],tr_fake_label[index],feature_flip_tr_true[index], feature_flip_tr[index])

                        if k == 0:
                            self.plot_subplot(ax, tr_img[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tr_img[index][3-k, :], title=None)

                    if j == 1:
                        title = 'source_dom_recon'

                        if k == 0:
                            self.plot_subplot(ax, tr_1000_recon[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tr_1000_recon[index][3-k, :], title=None)

                    elif j == 2:
                        title = 'target_dom_recon'
                        if k == 0:
                            self.plot_subplot(ax, tr_1000_recon_fake[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, tr_1000_recon_fake[index][3-k, :], title=None)

            plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/fake_samples_tr_{}".format(self.cnt, str(i).zfill(3)): [wandb.Image(
                    '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        for i in range(plot_img):
            fig = plt.figure(figsize=(8, 6))
            plt.subplots_adjust(wspace=0.2, hspace=1.1)

            description = "fake_fwgan[positive= realistic]\nTrue_label[Expert = 1]\n sehat_label[Expert=1]\nflip_label[True]\nflip_label[sehat]"

            for j in range(3):
                for k in range(4):
                    index = jump_idx * i + k
                    ax = fig.add_subplot(3, 4, j * 4 + k + 1)
                    if j == 0:
                        title = '{}\n{}\n{}\n{}\n{}'.format(feature_sr[index], sr_true_label[index],sr_fake_label[index],feature_flip_sr_true[index], feature_flip_sr[index])

                        if k == 0:
                            self.plot_subplot(ax, sr_img[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, sr_img[index][3-k, :], title=None)

                    if j == 1:
                        title = 'source_dom_recon'

                        if k == 0:
                            self.plot_subplot(ax, sr_1000_recon[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, sr_1000_recon[index][3-k, :], title=None)

                    elif j == 2:
                        title = 'target_dom_recon'
                        if k == 0:
                            self.plot_subplot(ax, sr_1000_recon_fake[index][3-k, :], title, description)
                        else:
                            self.plot_subplot(ax, sr_1000_recon_fake[index][3-k, :], title=None)

            plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/fake_samples_sr_{}".format(self.cnt, str(i).zfill(3)): [wandb.Image(
                    '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

        source_data = np.concatenate([se_img[4:self._batch_size+4], sr_img[4:self._batch_size+4]],axis=0)
        source_data = np.stack([source_data[:, 3, :],
                        source_data[:, 2, :],
                        source_data[:, 1, :],
                        source_data[:, 0, :]], axis=1)
        source_flip = self._pre_s(source_data)

        fakewgan_trueflip = self._fake_fwgan(source_flip)
        fakewgan_trueflips = tf.squeeze(fakewgan_trueflip, axis=1)
        artsortdata = tf.argsort(fakewgan_trueflips, axis=0)

        flip_recon = self._recon_layer_s(source_flip)

        sehat_flip = self._fake_gen(source_flip)
        flip_label = self._label_net(source_flip)
        sehat_recon = self._recon_layer_s(sehat_flip)
        flip_label_sehat = self._label_net(sehat_flip)

        # diff_sample = self.mse_loss_batch(source_data, sehat_recon)

        # artsortdata = tf.argsort(diff_sample, axis=0).numpy().flatten()

        # half_sehat_flip = tf.gather(source_flip, artsortdata)

        # diff_fakewgan = np.round(feature_difference, decimals=4)
        # artsortdata = tf.argsort(diff_fakewgan, axis=0).numpy().flatten()

        top_10_indices = artsortdata[-10:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:10]  # 가장 작은 값의 인덱스

        top_10_data_org = tf.gather(flip_recon, top_10_indices).numpy()
        top_10_data = tf.gather(sehat_recon, top_10_indices).numpy()
        top_10_data_label = tf.gather(fakewgan_trueflip, top_10_indices).numpy()
        top_10_data_label_smooth = tf.gather(flip_label, top_10_indices).numpy()
        top_10_data_label_reward = tf.gather(flip_label_sehat, top_10_indices).numpy()

        bottom_10_data_org = tf.gather(flip_recon, top_10_indices).numpy()
        bottom_10_data = tf.gather(sehat_recon, bottom_10_indices).numpy()
        bottom_10_data_label = tf.gather(fakewgan_trueflip, bottom_10_indices).numpy()
        bottom_10_data_label_smooth = tf.gather(flip_label, bottom_10_indices).numpy()
        bottom_10_data_label_reward = tf.gather(flip_label_sehat, bottom_10_indices).numpy()

        plt.figure(figsize=(5, 40))

        for i in range(10):
            for j in range(4):
                plt.subplot(20, 4, 8*i +j + 1)
                plt.imshow(top_10_data_org[i][3-j, :])
                plt.subplot(20, 4, 8*i +j + 5)
                plt.imshow(top_10_data[i][3-j, :])
                if j==0:
                    plt.title('{}\n{}\n{}'.format(top_10_data_label[i],top_10_data_label_smooth[i], top_10_data_label_reward[i]))
            plt.axis('off')

        plt.savefig('{}_high'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
        self._run_wandb.log({
            "fakesamples/{}_epoch/fake_high_from_tl".format(self.cnt): [wandb.Image(
                '{}_high'.format(save_tl_dir) + '.png')]
        }, step=self._tr_cnt)

        plt.close()

        plt.figure(figsize=(5, 40))
        for i in range(10):
            for j in range(4):
                plt.subplot(20, 4, 8*i +j + 1)
                plt.imshow(bottom_10_data_org[i][3-j, :])
                plt.subplot(20, 4, 8*i +j + 5)
                plt.imshow(bottom_10_data[i][3-j, :])
                if j==0:
                    plt.title('{}\n{}\n{}'.format(bottom_10_data_label[i],bottom_10_data_label_smooth[i], bottom_10_data_label_reward[i]))
            plt.axis('off')

        plt.savefig('{}_low'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
        self._run_wandb.log({
            "fakesamples/{}_epoch/fake_low_from_tl".format(self.cnt): [wandb.Image(
                '{}_low'.format(save_tl_dir) + '.png')]
        }, step=self._tr_cnt)
        plt.close()

        artsortdata = tf.argsort(feature_difference, axis=0).numpy().flatten()
        # artsortdata = tf.argsort(feature_difference, axis=0)

        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스


        top_10_data_True_img = tf.gather(tl_img, top_10_indices).numpy()
        top_10_data_True_img_source = tf.gather(tl_1000_true_to_source, top_10_indices).numpy()
        top_10_data_true_sehat_fwgan = tf.gather(true_sehat_fwgan, top_10_indices).numpy()
        top_10_data_true_sehat_label = tf.gather(true_sehat_label, top_10_indices).numpy()

        top_10_data_True_fwgan = tf.gather(feature_tl_true, top_10_indices).numpy()
        top_10_data_True_label = tf.gather(tl_true_label, top_10_indices).numpy()

        top_10_data_sehat_target_img = tf.gather(tl_1000_recon, top_10_indices).numpy()
        top_10_data_sehat_target_fwgan = tf.gather(feature_tl, top_10_indices).numpy()
        top_10_data_sehat_target_label = tf.gather(tl_fake_label, top_10_indices).numpy()

        top_10_data_sehat_source_img = tf.gather(tl_1000_recon_fake, top_10_indices).numpy()
        top_10_data_sehat_source_fwgan = tf.gather(feature_tl, top_10_indices).numpy()
        top_10_data_sehat_source_label = tf.gather(tl_fake_label, top_10_indices).numpy()

        top_10_data_sehat_fake_source_img = tf.gather(tl_1000_recon_from_source, top_10_indices).numpy()
        top_10_data_sehat_fake_source_fwgan = tf.gather(feature_fake_source, top_10_indices).numpy()
        top_10_data_sehat_fake_source_label = tf.gather(tl_fake_label_from_source, top_10_indices).numpy()


        bot_10_data_True_img = tf.gather(tl_img, bottom_10_indices).numpy()
        bot_10_data_True_img_source = tf.gather(tl_1000_true_to_source, bottom_10_indices).numpy()
        bot_10_data_true_sehat_fwgan = tf.gather(true_sehat_fwgan, bottom_10_indices).numpy()
        bot_10_data_true_sehat_label = tf.gather(true_sehat_label, bottom_10_indices).numpy()

        bot_10_data_True_fwgan = tf.gather(feature_tl_true, bottom_10_indices).numpy()
        bot_10_data_True_label = tf.gather(tl_true_label, bottom_10_indices).numpy()

        bot_10_data_sehat_target_img = tf.gather(tl_1000_recon, bottom_10_indices).numpy()
        bot_10_data_sehat_target_fwgan = tf.gather(feature_tl, bottom_10_indices).numpy()
        bot_10_data_sehat_target_label = tf.gather(tl_fake_label, bottom_10_indices).numpy()

        bot_10_data_sehat_source_img = tf.gather(tl_1000_recon_fake, bottom_10_indices).numpy()
        bot_10_data_sehat_source_fwgan = tf.gather(feature_tl, bottom_10_indices).numpy()
        bot_10_data_sehat_source_label = tf.gather(tl_fake_label, bottom_10_indices).numpy()

        bot_10_data_sehat_fake_source_img = tf.gather(tl_1000_recon_from_source, bottom_10_indices).numpy()
        bot_10_data_sehat_fake_source_fwgan = tf.gather(feature_fake_source, bottom_10_indices).numpy()
        bot_10_data_sehat_fake_source_label = tf.gather(tl_fake_label_from_source, bottom_10_indices).numpy()

        plt.figure(figsize=(5, 40))
        for i in range(10):
            for j in range(4):
                plt.subplot(20, 4, 8*i +j + 1)
                plt.imshow(bottom_10_data_org[i][3-j, :])
                plt.subplot(20, 4, 8*i +j + 5)
                plt.imshow(bottom_10_data[i][3-j, :])
                if j==0:
                    plt.title('{}\n{}\n{}'.format(bottom_10_data_label[i],bottom_10_data_label_smooth[i], bottom_10_data_label_reward[i]))
            plt.axis('off')

        plt.savefig('{}_low'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
        self._run_wandb.log({
            "fakesamples/{}_epoch/fake_low_from_tl".format(self.cnt): [wandb.Image(
                '{}_low'.format(save_tl_dir) + '.png')]
        }, step=self._tr_cnt)
        plt.close()

        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(top_10_data_True_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "True_feature_target_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(top_10_data_True_fwgan[i],top_10_data_True_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(top_10_data_True_img_source[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "True_feature_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(top_10_data_true_sehat_fwgan[i],top_10_data_true_sehat_label[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(top_10_data_sehat_target_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_target_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(top_10_data_sehat_target_fwgan[i],top_10_data_sehat_target_label[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(top_10_data_sehat_source_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(top_10_data_sehat_source_fwgan[i],top_10_data_sehat_source_label[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(top_10_data_sehat_fake_source_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_from_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(top_10_data_sehat_fake_source_fwgan[i],top_10_data_sehat_fake_source_label[i]))
            plt.axis('off')
            plt.savefig('{}_top_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/diff_high_{}".format(self.cnt, i): [wandb.Image(
                    '{}_top_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.1)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(bot_10_data_True_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "True_feature_target_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(bot_10_data_True_fwgan[i],bot_10_data_True_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.1)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(bot_10_data_True_img_source[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "True_feature_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(bot_10_data_true_sehat_fwgan[i],bot_10_data_true_sehat_label[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.1)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(bot_10_data_sehat_target_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_target_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(bot_10_data_sehat_target_fwgan[i],bot_10_data_sehat_target_label[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.1)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(bot_10_data_sehat_source_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(bot_10_data_sehat_source_fwgan[i],bot_10_data_sehat_source_label[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.1)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(bot_10_data_sehat_fake_source_img[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "sehat_feature_from_source_recon\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(bot_10_data_sehat_fake_source_fwgan[i],bot_10_data_sehat_fake_source_label[i]))
            plt.axis('off')
            plt.savefig('{}_bot_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "fakesamples/{}_epoch/diff_low_{}".format(self.cnt, i): [wandb.Image(
                    '{}_bot_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()
        merge = [tl_fake_label]

        #이 아래로 Label등의 추정값 이용하는 코드 추가
        artsortdata = tf.argsort(tl_true_label, axis=0).numpy().flatten()
        # artsortdata = tf.argsort(feature_difference, axis=0)

        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_tl = tf.gather(tl_img, top_10_indices).numpy()
        label_high_tl_label = tf.gather(tl_true_label, top_10_indices).numpy()
        label_high_tl_fwgan = tf.gather(feature_tl_true, top_10_indices).numpy()

        Label_low_tl = tf.gather(tl_img, bottom_10_indices).numpy()
        label_low_tl_label = tf.gather(tl_true_label, bottom_10_indices).numpy()
        label_low_tl_fwgan = tf.gather(feature_tl_true, bottom_10_indices).numpy()

        #위에 내용이 첫 이미지에 들어갈 재료들

        artsortdata = tf.argsort(true_sehat_label, axis=0).numpy().flatten()
        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_tl_sec = tf.gather(tl_img, top_10_indices).numpy()
        label_high_tl_label_sec = tf.gather(tl_true_label, top_10_indices).numpy()
        label_high_tl_fwgan_sec = tf.gather(feature_tl_true, top_10_indices).numpy()

        Label_high_tlse_sec = tf.gather(tl_1000_true_to_source, top_10_indices).numpy()
        Label_high_tlse_label_sec = tf.gather(true_sehat_fwgan, top_10_indices).numpy()
        Label_high_tlse_fwgan_sec = tf.gather(true_sehat_label, top_10_indices).numpy()

        Label_low_tl_sec = tf.gather(tl_img, bottom_10_indices).numpy()
        label_low_tl_label_sec = tf.gather(tl_true_label, bottom_10_indices).numpy()
        label_low_tl_fwgan_sec = tf.gather(feature_tl_true, bottom_10_indices).numpy()

        Label_low_tlse_sec = tf.gather(tl_1000_true_to_source, bottom_10_indices).numpy()
        Label_low_tlse_label_sec = tf.gather(true_sehat_fwgan, bottom_10_indices).numpy()
        Label_low_tlse_fwgan_sec = tf.gather(true_sehat_label, bottom_10_indices).numpy()

        #이 다음...

        artsortdata = tf.argsort(true_sehat_fwgan, axis=0).numpy().flatten()
        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_tl_trd = tf.gather(tl_img, top_10_indices).numpy()
        label_high_tl_label_trd = tf.gather(tl_true_label, top_10_indices).numpy()
        label_high_tl_fwgan_trd = tf.gather(feature_tl_true, top_10_indices).numpy()

        Label_high_tlse_trd = tf.gather(tl_1000_true_to_source, top_10_indices).numpy()
        Label_high_tlse_label_trd = tf.gather(true_sehat_fwgan, top_10_indices).numpy()
        Label_high_tlse_fwgan_trd = tf.gather(true_sehat_label, top_10_indices).numpy()

        Label_low_tl_trd = tf.gather(tl_img, bottom_10_indices).numpy()
        label_low_tl_label_trd = tf.gather(tl_true_label, bottom_10_indices).numpy()
        label_low_tl_fwgan_trd = tf.gather(feature_tl_true, bottom_10_indices).numpy()

        Label_low_tlse_trd = tf.gather(tl_1000_true_to_source, bottom_10_indices).numpy()
        Label_low_tlse_label_trd = tf.gather(true_sehat_fwgan, bottom_10_indices).numpy()
        Label_low_tlse_fwgan_trd = tf.gather(true_sehat_label, bottom_10_indices).numpy()

    # #아래는 source to target
        artsortdata = tf.argsort(se_true_label, axis=0).numpy().flatten()
        # artsortdata = tf.argsort(feature_difference, axis=0)

        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_se = tf.gather(se_img, top_10_indices).numpy()
        label_high_se_label = tf.gather(se_true_label, top_10_indices).numpy()
        label_high_se_fwgan = tf.gather(true_se_fwgan, top_10_indices).numpy()

        Label_low_se = tf.gather(se_img, bottom_10_indices).numpy()
        label_low_se_label = tf.gather(se_true_label, bottom_10_indices).numpy()
        label_low_se_fwgan = tf.gather(true_se_fwgan, bottom_10_indices).numpy()

        artsortdata = tf.argsort(se_tl_label, axis=0).numpy().flatten()
        # artsortdata = tf.argsort(feature_difference, axis=0)

        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_se_sec = tf.gather(se_img, top_10_indices).numpy()
        label_high_se_label_sec = tf.gather(se_true_label, top_10_indices).numpy()
        label_high_se_fwgan_sec = tf.gather(true_se_fwgan, top_10_indices).numpy()

        Label_high_setl_sec = tf.gather(se_tl_img, top_10_indices).numpy()
        Label_high_setl_label_sec = tf.gather(se_tl_label, top_10_indices).numpy()
        Label_high_setl_fwgan_sec = tf.gather(se_tl_fwgan, top_10_indices).numpy()

        Label_low_se_sec = tf.gather(se_img, bottom_10_indices).numpy()
        label_low_se_label_sec = tf.gather(se_true_label, bottom_10_indices).numpy()
        label_low_se_fwgan_sec = tf.gather(true_se_fwgan, bottom_10_indices).numpy()

        Label_low_setl_sec = tf.gather(se_tl_img, bottom_10_indices).numpy()
        Label_low_setl_label_sec = tf.gather(se_tl_label, bottom_10_indices).numpy()
        Label_low_setl_fwgan_sec = tf.gather(se_tl_fwgan, bottom_10_indices).numpy()

        artsortdata = tf.argsort(true_se_fwgan, axis=0).numpy().flatten()
        # artsortdata = tf.argsort(feature_difference, axis=0)

        top_10_indices = artsortdata[-5:]  # 가장 큰 값의 인덱스
        bottom_10_indices = artsortdata[:5]  # 가장 작은 값의 인덱스

        Label_high_se_trd = tf.gather(se_img, top_10_indices).numpy()
        label_high_se_label_trd = tf.gather(se_true_label, top_10_indices).numpy()
        label_high_se_fwgan_trd = tf.gather(true_se_fwgan, top_10_indices).numpy()

        Label_high_setl_trd = tf.gather(se_tl_img, top_10_indices).numpy()
        Label_high_setl_label_trd = tf.gather(se_tl_label, top_10_indices).numpy()
        Label_high_setl_fwgan_trd = tf.gather(se_tl_fwgan, top_10_indices).numpy()

        Label_low_se_trd = tf.gather(se_img, bottom_10_indices).numpy()
        label_low_se_label_trd = tf.gather(se_true_label, bottom_10_indices).numpy()
        label_low_se_fwgan_trd = tf.gather(true_se_fwgan, bottom_10_indices).numpy()

        Label_low_setl_trd = tf.gather(se_tl_img, bottom_10_indices).numpy()
        Label_low_setl_label_trd = tf.gather(se_tl_label, bottom_10_indices).numpy()
        Label_low_setl_fwgan_trd = tf.gather(se_tl_fwgan, bottom_10_indices).numpy()


        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_tl[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_True_feature\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_tl_fwgan[i],label_high_tl_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_tl_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_tl_fwgan_sec[i],label_high_tl_label_sec[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_tlse_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_high_tlse_fwgan_sec[i],Label_high_tlse_label_sec[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_tl_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_tl_fwgan_trd[i],label_high_tl_label_trd[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_tlse_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_high_tlse_fwgan_trd[i],Label_high_tlse_label_trd[i]))
            plt.axis('off')
            plt.savefig('{}_high_tl_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "compares/{}_epoch/tl_estim_high_{}".format(self.cnt, i): [wandb.Image(
                    '{}_high_tl_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_tl[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_low_True_feature\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_tl_fwgan[i],label_low_tl_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_tl_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_tl_fwgan_sec[i],label_low_tl_label_sec[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_tlse_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_low_tlse_fwgan_sec[i],Label_low_tlse_label_sec[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_tl_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_tl_fwgan_trd[i],label_low_tl_label_trd[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_tlse_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_low_tlse_fwgan_trd[i],Label_low_tlse_label_trd[i]))
            plt.axis('off')
            plt.savefig('{}_lower_tl_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "compares/{}_epoch/tl_estim_low_{}".format(self.cnt, i): [wandb.Image(
                    '{}_lower_tl_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_se[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_True_feature\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_se_fwgan[i],label_high_se_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_se_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_se_fwgan_sec[i],label_high_se_label_sec[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_setl_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_high_setl_fwgan_sec[i],Label_high_setl_label_sec[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_se_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_high_se_fwgan_trd[i],label_high_se_label_trd[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_high_setl_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_high_setl_fwgan_trd[i],Label_high_setl_label_trd[i]))
            plt.axis('off')
            plt.savefig('{}_high_se_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "compares/{}_epoch/se_estim_high_{}".format(self.cnt, i): [wandb.Image(
                    '{}_high_se_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        for i in range(5):
            plt.figure(figsize=(8, 8))
            for j in range(5):
                if j ==0:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_se[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_low_True_feature\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_se_fwgan[i],label_low_se_label[i]))
                elif j ==1:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_se_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_se_fwgan_sec[i],label_low_se_label_sec[i]))
                elif j ==2:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_setl_sec[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "Label_high_target_to_source(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_low_setl_fwgan_sec[i],Label_low_setl_label_sec[i]))
                elif j ==3:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_se_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(True_tl)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(label_low_se_fwgan_trd[i],label_low_se_label_trd[i]))
                else:
                    for k in range(4):
                        plt.subplots_adjust(wspace=0.2, hspace=1.2)
                        plt.subplot(5, 4, 4*j +k + 1)
                        plt.imshow(Label_low_setl_trd[i][3-k, :])
                        if k==0:
                            plt.text(-60, -3, "fakewgan_high_True_feature(tl_to_source)\n fakewgan \n Label", fontsize=12, color='black')
                            plt.title('{}\n{}'.format(Label_low_setl_fwgan_trd[i],Label_low_setl_label_trd[i]))
            plt.axis('off')
            plt.savefig('{}_lower_se_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "compares/{}_epoch/se_estim_low_{}".format(self.cnt, i): [wandb.Image(
                    '{}_lower_se_{}'.format(save_tl_dir,i) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        return merge

    def plot_img_exec(self, my_buffer, domain, task):
        tl_ims_buf = []
        tl_rew_buf = []
        # for i in range(1000 // self._epi_len):
        traj_data = self.sampler.sample_trajectory(self.agent, 0)
        tl_uint = traj_data['ims'].astype(np.uint8)[:, 3, :]
        tl_ims_buf.append((traj_data['ims'].astype('float32') + 0.5) / 256)
        tl_rew_buf.append(traj_data['rew'])
        tl_img = np.concatenate(tl_ims_buf, axis=0)
        true_reward = np.concatenate(tl_rew_buf, axis=0)

        # tl_feature = self._pre_s(tl_img)
        # source_img = self._recon_layer_s(tl_feature)
        # uint_source_img = np.clip((source_img * 256) - 0.5, 0, 255).astype(np.uint8)[:, 3, :]

        reward_sum = np.sum(true_reward)

        save_tl_dir = self._log_dir + '/samples/{}_gif/reward_in_{}'.format(task, self.cnt)
        os.makedirs(save_tl_dir)

        # frames = [Image.fromarray(img) for img in uint_source_img]
        # frames[0].save('{}/render_fake.gif'.format(save_tl_dir), save_all=True, append_images=frames[1:], duration=100,
        #                loop=0)
        #
        # # plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
        # self._run_wandb.log({
        #     "gif/{}_epoch_fake_return_{}".format(self.cnt, reward_sum): [wandb.Video(
        #         '{}/render_fake'.format(save_tl_dir) + '.gif', fps=60, format="gif")]
        # }, step=self._tr_cnt)
        # frames[0].save('mnist_sample.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)

        frames = [Image.fromarray(img) for img in tl_uint]
        frames[0].save('{}/render.gif'.format(save_tl_dir), save_all=True, append_images=frames[1:], duration=100,
                       loop=0)

        # plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
        self._run_wandb.log({
            "gif/{}_epoch_return_{}".format(self.cnt, reward_sum): [wandb.Video(
                '{}/render'.format(save_tl_dir) + '.gif', fps=60, format="gif")]
        }, step=self._tr_cnt)

    def plot_img(self, my_buffer, domain, task):

        # if domain:
        #     buffer_index = random.randint(0, 48)
        # else:
        #     buffer_index = random.randint(0, 45)

        buffer_index = random.randint(0, 9)
        tl_1000_data, tl_1000_data_next = my_buffer.get_det_batch_index_double(1000, buffer_index * 1000)
        tl_img = tl_1000_data['ims']
        if task == 'tl':
            true_reward = tl_1000_data['rew']

        tl_ims_buf = []
        tl_rew_buf = []
        if task == 'tl_exec':
            for i in range(1000 // self._epi_len):
                traj_data = self.sampler.sample_trajectory(self.agent, 0)
                tl_ims_buf.append((traj_data['ims'].astype('float32') + 0.5) / 256)
                tl_rew_buf.append(traj_data['rew'])
            tl_img = np.concatenate(tl_ims_buf, axis=0)
            true_reward = np.concatenate(tl_rew_buf, axis=0)

        x = 500
        total_data = len(tl_img)

        recon_img_array = []
        fake_img_array = []
        cycle_img_array = []
        rerec_img_array = []

        fwgan_frame_array = []
        fwgan_seq_array = []
        recon_fwgan_frame_array = []
        recon_fwgan_seq_array = []
        fake_fwgan_frame_array = []
        fake_fwgan_seq_array = []
        cycle_fwgan_frame_array = []
        cycle_fwgan_seq_array = []
        rerec_fwgan_frame_array = []
        rerec_fwgan_seq_array = []

        true_label_array = []
        recon_label_array = []
        fake_label_array = []
        cycle_label_array = []
        rerec_label_array = []

        true_label_array_frame = []
        recon_label_array_frame = []
        fake_label_array_frame = []
        cycle_label_array_frame = []
        rerec_label_array_frame = []

        feature_array = []
        feature_recon_array = []
        feature_fake_array = []
        feature_cycle_array = []
        feature_rerec_array = []

        true_reward_array = []
        recon_reward_array = []
        fake_reward_array = []
        cycle_reward_array = []
        rerec_reward_array = []

        # true_reward_array= []

        for i in range(0, total_data, x):
            tl_img_chunk = tl_img[i:i + x]
            # tl_reward_chunk = true_reward[i:i+x]

            tl_feature = self._pre_s.get_state_feature_plot(tl_img_chunk)

            if domain:

                tl_1000_recon = self._recon_layer_t(tl_feature)
                tl_1000_recon_fake = self._recon_layer_s(tl_feature)

                tl_1000_label = self._label_net(tl_feature)
                tl_1000_label_frame = self._label_net_frame(tl_feature)

                tl_1000_reward = -tf.math.log((1-tl_1000_label) + 1e-12)

                fwgan_frame, fwgan_seq = self._feature_wgan.chk_loss(tl_feature)

                recon_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon)

                recon_tl_1000_label = self._label_net(recon_tl_1000_feature)
                recon_tl_1000_label_frame = self._label_net_frame(recon_tl_1000_feature)

                recon_tl_1000_reward = -tf.math.log((1-recon_tl_1000_label) + 1e-12)

                recon_fwgan_frame, recon_fwgan_seq = self._feature_wgan.chk_loss(recon_tl_1000_feature)

                fake_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon_fake)

                fake_tl_1000_label = self._label_net(fake_tl_1000_feature)
                fake_tl_1000_label_frame = self._label_net_frame(fake_tl_1000_feature)

                fake_tl_1000_reward = -tf.math.log((1-fake_tl_1000_label) + 1e-12)

                fake_fwgan_frame, fake_fwgan_seq = self._feature_wgan.chk_loss(fake_tl_1000_feature)

                tl_1000_cycle = self._recon_layer_t(fake_tl_1000_feature)
                cycle_feature = self._pre_s.get_state_feature_plot(tl_1000_cycle)

                cycle_tl_1000_label = self._label_net(cycle_feature)
                cycle_tl_1000_label_frame = self._label_net_frame(cycle_feature)

                cycle_tl_1000_reward = -tf.math.log((1-cycle_tl_1000_label) + 1e-12)

                cycle_fwgan_frame, cycle_fwgan_seq = self._feature_wgan.chk_loss(cycle_feature)

                tl_1000_rerecon = self._recon_layer_t(recon_tl_1000_feature)
                rerecon_feature = self._pre_s.get_state_feature_plot(tl_1000_rerecon)

                rerecon_tl_1000_label = self._label_net(rerecon_feature)
                rerecon_tl_1000_label_frame = self._label_net_frame(rerecon_feature)

                rerecon_tl_1000_reward = -tf.math.log((1-rerecon_tl_1000_label) + 1e-12)

                rerecon_fwgan_frame, rerecon_fwgan_seq = self._feature_wgan.chk_loss(rerecon_feature)

            else:

                tl_1000_recon = self._recon_layer_s(tl_feature)
                tl_1000_recon_fake = self._recon_layer_t(tl_feature)

                tl_1000_label = self._label_net(tl_feature)
                tl_1000_label_frame = self._label_net_frame(tl_feature)

                tl_1000_reward = -tf.math.log((1-tl_1000_label) + 1e-12)

                fwgan_frame, fwgan_seq = self._feature_wgan.chk_loss(tl_feature)

                recon_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon)

                recon_tl_1000_label = self._label_net(recon_tl_1000_feature)
                recon_tl_1000_label_frame = self._label_net_frame(recon_tl_1000_feature)

                recon_tl_1000_reward = -tf.math.log((1-recon_tl_1000_label) + 1e-12)

                recon_fwgan_frame, recon_fwgan_seq = self._feature_wgan.chk_loss(recon_tl_1000_feature)

                fake_tl_1000_feature = self._pre_s.get_state_feature_plot(tl_1000_recon_fake)

                fake_tl_1000_label = self._label_net(fake_tl_1000_feature)
                fake_tl_1000_label_frame = self._label_net_frame(fake_tl_1000_feature)

                fake_tl_1000_reward = -tf.math.log((1-fake_tl_1000_label) + 1e-12)

                fake_fwgan_frame, fake_fwgan_seq = self._feature_wgan.chk_loss(fake_tl_1000_feature)

                tl_1000_cycle = self._recon_layer_s(fake_tl_1000_feature)
                cycle_feature = self._pre_s.get_state_feature_plot(tl_1000_cycle)

                cycle_fwgan_frame, cycle_fwgan_seq = self._feature_wgan.chk_loss(cycle_feature)

                cycle_tl_1000_label = self._label_net(cycle_feature)
                cycle_tl_1000_label_frame = self._label_net_frame(cycle_feature)

                cycle_tl_1000_reward = -tf.math.log((1-cycle_tl_1000_label) + 1e-12)

                tl_1000_rerecon = self._recon_layer_s(recon_tl_1000_feature)
                rerecon_feature = self._pre_s.get_state_feature_plot(tl_1000_rerecon)

                rerecon_tl_1000_label = self._label_net(rerecon_feature)
                rerecon_tl_1000_label_frame = self._label_net_frame(rerecon_feature)

                rerecon_tl_1000_reward = -tf.math.log((1-rerecon_tl_1000_label) + 1e-12)
                rerecon_fwgan_frame, rerecon_fwgan_seq = self._feature_wgan.chk_loss(rerecon_feature)


            recon_img_array.append(tl_1000_recon)

            fake_img_array.append(tl_1000_recon_fake)
            cycle_img_array.append(tl_1000_cycle)

            rerec_img_array.append(tl_1000_rerecon)

            fwgan_frame_array.append(fwgan_frame)
            fwgan_seq_array.append(fwgan_seq)

            recon_fwgan_frame_array.append(recon_fwgan_frame)
            recon_fwgan_seq_array.append(recon_fwgan_seq)

            fake_fwgan_frame_array.append(fake_fwgan_frame)
            fake_fwgan_seq_array.append(fake_fwgan_seq)

            cycle_fwgan_frame_array.append(cycle_fwgan_frame)
            cycle_fwgan_seq_array.append(cycle_fwgan_seq)

            rerec_fwgan_frame_array.append(rerecon_fwgan_frame)
            rerec_fwgan_seq_array.append(rerecon_fwgan_seq)

            true_label_array.append(tl_1000_label)
            recon_label_array.append(recon_tl_1000_label)
            fake_label_array.append(fake_tl_1000_label)
            cycle_label_array.append(cycle_tl_1000_label)
            rerec_label_array.append(rerecon_tl_1000_label)

            true_label_array_frame.append(tl_1000_label_frame)
            recon_label_array_frame.append(recon_tl_1000_label_frame)
            fake_label_array_frame.append(fake_tl_1000_label_frame)
            cycle_label_array_frame.append(cycle_tl_1000_label_frame)
            rerec_label_array_frame.append(rerecon_tl_1000_label_frame)


            feature_array.append(tl_feature)
            feature_recon_array.append(recon_tl_1000_feature)
            feature_fake_array.append(fake_tl_1000_feature)
            feature_cycle_array.append(cycle_feature)
            feature_rerec_array.append(rerecon_feature)

            true_reward_array.append(tl_1000_reward)
            recon_reward_array.append(recon_tl_1000_reward)
            fake_reward_array.append(fake_tl_1000_reward)
            cycle_reward_array.append(cycle_tl_1000_reward)
            rerec_reward_array.append(rerecon_tl_1000_reward)

        tl_1000_recon = np.concatenate(recon_img_array, axis=0)
        tl_1000_recon_fake = np.concatenate(fake_img_array, axis=0)
        tl_1000_cycle = np.concatenate(cycle_img_array, axis=0)
        tl_1000_rerec = np.concatenate(rerec_img_array, axis=0)

        fwgan_f = np.round(np.concatenate(fwgan_frame_array, axis=0), decimals=2)
        fwgan_s = np.round(np.concatenate(fwgan_seq_array, axis=0), decimals=2)

        r_fwgan_f = np.round(np.concatenate(recon_fwgan_frame_array, axis=0), decimals=2)
        r_fwgan_s = np.round(np.concatenate(recon_fwgan_seq_array, axis=0), decimals=2)

        f_fwgan_f = np.round(np.concatenate(fake_fwgan_frame_array, axis=0), decimals=2)
        f_fwgan_s = np.round(np.concatenate(fake_fwgan_seq_array, axis=0), decimals=2)

        c_fwgan_f = np.round(np.concatenate(cycle_fwgan_frame_array, axis=0), decimals=2)
        c_fwgan_s = np.round(np.concatenate(cycle_fwgan_seq_array, axis=0), decimals=2)

        rr_fwgan_f = np.round(np.concatenate(rerec_fwgan_frame_array, axis=0), decimals=2)
        rr_fwgan_s = np.round(np.concatenate(rerec_fwgan_seq_array, axis=0), decimals=2)

        true_label_preds = np.round(np.concatenate(true_label_array, axis=0), decimals=2)
        recon_label_preds = np.round(np.concatenate(recon_label_array, axis=0), decimals=2)
        fake_label_preds = np.round(np.concatenate(fake_label_array, axis=0), decimals=2)
        cycle_label_preds = np.round(np.concatenate(cycle_label_array, axis=0), decimals=2)
        rerecon_label_preds = np.round(np.concatenate(rerec_label_array, axis=0), decimals=2)

        true_label_preds_frame = np.round(np.concatenate(true_label_array_frame, axis=0), decimals=2)
        recon_label_preds_frame = np.round(np.concatenate(recon_label_array_frame, axis=0), decimals=2)
        fake_label_preds_frame = np.round(np.concatenate(fake_label_array_frame, axis=0), decimals=2)
        cycle_label_preds_frame = np.round(np.concatenate(cycle_label_array_frame, axis=0), decimals=2)
        rerecon_label_preds_frame = np.round(np.concatenate(rerec_label_array_frame, axis=0), decimals=2)

        feature_tl = np.concatenate(feature_array, axis=0)
        feature_recon = np.concatenate(feature_recon_array, axis=0)
        feature_fake = np.concatenate(feature_fake_array, axis=0)
        feature_cycle = np.concatenate(feature_cycle_array, axis=0)
        feature_rerec = np.concatenate(feature_rerec_array, axis=0)

        predict_true_reward = np.concatenate(true_reward_array, axis=0)
        predict_recon_reward = np.concatenate(recon_reward_array, axis=0)
        predict_fake_reward = np.concatenate(fake_reward_array, axis=0)
        predict_cycle_reward = np.concatenate(cycle_reward_array, axis=0)
        predict_rerec_reward = np.concatenate(rerec_reward_array, axis=0)

        diff_scalar = np.round(self.mse_loss_batch(feature_tl[:,32:64], feature_tl[:,:32]), decimals=6)

        # diff_org = np.diff(tf.constant(feature_tl[:, self.feature_size:2 * self.feature_size]).numpy(), axis=0)
        # first_row = np.zeros((1, self.feature_size))
        # diff = np.vstack((first_row, diff_org))
        # diff_scalar = np.round(np.mean(diff, axis=1), decimals=6)

        recon_feature_dist = np.round(self.mse_loss_batch(feature_tl[:,32:64], feature_recon[:,32:64]), decimals=6)
        fake_feature_dist = np.round(self.mse_loss_batch(feature_tl[:,32:64], feature_fake[:,32:64]), decimals=6)
        cycle_feature_dist = np.round(self.mse_loss_batch(feature_tl[:,32:64], feature_cycle[:,32:64]), decimals=6)
        rerec_feature_dist = np.round(self.mse_loss_batch(feature_tl[:,32:64], feature_rerec[:,32:64]), decimals=6)

        save_tl_dir = self._log_dir + '/samples/{}_1000/reward_in_{}'.format(task, self.cnt)
        os.makedirs(save_tl_dir)


        if task == 'tl' or task =='tl_exec':
            for i in range(10):
                fig = plt.figure(figsize=(12, 10))
                plt.subplots_adjust(wspace=0.2, hspace=1.1)

                for j in range(5):
                    for k in range(9):
                        index = 45 * i + k
                        ax = fig.add_subplot(5, 9, j * 9 + k + 1)

                        if j == 0:
                            description = "d$(feat_{org_{t}}, feat_{org_{t+1}})$\n fwgan_frame \n fwgan_seq \n label_net[Expert=1]  \nlabel_frame[Expert=1]\n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{:.5e}\n{:.5e}'.format(diff_scalar[index], fwgan_f[index][0], fwgan_s[index][0],true_label_preds[index][0],true_label_preds_frame[index][0], predict_true_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_img[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_img[index][3, :], title)

                        elif j == 1:
                            description = "d$(feat_{org}, feat_{recon})$\n fwgan_frame \n fwgan_seq \nlabel_net[Expert=1]\n label_frame[Expert=1]\npred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(recon_feature_dist[index], r_fwgan_f[index][0], r_fwgan_s[index][0],recon_label_preds[index][0],recon_label_preds_frame[index][0],  predict_recon_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_recon[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_recon[index][3, :], title)

                        elif j == 2:
                            description = "d$(feat_{org}, feat_{fake})$\n fwgan_frame \n fwgan_seq \nlabel_net[Expert=1]\nlabel_frame[Expert=1]\n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(fake_feature_dist[index], f_fwgan_f[index], f_fwgan_s[index],fake_label_preds[index],fake_label_preds_frame[index],  predict_fake_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_recon_fake[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_recon_fake[index][3, :], title)
                        elif j == 3:
                            description = "d$(feat_{org}, feat_{cycle})$\n fwgan_frame \n fwgan_seq\nlabel_net[Expert=1]\nlabel_frame[Expert=1]\n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(cycle_feature_dist[index], c_fwgan_f[index], c_fwgan_s[index],cycle_label_preds[index], cycle_label_preds_frame[index],predict_cycle_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_cycle[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_cycle[index][3, :], title)
                        elif j == 4:
                            description = "d$(feat_{org}, feat_{rerec})$\n fwgan_frame \n fwgan_seq\nlabel_net[Expert=1]\nlabel_frame[Expert=1]\n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(rerec_feature_dist[index], rr_fwgan_f[index], rr_fwgan_s[index],rerecon_label_preds[index], rerecon_label_preds_frame[index],predict_rerec_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_rerec[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_rerec[index][3, :], title)
                plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
                self._run_wandb.log({
                    "samples_{}/{}_epoch/img_bundle_{}".format(task, self.cnt, str(i).zfill(3)): [wandb.Image(
                        '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
                }, step=self._tr_cnt)

                plt.close()
        else:
            for i in range(10):
                fig = plt.figure(figsize=(12, 10))
                plt.subplots_adjust(wspace=0.2, hspace=1.1)

                for j in range(5):
                    for k in range(9):
                        index = 45 * i + k
                        ax = fig.add_subplot(5, 9, j * 9 + k + 1)
                        if j == 0:
                            description = "d$(feat_{org_{t}}, feat_{org_{t+1}})$\n fwgan_frame \n fwgan_seq \nlabel_net[Expert=1] \nlabel_frame[Expert=1]   \n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(diff_scalar[index], fwgan_f[index], fwgan_s[index], true_label_preds[index], true_label_preds_frame[index],  predict_true_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_img[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_img[index][3, :], title)

                        elif j == 1:
                            description = "d$(feat_{org}, feat_{recon})$\n fwgan_frame \n fwgan_seq \nlabel_net[Expert=1]\n label_frame[Expert=1]   \n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(recon_feature_dist[index], r_fwgan_f[index], r_fwgan_s[index],recon_label_preds[index], recon_label_preds_frame[index],predict_recon_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_recon[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_recon[index][3, :], title)

                        elif j == 2:
                            description = "d$(feat_{org}, feat_{fake})$\n fwgan_frame \n fwgan_seq \nlabel_net[Expert=1]\nlabel_frame[Expert=1]   \n  pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(fake_feature_dist[index], f_fwgan_f[index], f_fwgan_s[index],fake_label_preds[index], fake_label_preds_frame[index],predict_fake_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_recon_fake[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_recon_fake[index][3, :], title)
                        elif j == 3:
                            description = "d$(feat_{org}, feat_{cycle})$\n fwgan_frame \n fwgan_seq\nlabel_net[Expert=1]\n label_frame[Expert=1]   \n pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(cycle_feature_dist[index], c_fwgan_f[index], c_fwgan_s[index],cycle_label_preds[index],cycle_label_preds_frame[index], predict_cycle_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_cycle[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_cycle[index][3, :], title)
                        elif j == 4:
                            description = "d$(feat_{org}, feat_{rerec})$\n fwgan_frame \n fwgan_seq\nlabel_net[Expert=1]\nlabel_frame[Expert=1]   \n  pred_reward"
                            title = '{:.5e}\n{}\n{}\n{}\n{}\n{:.5e}'.format(rerec_feature_dist[index], rr_fwgan_f[index], rr_fwgan_s[index],rerecon_label_preds[index],rerecon_label_preds_frame[index],  predict_rerec_reward[index][0])
                            if k == 0:
                                self.plot_subplot(ax, tl_1000_rerec[index][3, :], title, description)
                            else:
                                self.plot_subplot(ax, tl_1000_rerec[index][3, :], title)
                plt.savefig('{}/{}th_sequence'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
                self._run_wandb.log({
                    "samples_{}/{}_epoch/img_bundle_{}".format(task, self.cnt, str(i).zfill(3)): [wandb.Image(
                        '{}/{}th_sequence'.format(save_tl_dir, i) + '.png')]
                }, step=self._tr_cnt)

                plt.close()

        if task == 'tl':
            artsortdata = tf.argsort(fwgan_f, axis=0).numpy().flatten()
            top_10_indices = artsortdata[-10:]
            bottom_10_indices = artsortdata[:10]

            top_10_data = tf.gather(tl_img, top_10_indices).numpy()
            top_10_data_label = tf.gather(fwgan_f, top_10_indices).numpy()
            top_10_data_fake = tf.gather(tl_1000_recon_fake, top_10_indices).numpy()

            bottom_10_data = tf.gather(tl_img, bottom_10_indices).numpy()
            bottom_10_data_label = tf.gather(fwgan_f, bottom_10_indices).numpy()
            bottom_10_data_fake = tf.gather(tl_1000_recon_fake, bottom_10_indices).numpy()

            plt.figure(figsize=(15, 7))
            for i in range(10):
                plt.subplot(4, 10, i + 1)
                plt.imshow(top_10_data[i][1, :])
                plt.title(top_10_data_label[i])
                plt.axis('off')
                plt.subplot(4, 10, 10 + i + 1)
                plt.imshow(top_10_data_fake[i][1, :])
                plt.title('fake')
                plt.axis('off')

            for i in range(10):
                plt.subplot(4, 10, 20 + i + 1)
                plt.imshow(bottom_10_data[i][1, :])
                plt.title(bottom_10_data_label[i])
                plt.axis('off')
                plt.subplot(4, 10, 30 + i + 1)
                plt.imshow(bottom_10_data_fake[i][1, :])
                plt.title('fake')
                plt.axis('off')

            save_tl_dir = self._log_dir + '/samples/{}_featurewgan_frame/{}_epoch'.format(task, self.cnt)
            os.makedirs(save_tl_dir)

            plt.savefig('{}'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "samples_{}/{}_epoch/featurewgan_img".format(task, self.cnt): [wandb.Image(
                    '{}'.format(save_tl_dir) + '.png')]
            }, step=self._tr_cnt)

        if task == 'tl':
            artsortdata = tf.argsort(fwgan_s, axis=0).numpy().flatten()
            top_10_indices = artsortdata[-10:]
            bottom_10_indices = artsortdata[:10]

            top_10_data = tf.gather(tl_img, top_10_indices).numpy()
            top_10_data_label = tf.gather(fwgan_s, top_10_indices).numpy()

            bottom_10_data = tf.gather(tl_img, bottom_10_indices).numpy()
            bottom_10_data_label = tf.gather(fwgan_s, bottom_10_indices).numpy()

            plt.figure(figsize=(5, 20))

            save_tl_dir = self._log_dir + '/samples/{}_featurewgan_seq/{}_epoch'.format(task, self.cnt)
            os.makedirs(save_tl_dir)
            for i in range(10):
                for j in range(4):
                    plt.subplot(10, 4, 4*i +j + 1)
                    plt.imshow(top_10_data[i][3-j, :])
                    if j==0:
                        plt.title(top_10_data_label[i])
                plt.axis('off')

            plt.savefig('{}_high'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "samples_{}/{}_epoch/featurewgan_seq_high".format(task, self.cnt): [wandb.Image(
                    '{}_high'.format(save_tl_dir) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

            plt.figure(figsize=(5, 20))
            for i in range(10):
                for j in range(4):
                    plt.subplot(10, 4, 4*i +j + 1)
                    plt.imshow(bottom_10_data[i][3-j, :])
                    if j==0:
                        plt.title(bottom_10_data_label[i])
                plt.axis('off')

            plt.savefig('{}_low'.format(save_tl_dir), bbox_inches='tight', pad_inches=0.1)
            self._run_wandb.log({
                "samples_{}/{}_epoch/featurewgan_seq_low".format(task, self.cnt): [wandb.Image(
                    '{}_low'.format(save_tl_dir) + '.png')]
            }, step=self._tr_cnt)
            plt.close()

        merge = [1]
        return merge

    def sampling_from_prob(self, prob, non_init_num,init_num, domain, agent_buffer):

        if domain =='source':
            selected_indices_endec = np.random.choice(len(self.frozen_source), size=non_init_num, p=prob,
                                                      replace=False)
            lower_indices = []
            higher_indices = []

            for idx in selected_indices_endec:
                if idx < self.expert_len:
                    lower_indices.append(idx)
                else:
                    higher_indices.append(idx - self.expert_len)
            long_data = self._exp_buff.get_noninit_data_from_index(lower_indices)
            long_ims = long_data['ims']
            short_data = self._pr_exp_buff.get_noninit_data_from_index(higher_indices)
            short_ims = short_data['ims']
            short_data_init = self._pr_exp_buff.get_noninit_data_from_index_init(init_num)
            short_init_ims = short_data_init['ims']

            batch_data = tf.concat([long_ims, short_ims, short_init_ims],axis=0)
        else:
            selected_indices_endec = np.random.choice(len(self.frozen_target), size=non_init_num, p=prob,
                                                      replace=False)
            lower_indices = []
            higher_indices = []

            for idx in selected_indices_endec:
                if idx < self.expert_len:
                    lower_indices.append(idx)
                else:
                    higher_indices.append(idx - self.expert_len)
            long_data = agent_buffer.get_noninit_data_from_index(lower_indices)
            long_ims = long_data['ims']
            short_data = self._pr_age_buff.get_noninit_data_from_index(higher_indices)
            short_ims = short_data['ims']
            short_data_init = self._pr_age_buff.get_noninit_data_from_index_init(init_num)
            short_init_ims = short_data_init['ims']

            batch_data = tf.concat([long_ims, short_ims, short_init_ims],axis=0)

        return batch_data

    def process_data(self, a, b, c, d, m):
        def split_and_shuffle(data1, data2, m):
            # 1. 마지막 m개 데이터를 분리
            data1_main, data1_tail = data1[:-m], data1[-m:]
            data2_main, data2_tail = data2[:-m], data2[-m:]

            # 2. 두 데이터를 합치고 무작위로 섞기
            combined = np.concatenate([data1_main, data2_main])
            np.random.shuffle(combined)

            # 3. 섞은 데이터를 두 부분으로 나누기
            split_idx = len(combined) // 2
            part1 = combined[:split_idx]
            part2 = combined[split_idx:]

            # 4. 각각의 부분에 원래의 m개 데이터를 붙이기
            part1 = np.concatenate([part1, data1_tail])
            part2 = np.concatenate([part2, data2_tail])

            return part1, part2

        # a, b 처리
        new_a, new_b = split_and_shuffle(a, b, m)
        # c, d 처리
        new_c, new_d = split_and_shuffle(c, d, m)

        return new_a, new_b, new_c, new_d

    def train(self, agent_buffer, l_batch_size=128, l_updates=1, l_act_delay=1,
              d_updates=1, mi_updates=1, d_e_batch_size=128, d_l_batch_size=128, sampling_alpha=0):
        """Train discriminator, statistics network, and learner agent models.

        Parameters
        ----------
        agent_buffer : Buffer containing the agent-collected experience.
        l_batch_size : Batch size of agent experience used to train the learner agent models, default is 128.
        l_updates : Number of updates to train learner agent, default is 1.
        l_act_delay : Actor delay (1/frequency) to train the learner agent policy, default is 1.
        d_updates : Number of updates to train the discriminator, default is 1.
        mi_updates : Number of updates to train the statistics network, default is 1.
        d_e_batch_size : Batch size of agent experience used to train the discriminator models, default is 128.
        d_l_batch_size : Batch size of expert experience used to train the discriminator models, default is 128.
        """
        self.cnt = self.cnt +1

        tr_cnt = 0
        self.int_cnt = 0
        self._batch_size = d_e_batch_size
        start_time = time.time()
        training_ratio = 5

        if self.cnt <= self.sehat:
            use_sehat = False
        else:
            use_sehat = True

        if self.cnt <= 10000:
            training_model = True
        else:
            training_model = False

        # training_model = False
        nstep_list = [0]
        sampling_alpha = sampling_alpha
        self.nstep_list = nstep_list

        # if self.init:
        #
        #     print("update clustering")
        #     se_init, se_tot = self._exp_buff.get_noninit_split_data()
        #     se_init_img, se_img = se_init['ims'], se_tot['ims']
        #
        #     tl_init, tl_tot = agent_buffer.get_noninit_split_data()
        #     tl_init_img, tl_img = tl_init['ims'], tl_tot['ims']
        #
        #     sr_init, sr_tot = self._pr_exp_buff.get_noninit_split_data()
        #     sr_init_img, sr_img = sr_init['ims'], sr_tot['ims']
        #
        #     tr_init, tr_tot = self._pr_age_buff.get_noninit_split_data()
        #     tr_init_img, tr_img = tr_init['ims'], tr_tot['ims']
        #
        #     se_len = len(se_img)
        #     self.expert_len = len(se_img)
        #     # source_img = np.concatenate([se_img, sr_img], axis=0)
        #
        #     # target_img = np.concatenate([tl_img, tr_img], axis=0)
        #
        #     split = 250
        #     se_buf = []
        #     tl_buf = []
        #     sr_buf = []
        #     tr_buf = []
        #
        #     for i in range(0, se_len, split):
        #         img_chunk = se_img[i:i + split]
        #         se_buf.append(self._frozen_net(img_chunk[:,1]))
        #
        #     for i in range(0, len(sr_img), split):
        #         img_chunk = sr_img[i:i + split]
        #         sr_buf.append(self._frozen_net(img_chunk[:,1]))
        #
        #     for i in range(0, len(tl_img), split):
        #         img_chunk = tl_img[i:i + split]
        #         tl_buf.append(self._frozen_net(img_chunk[:,1]))
        #
        #     for i in range(0, len(tr_img), split):
        #         img_chunk = tr_img[i:i + split]
        #         tr_buf.append(self._frozen_net(img_chunk[:,1]))
        #     se_arr = np.concatenate(se_buf,axis=0)
        #     sr_arr = np.concatenate(sr_buf,axis=0)
        #     tl_arr = np.concatenate(tl_buf,axis=0)
        #     tr_arr = np.concatenate(tr_buf,axis=0)
        #
        #     self.frozen_se = se_arr
        #     self.frozen_source = np.concatenate([se_arr, sr_arr],axis=0)
        #     self.frozen_target = np.concatenate([tl_arr, tr_arr],axis=0)
        #
        #     n_clusters = 100
        #
        #     kmeans_se = KMeans(n_clusters=n_clusters)
        #     kmeans_se.fit(self.frozen_se)
        #     labels = kmeans_se.labels_
        #     self._label_se = labels
        #
        #     label_counts = Counter(labels)
        #
        #     sample_probabilities = np.array([1.0 / label_counts[label] for label in labels])
        #     exponentiated_array = np.exp(0 * np.log(sample_probabilities + 1e-10))
        #
        #     self.se_sample_sehat_prob = (exponentiated_array/exponentiated_array.sum())
        #
        #     exponentiated_array = np.exp(sampling_alpha * np.log(sample_probabilities + 1e-10))
        #
        #     self.se_sample_prob = (exponentiated_array /exponentiated_array.sum())
        #
        #     n_clusters = 100
        #
        #     kmeans_source = KMeans(n_clusters=n_clusters)
        #     kmeans_source.fit(self.frozen_source)
        #     labels = kmeans_source.labels_
        #     self._label_source = labels
        #
        #     label_counts = Counter(labels)
        #
        #     sample_probabilities = np.array([1.0 / label_counts[label] for label in labels])
        #
        #     exponentiated_array = np.exp(sampling_alpha * np.log(sample_probabilities + 1e-10))
        #
        #     self.source_prob = (exponentiated_array/exponentiated_array.sum())
        #
        #     # with tf.device('/CPU:0'):
        #     #     self.frozen_tl = self._pre_s.one_img(tl_img[:,1])
        #     #     frozen_tr = self._pre_s.one_img(tr_img[:,1])
        #     #     self.frozen_target = np.concatenate([self.frozen_tl, frozen_tr],axis=0)
        #
        #     n_clusters = 100
        #
        #     kmeans_target = KMeans(n_clusters=n_clusters)
        #     kmeans_target.fit(self.frozen_target)
        #     labels = kmeans_target.labels_
        #     self._label_target = labels
        #
        #     label_counts = Counter(labels)
        #
        #     sample_probabilities = np.array([1.0 / label_counts[label] for label in labels])
        #     exponentiated_array = np.exp(sampling_alpha * np.log(sample_probabilities + 1e-10))
        #
        #     self.target_prob = (exponentiated_array / exponentiated_array.sum())
        #
        #     del se_img, tl_img, sr_img, tr_img, se_init_img, tl_init_img, sr_init_img, tr_init_img
        #
        #     gc.collect()
        #
        #     self.init = False
        # else:
        #     split = 250
        #     tl_buf = []
        #     tl_init, tl_tot = agent_buffer.get_noninit_split_data_recent()
        #     tl_init_img, tl_img = tl_init['ims'], tl_tot['ims']
        #
        #     for i in range(0, len(tl_img), split):
        #         img_chunk = tl_img[i:i + split]
        #         tl_buf.append(self._frozen_net(img_chunk[:, 1]))
        #
        #     tl_arr = np.concatenate(tl_buf,axis=0)
        #     tl = self.frozen_target[:len(self.frozen_se)]
        #     tl_update = np.concatenate([tl[len(tl_arr):], tl_arr], axis=0)
        #     tr = self.frozen_target[len(self.frozen_se):]
        #
        #     self.frozen_target = np.concatenate([tl_update, tr], axis=0)
        #
        #     n_clusters = 100
        #
        #     kmeans_target = KMeans(n_clusters=n_clusters)
        #     kmeans_target.fit(self.frozen_target)
        #     labels = kmeans_target.labels_
        #     self._label_target = labels
        #
        #     label_counts = Counter(labels)
        #
        #     sample_probabilities = np.array([1.0 / label_counts[label] for label in labels])
        #     exponentiated_array = np.exp(sampling_alpha * np.log(sample_probabilities + 1e-10))
        #
        #     self.target_prob = (exponentiated_array / exponentiated_array.sum())

        # label_batch1 = []
        # label_batch2 = []
        # label_batch3 = []
        #
        # label_batch1_tl = []
        # label_batch2_tl = []
        # label_batch3_tl = []
        #
        # uniq_label = np.arange(100)
        # np.random.shuffle(uniq_label)

        # for i in uniq_label:
        #     cluster_indices = np.where(self._label_source == i)[0]
        #     if len(cluster_indices) >= 3:  # 클러스터 내에 샘플이 두 개 이상 있는 경우에만 선택
        #         selected_indices = np.random.choice(cluster_indices, size=3, replace=False)
        #         label_batch1.append(source_img[selected_indices[0]])
        #         label_batch2.append(source_img[selected_indices[1]])
        #         label_batch3.append(source_img[selected_indices[2]])
        #     else:
        #         # 클러스터 내에 샘플이 두 개 미만인 경우 임의로 하나를 선택하여 복사 (에러 방지)
        #         selected_index = cluster_indices[0]
        #         label_batch1.append(source_img[selected_index])
        #         label_batch2.append(source_img[selected_index])
        #         label_batch3.append(source_img[selected_index])
        #
        # batch1 = np.array(label_batch1)
        # batch2 = np.array(label_batch2)
        # batch3 = np.array(label_batch3)
        # contrastive_batch2 = np.roll(batch2, shift=1, axis=0)
        # contrastive_batch3 = np.roll(batch3, shift=2, axis=0)
        #
        # for i in uniq_label:
        #     cluster_indices = np.where(self._label_target == i)[0]
        #     if len(cluster_indices) >= 3:  # 클러스터 내에 샘플이 두 개 이상 있는 경우에만 선택
        #         selected_indices = np.random.choice(cluster_indices, size=3, replace=False)
        #         label_batch1_tl.append(target_img[selected_indices[0]])
        #         label_batch2_tl.append(target_img[selected_indices[1]])
        #         label_batch3_tl.append(target_img[selected_indices[2]])
        #     else:
        #         # 클러스터 내에 샘플이 두 개 미만인 경우 임의로 하나를 선택하여 복사 (에러 방지)
        #         selected_index = cluster_indices[0]
        #         label_batch1_tl.append(target_img[selected_index])
        #         label_batch2_tl.append(target_img[selected_index])
        #         label_batch3_tl.append(target_img[selected_index])
        #
        # batch1_tl = np.array(label_batch1_tl)
        # batch2_tl = np.array(label_batch2_tl)
        # batch3_tl = np.array(label_batch3_tl)
        # contrastive_batch2_tl = np.roll(batch2_tl, shift=1, axis=0)
        # contrastive_batch3_tl = np.roll(batch3_tl, shift=2, axis=0)

        if training_model:
            for i in range(d_updates):
            # for i in range(10):

                init_num = int(d_l_batch_size // 32)
                non_init_num = d_l_batch_size - init_num

                tl_batch, tl_timestep_list = agent_buffer.get_balance_batch_nsteps_with_step(d_l_batch_size, init_num, nstep_list)
                tl_combine_dense, tl_timesteps = tl_batch[0]['ims'], tl_timestep_list[0]
                se_batch, se_timestep_list = self._exp_buff.get_balance_batch_nsteps_with_step(d_e_batch_size, init_num, nstep_list)
                se_combine_dense, se_timesteps = se_batch[0]['ims'], se_timestep_list[0]
                tr_batch, tr_timestep_list = self._pr_age_buff.get_balance_batch_nsteps_with_step(d_l_batch_size, init_num, nstep_list)
                tr_combine, tr_timesteps = tr_batch[0]['ims'], tr_timestep_list[0]
                sr_batch, sr_timestep_list = self._pr_exp_buff.get_balance_batch_nsteps_with_step(d_e_batch_size, init_num, nstep_list)
                sr_combine, sr_timesteps = sr_batch[0]['ims'], sr_timestep_list[0]

                source_combine_cluster,source_combine_cluster2,target_combine_cluster,target_combine_cluster2  \
                    = self.process_data(se_combine_dense, sr_combine, tl_combine_dense, tr_combine,init_num)

                dense_sample = tf.concat([se_combine_dense, sr_combine, tl_combine_dense, tr_combine],axis=0)

                data_combine = np.concatenate([source_combine_cluster, source_combine_cluster2, target_combine_cluster, target_combine_cluster2], axis=0)

                dense_timesteps = tf.concat([se_timesteps, sr_timesteps, tl_timesteps, tr_timesteps] , axis=0)

                feat_discriminator_loss, feat_grad_norms, feat_gradient_penalty, disc_loss_single, disc_loss_seq = \
                    self.training_featurewgan(data_combine)

                if (self.int_cnt % training_ratio) == 0:
                    recon_loss, feature_recon_loss, feature_fake_loss,recon_label_loss, \
                    gen_loss, se_dist, sr_dist, tl_dist, tr_dist,frame_label_loss= \
                            self.SA_training(data_combine, dense_sample, dense_timesteps)

                if (self.int_cnt % d_updates) == (d_updates-1):
                    tr_cnt += 1
                    self._tr_cnt += 1
                    print("Model_training_done")
                    print("=================================")
                    loss_list = [recon_loss, feature_recon_loss,feature_fake_loss, frame_label_loss]
                    loss_list_name = ['recon_loss','feature_recon_loss','feature_fake_loss', 'recon_label_loss_frame']

                    for i, j in zip(loss_list_name, loss_list):

                        self._run_wandb.log(data={'{}_se'.format(i): tf.reduce_mean(j[:self._batch_size])}, step=self._tr_cnt)
                        self._run_wandb.log(data={'{}_sr'.format(i): tf.reduce_mean(j[self._batch_size:2 * self._batch_size])}, step=self._tr_cnt)
                        self._run_wandb.log(data={'{}_tl'.format(i): tf.reduce_mean(j[2*self._batch_size:3 * self._batch_size])}, step=self._tr_cnt)
                        self._run_wandb.log(data={'{}_tr'.format(i): tf.reduce_mean(j[3 * self._batch_size:])}, step=self._tr_cnt)

                    self._run_wandb.log(data={'recon_label_loss_se': tf.reduce_mean(recon_label_loss[:self._batch_size])}, step=self._tr_cnt)
                    self._run_wandb.log(data={'recon_label_loss_sr': tf.reduce_mean(recon_label_loss[self._batch_size:2 * self._batch_size])}, step=self._tr_cnt)
                    self._run_wandb.log(data={'recon_label_loss_tl_true': tf.reduce_mean(recon_label_loss[2*self._batch_size:3 * self._batch_size])}, step=self._tr_cnt)
                    self._run_wandb.log(data={'recon_label_loss_tr_true': tf.reduce_mean(recon_label_loss[3 * self._batch_size:4 * self._batch_size])}, step=self._tr_cnt)
                    self._run_wandb.log(data={'recon_label_loss_tl_sourcedom': tf.reduce_mean(recon_label_loss[4 * self._batch_size:5 * self._batch_size])}, step=self._tr_cnt)
                    self._run_wandb.log(data={'recon_label_loss_tr_sourcedom': tf.reduce_mean(recon_label_loss[5 * self._batch_size:6 * self._batch_size])}, step=self._tr_cnt)

                    self._run_wandb.log(
                        data={"discriminator_loss_feature": tf.reduce_mean(feat_discriminator_loss)},
                        step=self._tr_cnt)
                    self._run_wandb.log(data={"grad_norms_feature": tf.reduce_mean(feat_grad_norms)},
                                        step=self._tr_cnt)

                    self._run_wandb.log(data={"feat_gradient_penalty": tf.reduce_mean(feat_gradient_penalty)},
                                        step=self._tr_cnt)
                    self._run_wandb.log(data={"feat_discriminator_loss_frame": tf.reduce_mean(disc_loss_single)},
                                        step=self._tr_cnt)
                    self._run_wandb.log(data={"feat_discriminator_loss_seq": tf.reduce_mean(disc_loss_seq)},
                                        step=self._tr_cnt)

                    self._run_wandb.log(data={"se_dist": tf.reduce_mean(se_dist)},
                                        step=self._tr_cnt)
                    self._run_wandb.log(data={"sr_dist": tf.reduce_mean(sr_dist)},
                                        step=self._tr_cnt)
                    self._run_wandb.log(data={"tl_dist": tf.reduce_mean(tl_dist)},
                                        step=self._tr_cnt)
                    self._run_wandb.log(data={"tr_dist": tf.reduce_mean(tr_dist)},
                                        step=self._tr_cnt)

                self.int_cnt += 1
        else:
            self._tr_cnt += 1
            print("stop model training")

        if self.cnt == 10:
            plotting = True
        elif self.cnt == 20:
            plotting = True
        elif self.cnt == 50:
            plotting = True
        elif self.cnt % 100 == 0:
            plotting = True
        else:
            plotting = False

        # plotting = True
        # if True:
        # if self.cnt % 20 ==0:
        #     merge_tl_exec_gif = self.plot_img_exec(agent_buffer, 1, task='tl_exec')

        if plotting:
            simplefilter(action='ignore', category=FutureWarning)
            merge_tl = self.plot_img(agent_buffer, 1, task='tl')
            # merge_tl_exec = self.plot_img(agent_buffer, 1, task='tl_exec')
            merge_tr = self.plot_img(self._pr_age_buff, 1, task='tr')
            merge_se = self.plot_img(self._exp_buff, 0, task='se')
            merge_sr = self.plot_img(self._pr_exp_buff, 0, task='sr')
            # test_cluster = self.plot_cluster_samplee(batch1, batch2, batch3, contrastive_batch2, contrastive_batch3, 'se')
            # test_cluster = self.plot_cluster_samplee(batch1_tl, batch2_tl, batch3_tl, contrastive_batch2_tl, contrastive_batch3_tl, 'tl')
            print("image pairwise plotting complete")

        # if (self.cnt % 100)==0 or self.cnt ==1:
        #     tsne_plotting = True
        # else:
        #     tsne_plotting = False
        if plotting:
        # if plotting or self.cnt==1:
            if True:
                epi_len = 1000
                tl_img, f_tl, f_tl_hat, f_tl_recon = self.get_10000_feature(agent_buffer, 1, 'tl')
                tr_img, f_tr, f_tr_hat, f_tr_recon = self.get_10000_feature(self._pr_age_buff, 1, 'tr')
                se_img, f_se, f_se_hat, f_se_recon = self.get_10000_feature(self._exp_buff, 0, 'se')
                sr_img, f_sr, f_sr_hat, f_sr_recon = self.get_10000_feature(self._pr_exp_buff, 0, 'sr')


                feature_size = self.feature_size

                ln = f_tl[:, 3 * feature_size: 4 * feature_size]
                en = f_se[:, 3 * feature_size: 4 * feature_size]
                lpn = f_tr[:, 3 * feature_size: 4 * feature_size]
                epn = f_sr[:, 3 * feature_size: 4 * feature_size]
                # e_fake = f_tl_fake[:, 3 * feature_size: 4 * feature_size]

                ln_h = f_tl_hat[:, 3 * feature_size: 4 * feature_size]
                en_h = f_se_hat[:, 3 * feature_size: 4 * feature_size]
                lpn_h = f_tr_hat[:, 3 * feature_size: 4 * feature_size]
                epn_h = f_sr_hat[:, 3 * feature_size: 4 * feature_size]
                # e_fake_h = f_tl_hat_fake[:, 3 * feature_size: 4 * feature_size]

                ln_r = f_tl_recon[:, 3 * feature_size: 4 * feature_size]
                en_r = f_se_recon[:, 3 * feature_size: 4 * feature_size]
                lpn_r = f_tr_recon[:, 3 * feature_size: 4 * feature_size]
                epn_r = f_sr_recon[:, 3 * feature_size: 4 * feature_size]

                ln_seq = f_tl
                en_seq = f_se
                lpn_seq = f_tr
                epn_seq = f_sr
                # e_fake_seq = f_tl_fake

                ln_h_seq = f_tl_hat
                en_h_seq = f_se_hat
                lpn_h_seq = f_tr_hat
                epn_h_seq = f_sr_hat
                # e_fake_h_seq = f_tl_hat_fake

                ln_r_seq = f_tl_recon
                en_r_seq = f_se_recon
                lpn_r_seq = f_tr_recon
                epn_r_seq = f_sr_recon
                # e_fake_r_seq = f_tl_hat_recon

                sampling_size = 20
                # random_indices = en[:sampling_size]
                # random_indices = np.random.choice(epn.shape[0], size=sampling_size, replace=False)
                selected_feature = en[:sampling_size]
                selected_img = se_img[:sampling_size]

                # b에서 가장 가까운 데이터의 인덱스를 찾는 함수
                def find_closest_indices(selected_data, b_data):
                    closest_indices = []
                    for data in selected_data:
                        distances = np.linalg.norm(b_data - data, axis=1)  # 유클리디안 거리 계산
                        closest_index = np.argmin(distances)  # 가장 가까운 데이터의 인덱스
                        closest_indices.append(closest_index)
                    return closest_indices

                # 실행
                closest_indices_tl = find_closest_indices(selected_feature, ln)
                selected_img_tl = tl_img[closest_indices_tl]

                selected_feature_tl = f_tl[closest_indices_tl]

                selected_img_tr = tr_img[:sampling_size]

                selected_feature_tr = f_tr[:sampling_size]
                # closest_indices = find_closest_indices(selected_feature, lpn)
                # selected_img_tr = tr_img[closest_indices]
                #
                # selected_feature_tr = f_tr[closest_indices]

                selected_feature = f_se[:sampling_size]
                se_frame_label = self._label_net_frame(selected_feature)
                se_seq_label = self._label_net(selected_feature)

                tl_frame_label = self._label_net_frame(selected_feature_tl)
                tl_seq_label = self._label_net(selected_feature_tl)

                tr_frame_label = self._label_net_frame(selected_feature_tr)
                tr_seq_label = self._label_net(selected_feature_tr)

                tl_reward = -tf.math.log((1 - (tl_frame_label * tl_seq_label)) + 1e-12)

                se_frame_label = np.round(se_frame_label, decimals=2)
                se_seq_label = np.round(se_seq_label, decimals=2)
                tl_frame_label = np.round(tl_frame_label, decimals=2)
                tl_seq_label = np.round(tl_seq_label, decimals=2)
                tl_estim_reward = np.round(tl_reward, decimals=4)
                tr_frame_label = np.round(tr_frame_label, decimals=2)
                tr_seq_label = np.round(tr_seq_label, decimals=2)

                save_tl_dir = self._log_dir + '/samples/closest_img_frame/{}_epoch'.format(self.cnt)
                os.makedirs(save_tl_dir)
                for i in range(sampling_size):
                    plt.figure(figsize=(5, 8))
                    for j in range(4):
                        plt.subplot(3, 4, 1 + j)
                        # plt.subplot(4, 2, 4 * i + j + 1)
                        plt.imshow(selected_img[i][3 - j, :])
                        plt.axis('off')
                        if j == 1:
                            plt.title(f"Frame: {se_frame_label[i]}\nSeq: {se_seq_label[i]}",
                                      fontsize=8, loc='center')
                        plt.subplot(3, 4, 5 + j)
                        plt.imshow(selected_img_tl[i][3 - j, :])
                        plt.axis('off')
                        if j == 1:
                            plt.title(
                                f"Frame: {tl_frame_label[i]}\nSeq: {tl_seq_label[i]}\n Reward:{tl_estim_reward[i]}",
                                fontsize=8, loc='center')
                        plt.subplot(3, 4, 9 + j)
                        plt.imshow(selected_img_tr[i][3 - j, :])
                        plt.axis('off')
                        if j == 1:
                            plt.title(f"Frame: {tr_frame_label[i]}\nSeq: {tr_seq_label[i]}",
                                      fontsize=8, loc='center')
                        # if j == 0:
                        #     plt.title(bottom_10_data_label[i])
                    # plt.axis('off')

                    plt.savefig('{}_frame_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
                    self._run_wandb.log({
                        "samples/closest_img_frame/{}_epoch_{}".format(self.cnt, i): [wandb.Image(
                            '{}_frame_{}'.format(save_tl_dir, i) + '.png')]
                    }, step=self._tr_cnt)
                    plt.close()

                # print(asfdsadfsdafe)

                ln_seq = f_tl
                en_seq = f_se
                lpn_seq = f_tr
                epn_seq = f_sr

                sampling_size = 50
                random_indices = np.random.choice(en_seq.shape[0], size=sampling_size, replace=False)
                selected_feature = en_seq[random_indices]
                selected_img = se_img[random_indices]

                # b에서 가장 가까운 데이터의 인덱스를 찾는 함수
                def find_closest_indices(selected_data, b_data):
                    closest_indices = []
                    for data in selected_data:
                        distances = np.linalg.norm(b_data - data, axis=1)  # 유클리디안 거리 계산
                        closest_index = np.argmin(distances)  # 가장 가까운 데이터의 인덱스
                        closest_indices.append(closest_index)
                    return closest_indices

                # 실행
                closest_indices = find_closest_indices(selected_feature, ln_seq)
                selected_img_target = tl_img[closest_indices]
                selected_feature_tl = ln_seq[closest_indices]

                se_frame_label = self._label_net_frame(selected_feature)
                se_seq_label = self._label_net(selected_feature)

                tl_frame_label = self._label_net_frame(selected_feature_tl)
                tl_seq_label = self._label_net(selected_feature_tl)

                se_frame_label = np.round(se_frame_label, decimals=2)
                se_seq_label = np.round(se_seq_label, decimals=2)
                tl_frame_label = np.round(tl_frame_label, decimals=2)
                tl_seq_label = np.round(tl_seq_label, decimals=2)

                save_tl_dir = self._log_dir + '/samples/closest_img_seq/{}_epoch'.format(self.cnt)
                os.makedirs(save_tl_dir)
                for i in range(sampling_size):
                    plt.figure(figsize=(5, 5))
                    for j in range(4):
                        plt.subplot(2, 4, 1 + j)
                        # plt.subplot(4, 2, 4 * i + j + 1)
                        plt.imshow(selected_img[i][3 - j, :])
                        plt.axis('off')
                        if j == 1:
                            plt.title(f"Frame: {se_frame_label[i]}\nSeq: {se_seq_label[i]}",
                                      fontsize=8, loc='center')
                        plt.subplot(2, 4, 5 + j)
                        plt.imshow(selected_img_target[i][3 - j, :])
                        plt.axis('off')
                        if j == 1:
                            plt.title(f"Frame: {tl_frame_label[i]}\nSeq: {tl_seq_label[i]}",
                                      fontsize=8, loc='center')
                        # if j == 0:
                        #     plt.title(bottom_10_data_label[i])
                    # plt.axis('off')

                    plt.savefig('{}_seq_{}'.format(save_tl_dir, i), bbox_inches='tight', pad_inches=0.1)
                    self._run_wandb.log({
                        "samples/closest_img_seq/{}_epoch_{}".format(self.cnt, i): [wandb.Image(
                            '{}_seq_{}'.format(save_tl_dir, i) + '.png')]
                    }, step=self._tr_cnt)
                    plt.close()

                save_feature_dir = self._log_dir + '/feature_save/in{}'.format(self.cnt)
                os.makedirs(save_feature_dir)
                np.save(save_feature_dir + '/ftl.npy', f_tl)
                np.save(save_feature_dir + '/fse.npy', f_se)
                np.save(save_feature_dir + '/ftr.npy', f_tr)
                np.save(save_feature_dir + '/fsr.npy', f_sr)
                np.save(save_feature_dir + '/ftl_hat.npy', f_tl_hat)
                np.save(save_feature_dir + '/fse_hat.npy', f_se_hat)
                np.save(save_feature_dir + '/ftr_hat.npy', f_tr_hat)
                np.save(save_feature_dir + '/fsr_hat.npy', f_sr_hat)

                tsne_2d = TSNE(n_components=2, random_state=0, init='random', perplexity=50)
                tsne_2d_seq = TSNE(n_components=2, random_state=0, init='random', perplexity=50)

                save_t_dir = self._log_dir + '/tsne/epoch_{}'.format(self.cnt)
                os.makedirs(save_t_dir)

                div_size = len(ln)
                alpha = 0.1

                true_feature_data = tf.concat([ln, en, lpn, epn], axis=0)
                f_tsne = tsne_2d.fit_transform(true_feature_data)

                true_seq_data = tf.concat([ln_seq, en_seq, lpn_seq, epn_seq], axis=0)
                s_tsne = tsne_2d_seq.fit_transform(true_seq_data)

                source_domain_frame_data = tf.concat([ln_h, en, lpn_h, epn], axis=0)
                s_f_tsne = tsne_2d.fit_transform(source_domain_frame_data)

                source_domain_seq_data = tf.concat([ln_h_seq, en_seq, lpn_h_seq, epn_seq], axis=0)
                s_s_tsne = tsne_2d_seq.fit_transform(source_domain_seq_data)

                target_domain_data = tf.concat([ln, en_h, lpn, epn_h], axis=0)
                t_f_tsne = tsne_2d.fit_transform(target_domain_data)

                target_domain_seq_data = tf.concat([ln_seq, en_h_seq, lpn_seq, epn_h_seq], axis=0)
                t_s_tsne = tsne_2d_seq.fit_transform(target_domain_seq_data)

                compare_true_recon_data = tf.concat([ln, en, lpn, epn, ln_r, en_r, lpn_r, epn_r], axis=0)
                comp_f_tsne = tsne_2d.fit_transform(compare_true_recon_data)

                compare_true_recon_data_seq = tf.concat(
                    [ln_seq, en_seq, lpn_seq, epn_seq, ln_r_seq, en_r_seq, lpn_r_seq, epn_r_seq], axis=0)
                comp_s_tsne = tsne_2d_seq.fit_transform(compare_true_recon_data_seq)

                # 우선 마커는 x로
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(comp_f_tsne[:div_size, 0], comp_f_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(comp_f_tsne[div_size:2 * div_size, 0], comp_f_tsne[div_size:2 * div_size, 1], c='red',
                           label='se',
                           alpha=alpha)
                ax.scatter(comp_f_tsne[2 * div_size:3 * div_size, 0], comp_f_tsne[2 * div_size:3 * div_size, 1],
                           c='green',
                           label='tr', alpha=alpha)
                ax.scatter(comp_f_tsne[3 * div_size:4 * div_size, 0], comp_f_tsne[3 * div_size:4 * div_size, 1],
                           c='yellow',
                           label='sr', alpha=alpha)
                ax.scatter(comp_f_tsne[4 * div_size:5 * div_size, 0], comp_f_tsne[4 * div_size:5 * div_size, 1],
                           c='cyan', marker='x', label='tl_recon', alpha=alpha)
                ax.scatter(comp_f_tsne[5 * div_size:6 * div_size, 0], comp_f_tsne[5 * div_size:6 * div_size, 1],
                           c='magenta', marker='x', label='se_recon',
                           alpha=alpha)
                ax.scatter(comp_f_tsne[6 * div_size:7 * div_size, 0], comp_f_tsne[6 * div_size:7 * div_size, 1],
                           c='lime', marker='x',
                           label='tr_recon', alpha=alpha)
                ax.scatter(comp_f_tsne[7 * div_size:8 * div_size, 0], comp_f_tsne[7 * div_size:8 * div_size, 1],
                           c='gold', marker='x',
                           label='sr_recon', alpha=alpha)

                ax.set_title('True_recon_feature_compare')
                ax.legend()
                plt.savefig('{}/true_recon_comp'.format(save_t_dir))

                self._run_wandb.log({
                    "tsne/{}_epoch/true_recon_compare".format(self.cnt): [wandb.Image(
                        "{}/true_recon_comp".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(comp_s_tsne[:div_size, 0], comp_s_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(comp_s_tsne[div_size:2 * div_size, 0], comp_s_tsne[div_size:2 * div_size, 1], c='red',
                           label='se',
                           alpha=alpha)
                ax.scatter(comp_s_tsne[2 * div_size:3 * div_size, 0], comp_s_tsne[2 * div_size:3 * div_size, 1],
                           c='green',
                           label='tr', alpha=alpha)
                ax.scatter(comp_s_tsne[3 * div_size:4 * div_size, 0], comp_s_tsne[3 * div_size:4 * div_size, 1],
                           c='yellow',
                           label='sr', alpha=alpha)
                ax.scatter(comp_s_tsne[4 * div_size:5 * div_size, 0], comp_s_tsne[4 * div_size:5 * div_size, 1],
                           c='cyan', marker='x', label='tl_recon', alpha=alpha)
                ax.scatter(comp_s_tsne[5 * div_size:6 * div_size, 0], comp_s_tsne[5 * div_size:6 * div_size, 1],
                           c='magenta', marker='x', label='se_recon',
                           alpha=alpha)
                ax.scatter(comp_s_tsne[6 * div_size:7 * div_size, 0], comp_s_tsne[6 * div_size:7 * div_size, 1],
                           c='lime', marker='x',
                           label='tr_recon', alpha=alpha)
                ax.scatter(comp_s_tsne[7 * div_size:8 * div_size, 0], comp_s_tsne[7 * div_size:8 * div_size, 1],
                           c='gold', marker='x',
                           label='sr_recon', alpha=alpha)

                ax.set_title('True_recon_feature_seq_compare')
                ax.legend()
                plt.savefig('{}/true_recon_seq_comp'.format(save_t_dir))

                self._run_wandb.log({
                    "tsne/{}_epoch/true_recon_seq_compare".format(self.cnt): [wandb.Image(
                        "{}/true_recon_seq_comp".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(f_tsne[:div_size, 0], f_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(f_tsne[div_size:2 * div_size, 0], f_tsne[div_size:2 * div_size, 1], c='red', label='se',
                           alpha=alpha)
                ax.scatter(f_tsne[2 * div_size:3 * div_size, 0], f_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='tr', alpha=alpha)
                ax.scatter(f_tsne[3 * div_size:4 * div_size, 0], f_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='sr', alpha=alpha)

                ax.set_title('True_feature_frame')
                ax.legend()
                plt.savefig('{}/true_frame'.format(save_t_dir))

                self._run_wandb.log({
                    "tsne/{}_epoch/true_frame".format(self.cnt): [wandb.Image(
                        "{}/true_frame".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(s_tsne[:div_size, 0], s_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(s_tsne[div_size:2 * div_size, 0], s_tsne[div_size:2 * div_size, 1], c='red', label='se',
                           alpha=alpha)
                ax.scatter(s_tsne[2 * div_size:3 * div_size, 0], s_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='tr', alpha=alpha)
                ax.scatter(s_tsne[3 * div_size:4 * div_size, 0], s_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='sr', alpha=alpha)

                ax.set_title('True_feature_seq')
                # ax.set_title('feature')
                ax.legend()
                plt.savefig('{}/true_seq'.format(save_t_dir))
                self._run_wandb.log({
                    "tsne/{}_epoch/true_seq".format(self.cnt): [wandb.Image(
                        "{}/true_seq".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(s_f_tsne[:div_size, 0], s_f_tsne[:div_size, 1], c='blue', label='sl_hat', alpha=alpha)
                ax.scatter(s_f_tsne[div_size:2 * div_size, 0], s_f_tsne[div_size:2 * div_size, 1], c='red', label='se',
                           alpha=alpha)
                ax.scatter(s_f_tsne[2 * div_size:3 * div_size, 0], s_f_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='sr_hat', alpha=alpha)
                ax.scatter(s_f_tsne[3 * div_size:4 * div_size, 0], s_f_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='sr', alpha=alpha)

                ax.set_title('source_domain_frame')
                # ax.set_title('feature')
                ax.legend()
                plt.savefig('{}/source_frame'.format(save_t_dir))

                self._run_wandb.log({
                    "tsne/{}_epoch/source_frame".format(self.cnt): [wandb.Image(
                        "{}/source_frame".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(t_f_tsne[:div_size, 0], t_f_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(t_f_tsne[div_size:2 * div_size, 0], t_f_tsne[div_size:2 * div_size, 1], c='red',
                           label='te_hat',
                           alpha=alpha)
                ax.scatter(t_f_tsne[2 * div_size:3 * div_size, 0], t_f_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='tr', alpha=alpha)
                ax.scatter(t_f_tsne[3 * div_size:4 * div_size, 0], t_f_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='tr_hat', alpha=alpha)

                ax.set_title('target_domain_frame')
                # ax.set_title('feature')
                ax.legend()
                plt.savefig('{}/target_frame'.format(save_t_dir))
                self._run_wandb.log({
                    "tsne/{}_epoch/target_frame".format(self.cnt): [wandb.Image(
                        "{}/target_frame".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)
                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(s_s_tsne[:div_size, 0], s_s_tsne[:div_size, 1], c='blue', label='sl_hat', alpha=alpha)
                ax.scatter(s_s_tsne[div_size:2 * div_size, 0], s_s_tsne[div_size:2 * div_size, 1], c='red',
                           label='se', alpha=alpha)
                ax.scatter(s_s_tsne[2 * div_size:3 * div_size, 0], s_s_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='sr_hat', alpha=alpha)
                ax.scatter(s_s_tsne[3 * div_size:4 * div_size, 0], s_s_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='sr', alpha=alpha)

                ax.set_title('source_domain_seq')
                # ax.set_title('feature')
                ax.legend()
                plt.savefig('{}/source_seq'.format(save_t_dir))
                self._run_wandb.log({
                    "tsne/{}_epoch/source_seq".format(self.cnt): [wandb.Image(
                        "{}/source_seq".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)

                plt.close()

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(t_s_tsne[:div_size, 0], t_s_tsne[:div_size, 1], c='blue', label='tl', alpha=alpha)
                ax.scatter(t_s_tsne[div_size:2 * div_size, 0], t_s_tsne[div_size:2 * div_size, 1], c='red',
                           label='te_hat',
                           alpha=alpha)
                ax.scatter(t_s_tsne[2 * div_size:3 * div_size, 0], t_s_tsne[2 * div_size:3 * div_size, 1], c='green',
                           label='tr', alpha=alpha)
                ax.scatter(t_s_tsne[3 * div_size:4 * div_size, 0], t_s_tsne[3 * div_size:4 * div_size, 1], c='yellow',
                           label='tr_hat', alpha=alpha)

                ax.set_title('target_domain_seq')
                # ax.set_title('feature')
                ax.legend()
                plt.savefig('{}/target_seq'.format(save_t_dir))
                self._run_wandb.log({
                    "tsne/{}_epoch/target_seq".format(self.cnt): [wandb.Image(
                        "{}/target_seq".format(save_t_dir) + '.png')]
                }, step=self._tr_cnt)

                plt.close()

        if plotting == True:
            save_tl_dir = self._log_dir + '/task_infer/epoch_{}'.format(self.cnt)
            os.makedirs(save_tl_dir)
            # save_label_dir = self._log_dir + '/label/recon/in_{}'.format((self.cnt))
            # os.makedirs(save_label_dir)

            tl_batch = agent_buffer.get_random_batch(500, re_eval_rw=False)
            tl_ims = tl_batch['ims']
            # se_batch = self._exp_buff.get_random_batch(500)
            # se_ims = scale * se_batch['ims']
            se_batch =self._exp_buff.get_random_batch(500)
            se_ims = se_batch['ims']
            tr_batch = self._pr_age_buff.get_random_batch(500)
            tr_ims = tr_batch['ims']
            sr_batch = self._pr_exp_buff.get_random_batch(500)
            sr_ims = sr_batch['ims']

            tl_feature = self._pre_s(tl_ims)
            merge_se = self._label_net(self._pre_s(se_ims))
            merge_sr = self._label_net(self._pre_s(sr_ims))
            merge_tl = self._label_net(tl_feature)
            merge_tr = self._label_net(self._pre_s(tr_ims))
            merge_sehat = self._label_net(self._fake_gen(tl_feature))

            name_list = ['True_label_preds']
            se_label_plot = merge_se
            sr_label_plot = merge_sr
            tl_label_plot = merge_tl
            tr_label_plot = merge_tr
            sehat_label_plot = merge_sehat

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            x_values = range(len(se_label_plot))

            alpha=0.25

            ax.scatter(x_values, se_label_plot[:, 0], c='red',
                       label='feature_se', alpha=alpha)
            ax.scatter(x_values, sr_label_plot[:, 0], c='yellow',
                       label='feature_sr', alpha=alpha)
            ax.scatter(x_values, tl_label_plot[:, 0],  c='blue',
                       label='feature_tl', alpha=alpha)
            ax.scatter(x_values, tr_label_plot[:, 0], c='green',
                       label='feature_tr', alpha=alpha)
            ax.scatter(x_values, sehat_label_plot[:, 0], c='black',
                       label='feature_sehat', marker='x',alpha=alpha)
            ax.set_title(name_list[0])
            ax.legend()
            plt.savefig('{}/{}'.format(save_tl_dir, name_list[0]))

            self._run_wandb.log({
                "task_infer/{}_epoch/{}".format(self.cnt, name_list[0]): [wandb.Image(
                    '{}/{}'.format(save_tl_dir, name_list[0]) + '.png')]
            }, step=self._tr_cnt)

            plt.close()

            print("image save complete")

        print("=*"*20)
        print("Preprocessing Independent RL buffer")
        agent_buffer.update_reward()
        print("Preprocessing Done!")
        print("=*"*20)

        print("Start SAC training")
        self.agent.train(agent_buffer, l_batch_size, l_updates, l_act_delay, self.cnt)

        end_time = time.time()
        elapsed_time_load_buf = end_time - start_time
        print(f"1 epoch: {elapsed_time_load_buf} 초")

        print("SAC training complete")

        return 0

