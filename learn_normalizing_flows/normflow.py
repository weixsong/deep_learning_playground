#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import synthetic_data
import visualize
import matplotlib.pyplot as plt


class PlanarFlow(object):
    """
    Planar normalizing flow
    equation 10-12, 21-23 in paper https://arxiv.org/pdf/1505.05770.pdf
    """

    def __init__(self, z_dim=2, var_scope='planarflow'):
        self.z_dim = z_dim
        self.h = tf.tanh
        self.var_scope = var_scope

        with tf.variable_scope(var_scope):
            initializer = tf.truncated_normal_initializer(stddev=z_dim)
            self.u = tf.get_variable('u', initializer=initializer(shape=(z_dim, 1)))
            self.w = tf.get_variable('w', initializer=initializer(shape=(z_dim, 1)))
            self.b = tf.get_variable('b', initializer=initializer(shape=(1, 1)))

    def __call__(self, z, logp, name='flow'):
        """
        :param z:  B*z_dim
        :param name:
        :return:
        """
        with tf.name_scope(name):
            a = self.h(tf.matmul(z, self.w) + self.b)
            psi = tf.matmul(1 - a ** 2, tf.transpose(self.w))

            # Section A.1, try to make the transformation invertible
            x = tf.matmul(tf.transpose(self.w), self.u)
            m = -1 + tf.nn.softplus(x)
            u_h = self.u + (m - x) * self.w / (tf.matmul(tf.transpose(self.w), self.w))

            logp = logp - tf.squeeze(tf.log(1 + tf.matmul(psi, u_h)))
            z = z + tf.matmul(a, tf.transpose(u_h))

            return z, logp


class NormalizingFlow(object):
    """
    Normalizing flow
    """
    def __init__(self, z_dim, K=3, name='normalizingflow'):
        self.z_dim = z_dim
        self.K = K
        self.planar_flows = []
        with tf.variable_scope(name):
            for i in range(K):
                flow = PlanarFlow(z_dim, var_scope='planarflow_' + str(i+1))
                self.planar_flows.append(flow)

    def __call__(self, z, logp, name='normflow'):
        with tf.name_scope(name):
            for flow in self.planar_flows:
                z, logp = flow(z, logp)

        return z, logp


# class NormalizingFlowTrainer(object):
#     """
#     Trainer for normalizing flow
#     """
#     def __init__(self, sess, U_func, L=256, steps=100000, lr=0.001, is_training=True):
#         self.sess = sess
#         self.lr = lr
#         self.is_training = is_training
#         self.sampler = synthetic_data.normal_sampler()
#         self.L = L
#         self.steps = steps
#         self.U_func = U_func
#         self.name = U_func.__name__
#
#     def compute_loss(self, logqk, z_k):
#         with tf.name_scope(self.name):
#             U_z = self.U_func(z_k)
#             U_z = tf.clip_by_value(U_z, -10000, 10000)
#             kld = logqk + U_z
#             kld = tf.reduce_mean(kld)
#         return kld
#
#     def build_network(self):
#         with tf.name_scope(self.name):
#             self.input = tf.placeholder(tf.float32, [None, 2])
#             self.logq0 = tf.placeholder(tf.float32, [None])
#
#             normFlow = NormalizingFlow(z_dim=z_dim, K=K)
#             zk, logqk = normFlow(self.input, self.logq0)
#
#             self.zk = zk
#             self.logqk = logqk
#             self.loss = self.compute_loss(logqk, zk)
#
#     def train(self):
#         print('training function {}'.format(self.U_func.__name__))
#         train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
#         for step in range(self.steps):
#             z0, log_q0 = self.sampler(self.L)
#             l, _ = sess.run([self.loss, train_op], feed_dict={self.input: z0, self.logq0: log_q0})
#             if step % 100 == 0:
#                 print("step {}, loss={}".format(step, l))



def build_network(K=32, z_dim=2):
    input = tf.placeholder(tf.float32, [None, 2])
    logq0 = tf.placeholder(tf.float32, [None])

    normFlow = NormalizingFlow(z_dim=z_dim, K=K)
    zk, logqk = normFlow(input, logq0)
    return zk, logqk


def compute_loss(U_func, sum_log_det, z_k):
    U_z = U_func(z_k)
    U_z = tf.clip_by_value(U_z, -10000, 10000)
    kld = sum_log_det + U_z
    kld = tf.reduce_mean(kld)
    return kld


def save(saver, sess, logdir, step, write_meta=False):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))

    # change here
    if not tf.gfile.Exists(logdir):
        tf.gfile.MakeDirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=write_meta)
    print('Save Model Done.')


def save_image(sess, zk, logqk, input_z0_placeholder, log_q0_placehoder, sampler, path):
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    ax = axes[0]

    side = np.linspace(-5, 5, 500)
    X, Y = np.meshgrid(side, side)
    counts = np.zeros(X.shape)
    p = np.zeros(X.shape)

    size = [-500, 500]
    num_side = 500

    L = 100
    print("Sampling", end='')
    for i in range(1000):
        z, logq = sampler(L)
        z_k, logq_k = sess.run([zk, logqk], feed_dict={input_z0_placeholder: z, log_q0_placehoder: logq})
        logq_k = logq_k
        q_k = np.exp(logq_k)
        z_k = (z_k - size[0]) * num_side / (size[1] - size[0])
        for l in range(L):
            x, y = int(z_k[l, 1]), int(z_k[l, 0])
            if 0 <= x < num_side and 0 <= y < num_side:
                counts[x, y] += 1
                p[x, y] += q_k[l]

    counts = np.maximum(counts, np.ones(counts.shape))
    p /= counts
    p /= np.sum(p)
    Y = -Y
    ax.pcolormesh(X, Y, p)

    fig.tight_layout()
    plt.savefig(path)


if __name__ == '__main__':
    # show data
    print("show synethtic data, close the data image and continue")
    visualize.plot_density()

    K = 32
    z_dim = 2
    L = 256
    steps = 4000000
    is_training = False
    learning_rate = 0.001
    save_model_every_steps = 10000
    logdir = './log/'
    logdir_image = './log/image/'
    checkpoint = r'model.ckpt-3980000'

    if not tf.gfile.Exists(logdir_image):
        tf.gfile.MakeDirs(logdir_image)

    U1 = getattr(synthetic_data, 'U1_tf')
    input_z0_placeholder = tf.placeholder(tf.float32, [None, 2])
    log_q0_placehoder = tf.placeholder(tf.float32, [None])

    normFlow = NormalizingFlow(z_dim=2, K=K)
    zk, logqk = normFlow(input_z0_placeholder, log_q0_placehoder)
    loss = compute_loss(U1, logqk, zk)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables())

    # TODO: restore from
    if not is_training:
        # restore model from checkpoint
        saver.restore(sess, checkpoint)
        print('Model restore successfully!')

    sampler = synthetic_data.normal_sampler()
    if is_training:
        for step in range(steps):
            z0, log_q0 = sampler(L)
            l, _ = sess.run([loss, train_op], feed_dict={input_z0_placeholder: z0, log_q0_placehoder: log_q0})
            if step % 1000 == 0:
                print("step {}, loss={}".format(step, l))

            if step % save_model_every_steps == 0:
                save(saver, sess, logdir, step, write_meta=False)
                path = os.path.join(logdir_image, str(step) + '.png')
                save_image(sess, zk, logqk, input_z0_placeholder, log_q0_placehoder, sampler, path)

    save(saver, sess, logdir, steps, write_meta=False)
    print("done!")
    path = os.path.join(logdir_image, 'final.png')
    save_image(sess, zk, logqk, input_z0_placeholder, log_q0_placehoder, sampler, path)
