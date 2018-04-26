#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import synthetic_data
import visualize
import matplotlib.pyplot as plt


def dtanh(x):
    return 1.0 - tf.square(tf.tanh(x))


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
            # initializer = tf.contrib.layers.xavier_initializer_conv2d()
            # const_initializer = tf.constant_initializer(value=0.5, dtype=tf.float32)
            initializer = tf.truncated_normal_initializer()
            self.u = tf.get_variable('u', initializer=initializer(shape=(z_dim,)))
            self.w = tf.get_variable('w', initializer=initializer(shape=(z_dim,)))
            self.b = tf.get_variable('b', initializer=initializer(shape=()))

    def __call__(self, z, name='flow'):
        """

        :param z:  B*z_dim
        :param name:
        :return:
        """
        with tf.name_scope(name):
            a = self.h(tf.reduce_sum(tf.multiply(z, self.w), axis=1, keep_dims=True) + self.b)  # B*1
            psi = dtanh(a) * tf.transpose(tf.expand_dims(self.w, axis=1))  # B*z

            # A.1, this will guarantee that the planar flow is invertible
            x = tf.reduce_sum(tf.multiply(self.w, self.u))
            m = -1 + tf.nn.softplus(x)
            u_h = self.u + (m - x) * self.w / tf.reduce_sum(tf.square(self.w))

            z = z + a * tf.transpose(tf.expand_dims(u_h, axis=1))  # B*z

            # compute log_det
            log_det = tf.log(tf.abs(1. + tf.reduce_sum(tf.multiply(psi, u_h), axis=1, keep_dims=True)))  # B*1

            return z, log_det


class NormalizingFlow(object):
    """
    Normalizing flow
    """
    def __init__(self, z_dim, K=3, name='normalizingflow'):
        self.z_dim = z_dim
        self.K = K
        self.planar_flows = []
        self.log_dets = []
        with tf.variable_scope(name):
            for i in range(K):
                flow = PlanarFlow(z_dim, var_scope='planarflow_' + str(i+1))
                self.planar_flows.append(flow)

    def __call__(self, z, name='normflow'):
        with tf.name_scope(name):
            for flow in self.planar_flows:
                z, log_det = flow(z)
                self.log_dets.append(log_det)

        log_det_sum = tf.concat(self.log_dets, axis=1)
        sum_log_det = tf.reduce_sum(log_det_sum, axis=1)
        return z, sum_log_det


# TODO: construct graph of norm flow


# show data
print("show synethtic data")
# visualize.plot_density()


K = 32
L = 500
steps = 5000
is_training = True
learning_rate = 0.001

# we need to train 4 models for each U{i} distribution

U1 = getattr(synthetic_data, 'U1_tf')

input_z0_ph = tf.placeholder(tf.float32, [None, 2])
log_q0_ph = tf.placeholder(tf.float32, [None])

normFlow = NormalizingFlow(z_dim=2, K=K)
zk, sum_log_det = normFlow(input_z0_ph)


def compute_loss(U_func, sum_log_det, log_q0, z_k):
    U_z = U_func(z_k)
    U_z = tf.clip_by_value(U_z, -10000, 10000)
    # U_z = tf.Print(U_z, [U_z], first_n=-1, message='U_z')
    # sum_log_det = tf.Print(sum_log_det, [sum_log_det], first_n=-1, message='sum_log_det')
    kld = log_q0 - sum_log_det + U_z
    kld = tf.reduce_mean(kld)
    return kld


loss = compute_loss(U1, sum_log_det, log_q0_ph, zk)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


sampler = synthetic_data.normal_sampler()

for step in range(steps):
    z0, log_q0 = sampler(L)
    l, _ = sess.run([loss, train_op], feed_dict={input_z0_ph: z0, log_q0_ph: log_q0})
    print("step {}, loss={}".format(step, l))


# space = np.linspace(-5, 5, 500)
# X, Y = np.meshgrid(space, space)
# shape = X.shape
# X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
# Z = np.concatenate([X_flatten, Y_flatten], 1)
#
#
# Z_placeholder = tf.placeholder(tf.float32, [None, 2])
#
# U1_output = synthetic_data.U1_tf(Z_placeholder)
#
# output = sess.run(U1_output, feed_dict={Z_placeholder:Z})
#
# print(output)

# sample from the trained model
fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
ax = axes[0]

side = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(side, side)
counts = np.zeros(X.shape)
p = np.zeros(X.shape)

size = [-500, 500]
num_side = 500

L = 100000
print("Sampling", end='')
for i in range(10):
    print('.', end='')
    z, logq = sampler(L)
    z_k, logq_k = sess.run([zk, sum_log_det], feed_dict={input_z0_ph: z, log_q0_ph: logq})
    logq_k = logq - logq_k
    q_k = np.exp(logq_k)
    z_k = (z_k - size[0]) * num_side / (size[1] - size[0])
    for l in range(L):
        x, y = int(z_k[l, 1]), int(z_k[l, 0])
        if 0 <= x and x < num_side and 0 <= y and y < num_side:
            counts[x, y] += 1
            p[x, y] += q_k[l]

print()
counts = np.maximum(counts, np.ones(counts.shape))
p /= counts
p /= np.sum(p)
Y = -Y
ax.pcolormesh(X, Y, p)

fig.tight_layout()
plt.show()
