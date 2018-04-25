#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


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
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            const_initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
            self.u = tf.get_variable('u', initializer=initializer(shape=(z_dim,)))
            self.w = tf.get_variable('w', initializer=initializer(shape=(z_dim,)))
            self.b = tf.get_variable('b', initializer=const_initializer(shape=()))

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
            u_h = self.u + (m - x) * self.w / tf.sqrt(tf.reduce_sum(tf.square(self.w)))

            z = z + a * tf.transpose(tf.expand_dims(u_h))  # B*z

            # compute log_det
            log_det = tf.abs(1. + tf.reduce_sum(tf.multiply(psi, self.u_h), axis=1, keep_dims=True))  # B*1

            return z, log_det


# TODO: construct graph of norm flow

