#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# synthetic data in table 1 of paper: https://arxiv.org/abs/1505.05770


def w1(z):
    return np.sin(2 * np.pi * z[:, 0] / 4)


def w2(z):
    return 3 * np.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)


def w3(z):
    return 3 * (1.0 / (1 + np.exp(-(z[:, 0] - 1) / 0.3)))


def U1(z):
    add1 = 0.5 * ((np.linalg.norm(z, 2, 1) - 2) / 0.4) ** 2
    add2 = -np.log(np.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + np.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2))
    return add1 + add2


def U2(z):
    return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2


def U3(z):
    in1 = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    in2 = np.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return -np.log(in1 + in2)


def U4(z):
    in1 = np.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    in2 = np.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return -np.log(in1 + in2)


def normal_sampler(mean=np.zeros(2), sigma=np.ones(2)):
    dim = mean.shape[0]

    def sampler(N):
        z = mean + np.random.randn(N, dim) * sigma
        logq = -0.5 * np.sum(2 * np.log(sigma) + np.log(2 * np.pi) + ((z - mean) / sigma) ** 2, 1)
        return z, logq

    return sampler
