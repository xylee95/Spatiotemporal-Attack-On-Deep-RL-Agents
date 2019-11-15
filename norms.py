# functions to compute norms and projections
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import numpy as np
import math
from numpy import linalg as LA

import cupy as cp
from cupy import linalg as LA_GPU

import chainerrl
from chainerrl import explorers
from logging import getLogger


def reducebudget_l1(used, budget):
    new_budget = max(0, budget - used)
    return new_budget


def reducebudget_l2(used, budget):
    new_budget = math.sqrt(budget**2 - used**2)
    new_budget = max(0, new_budget)
    # print('used:',used)
    # print('remainder',new_budget)
    return new_budget


def random_delta(n_dim, budget):
    real_budget = np.random.uniform(low=0, high=budget, size=1)
    x = np.random.randint(low=-100, high=100, size=n_dim)
    t = sum(np.abs(item) for item in x)
    x = [item / t for item in x]
    x = [item * real_budget for item in x]
    x = [item[0] for item in x]
    return x


def l1_time_project2(y_orig, budget):

    y_abs = list(map(abs, y_orig))
    u = sorted(y_abs, reverse=True)
    binK = 1
    K = 1
    bin_list = [0] * len(u)
    for i in range(1, len(u) + 1):
        if (sum([u[r] for r in range(i)]) - budget) / i < u[i - 1]:
            bin_list[i - 1] = binK
            binK += 1

    if sum(bin_list) > 0:
        K = np.argmax(bin_list) + 1

    tau = (sum([u[i] for i in range(K)]) - budget) / K
    xn = [max(item - tau, 0) for item in y_abs]
    l1_norm_y = np.linalg.norm(y_orig, 1)
    for i in range(len(y_orig)):
        if l1_norm_y > budget:
            y_orig[i] = np.sign(y_orig[i]) * xn[i]
    return y_orig


def l1_spatial_project2(y_orig, budget):

    y_abs = list(map(abs, y_orig))
    u = sorted(y_abs, reverse=True)
    binK = 1
    K = 1
    bin_list = [0] * len(u)
    for i in range(1, len(u) + 1):
        if (sum([u[r] for r in range(i)]) - budget) / i < u[i - 1]:
            bin_list[i - 1] = binK
            binK += 1

    if sum(bin_list) > 0:
        K = np.argmax(bin_list) + 1

    tau = (sum([u[i] for i in range(K)]) - budget) / K
    xn = [max(item - tau, 0) for item in y_abs]
    l1_norm_y = np.linalg.norm(y_orig, 1)
    for i in range(len(y_orig)):
        if l1_norm_y > budget:
            y_orig[i] = np.sign(y_orig[i]) * xn[i]
    return y_orig


def l1_time_project(y_orig, delta):
    y = list(map(abs, y_orig))
    v = [y[0]]
    vb = []
    rho = y[0] - delta
    for i in range(2, len(y)):
        if y[i] > rho:
            rho = rho + (y[i] - rho) / (np.linalg.norm(v, 1) + 1)
            if rho > y[i] - delta:
                v.append(y[i])
            else:
                vb = vb + v
                v = [y[i]]
                rho = y[i] - delta

    if len(vb) > 0:
        for item in vb:
            if item > rho:
                v.append(item)
                rho = rho + (item - rho) / np.linalg.norm(v, 1)

    v_proj = 0
    while not v_proj == np.linalg.norm(v, 1):
        v_proj = np.linalg.norm(v, 1)
        for i, item in enumerate(v):
            if item <= rho:
                y_loc = v.pop(i)
                rho = rho + (rho - y_loc) / np.linalg.norm(v, 1)

    tau = rho
    xn = [max(item - tau, 0) for item in y]
    l1_norm_y = np.linalg.norm(y_orig, 1)
    for i in range(len(y_orig)):
        if l1_norm_y > delta:
            y_orig[i] = np.sign(y_orig[i]) * xn[i]
    return y_orig


def l2_spatial_norm(x):
    return LA.norm(x, 2)


def l1_spatial_norm(x):
    return LA.norm(x, 1)


def l2_spatial_norm_gpu(x):
    return LA_GPU.norm(x, 2)


def l2_time_norm(x):
    return LA.norm(x, 2)


def linf_spatial_norm(x):
    return LA.norm(x, np.inf)


def linf_time_norm(x):
    return LA.norm(x, np.inf)


def l2_spatial_project(x, distance):
    norm = l2_spatial_norm(x)
    # print('x',x)
    # print('l2 norm', diff)
    # print('dist',distance)
    if norm <= distance:
        delta = x
    else:
        delta = (x / norm) * distance
    return delta


def l2_spatial_project_gpu(x, distance):
    norm = l2_spatial_norm_gpu(x)
    # print('x',x)
    # print('l2 norm', diff)
    if norm <= distance:
        delta = x
    else:
        delta = (x / norm) * distance
    return delta


def l2_time_project(x, distance):
    norm = l2_time_norm(x)
    # print('x',x)
    # print('l2 norm', diff)
    # print('dist',distance)
    if norm <= distance:
        delta = x
    else:
        delta = (x / norm) * distance
    return delta


def linf_spatial_project(x, distance):
    # print('x', x)
    norm = linf_spatial_norm(x)
    # print('norm', norm)
    if norm <= distance:
        x_hat = x
    else:
        x_hat = np.clip(x, -distance, distance)
    # print('xhat',x_hat)
    return x_hat


def linf_time_project(x, distance):
    # print('x', x)
    norm = linf_time_norm(x)
    if norm <= distance:
        x_hat = x
    else:
        x_hat = np.clip(x, -distance, distance)
    # print('xhat',x_hat)
    return x_hat


class DecayAdditiveOU(explorers.AdditiveOU):
    """Additive Ornstein-Uhlenbeck process.
Used in https://arxiv.org/abs/1509.02971 for exploration.
Args:
    mu (float): Mean of the OU process
    theta (float): Friction to pull towards the mean
    sigma (float or ndarray): Scale of noise
    start_with_mu (bool): Start the process without noise
"""

    def __init__(self, mu=0.0, theta=0.15, sigma=0.3, start_with_mu=False,
                 end_sigma=0.2, decay_steps=1e3, logger=getLogger(__name__)):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.start_with_mu = start_with_mu
        self.logger = logger
        self.ou_state = None

        self.steps = 0
        self.decay_steps = decay_steps
        self.delta = (self.sigma - end_sigma) / decay_steps

    def decay_sigma(self):
        self.sigma = self.sigma - self.delta
        return self.sigma

    def select_action(self, t, greedy_action_func, action_value=None):
        if self.steps < self.decay_steps:
            self.sigma = self.decay_sigma()
            self.steps += 1

        a = greedy_action_func()
        if self.ou_state is None:
            if self.start_with_mu:
                self.ou_state = np.full(a.shape, self.mu, dtype=np.float32)
            else:
                sigma_stable = (self.sigma
                                / np.sqrt(2 * self.theta - self.theta ** 2))
                self.ou_state = np.random.normal(
                    size=a.shape, loc=self.mu, scale=sigma_stable).astype(np.float32)
        else:
            self.evolve()
        noise = self.ou_state
        self.logger.debug('t:%s noise:%s', t, noise)
        return a + noise


if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, -5, 0]
    x = random_delta(2, 1)
    # print(l1_norm(x))
    # print(l2_norm(x))
    # print(linf_norm(x))

    # x_hat = l1_project(x, 5)
    # print(x_hat)

    # x_hat = l2_project(x, 5)
    # print(x_hat)

    # x_hat = linf_project(x, 6)
    # print(x_hat)
