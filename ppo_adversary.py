from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import collections
import copy
import itertools

import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np

from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.agents import a3c
from chainerrl.agents import PPO

import cupy as cp


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def _elementwise_clip(x, x_min, x_max):
    """Elementwise clipping

    Note: chainer.functions.clip supports clipping to constant intervals
    """
    return F.minimum(F.maximum(x, x_min), x_max)

class PPO_Adversary(PPO):
    """Proximal Policy Optimization

    See https://arxiv.org/abs/1707.06347

    Args:
        model (A3CModel): Model to train.  Recurrent models are not supported.
            state s  |->  (pi(s, _), v(s))
        optimizer (chainer.Optimizer): Optimizer used to train the model
        gpu (int): GPU device id if not None nor negative
        gamma (float): Discount factor [0, 1]
        lambd (float): Lambda-return factor [0, 1]
        phi (callable): Feature extractor function
        value_func_coef (float): Weight coefficient for loss of
            value function (0, inf)
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        update_interval (int): Model update interval in step
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        clip_eps (float): Epsilon for pessimistic clipping of likelihood ratio
            to update policy
        clip_eps_vf (float): Epsilon for pessimistic clipping of value
            to update value function. If it is ``None``, value function is not
            clipped on updates.
        standardize_advantages (bool): Use standardized advantages on updates
        value_stats_window (int): Window size used to compute statistics
            of value predictions.
        entropy_stats_window (int): Window size used to compute statistics
            of entropy of action distributions.
        value_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the value function.
        policy_loss_stats_window (int): Window size used to compute statistics
            of loss values regarding the policy.

    Statistics:
        average_value: Average of value predictions on non-terminal states.
            It's updated on (batch_)act_and_train.
        average_entropy: Average of entropy of action distributions on
            non-terminal states. It's updated on (batch_)act_and_train.
        average_value_loss: Average of losses regarding the value function.
            It's updated after the model is updated.
        average_policy_loss: Average of losses regarding the policy.
            It's updated after the model is updated.
    """

    saved_attributes = ['model', 'optimizer', 'obs_normalizer']

    def __init__(self,
                 model,
                 optimizer,
                 obs_normalizer=None,
                 gpu=None,
                 gamma=0.99,
                 lambd=0.95,
                 phi=lambda x: x,
                 value_func_coef=1.0,
                 entropy_coef=0.01,
                 update_interval=2048,
                 minibatch_size=64,
                 epochs=10,
                 clip_eps=0.2,
                 clip_eps_vf=None,
                 standardize_advantages=True,
                 batch_states=batch_states,
                 value_stats_window=1000,
                 entropy_stats_window=1000,
                 value_loss_stats_window=100,
                 policy_loss_stats_window=100,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to_gpu(device=gpu)

        self.gamma = gamma
        self.lambd = lambd
        self.phi = phi
        self.value_func_coef = value_func_coef
        self.entropy_coef = entropy_coef
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.clip_eps = clip_eps
        self.clip_eps_vf = clip_eps_vf
        self.standardize_advantages = standardize_advantages
        self.batch_states = batch_states

        self.xp = self.model.xp

        # Contains episodes used for next update iteration
        self.memory = []

        # Contains transitions of the last episode not moved to self.memory yet
        self.last_episode = []
        self.last_state = None
        self.last_action = None

        # Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_state = None
        self.batch_last_action = None

        self.value_record = collections.deque(maxlen=value_stats_window)
        self.entropy_record = collections.deque(maxlen=entropy_stats_window)
        self.value_loss_record = collections.deque(
            maxlen=value_loss_stats_window)
        self.policy_loss_record = collections.deque(
            maxlen=policy_loss_stats_window)

    def act(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]

        return action

    # for option 1 analytical gradients
    # not evaluating adversarial action because option 1 has no metric
    def act_forward(self, obs):
        xp = self.xp
        b_state = self.batch_states([obs], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, vs_pred  = self.model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().data)[0]
            
            # means = []
            # dims = len(action_distrib.mean.data[0])
            # for j in range(dims):
            #     means.append(cp.asnumpy(action_distrib.mean[0][j].data))
            #     # var = np.exp(cp.asnumpy(action_dist.ln_var[0][j].data))
            #     # std_devs.append(np.sqrt(var))

            # means = list(float(elem) for elem in means)
            # action = means

            #original ppo is to max entropy (min(-ent))
            #entropy = action_distrib.entropy

        #loss = vs_pred + loss_prob_ratio + loss_entropy

        return action, action_distrib, vs_pred, #loss

    def _lossfun(self,
                 distribs, vs_pred, log_probs,
                 vs_pred_old, target_log_probs,
                 advs, vs_teacher):

        prob_ratio = F.exp(log_probs - target_log_probs)
        ent = distribs.entropy

        loss_policy = - F.mean(F.minimum(
            prob_ratio * advs,
            F.clip(prob_ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs))

        if self.clip_eps_vf is None:
            loss_value_func = F.mean_squared_error(vs_pred, vs_teacher)
        else:
            loss_value_func = F.mean(F.maximum(
                F.square(vs_pred - vs_teacher),
                F.square(_elementwise_clip(vs_pred,
                                           vs_pred_old - self.clip_eps_vf,
                                           vs_pred_old + self.clip_eps_vf)
                         - vs_teacher)
            ))
        loss_entropy = -F.mean(ent)

        self.value_loss_record.append(float(loss_value_func.array))
        self.policy_loss_record.append(float(loss_policy.array))

        loss = (
            loss_policy
            + self.value_func_coef * loss_value_func
            + self.entropy_coef * loss_entropy
        )

        return loss
