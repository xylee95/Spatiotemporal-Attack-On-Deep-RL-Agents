from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import copy
from logging import getLogger

import chainer
from chainer import cuda
import chainer.functions as F

import chainerrl
from chainerrl import agent
from chainerrl.misc.batch_states import batch_states
from chainerrl.misc.copy_param import synchronize_parameters
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_reset
from chainerrl.replay_buffer import batch_experiences
from chainerrl.replay_buffer import ReplayUpdater

from chainerrl.agents.dqn import DQN
from chainerrl.recurrent import state_kept

from chainerrl.action_value import DiscreteActionValue
from chainerrl.action_value import DistributionalDiscreteActionValue
from chainerrl.action_value import QuadraticActionValue
from chainerrl.functions.lower_triangular_matrix import lower_triangular_matrix
from chainerrl.links.mlp import MLP
from chainerrl.links.mlp_bn import MLPBN
from chainerrl.q_function import StateQFunction
from chainerrl.recurrent import RecurrentChainMixin


def scale_by_tanh(x, low, high):
    xp = cuda.get_array_module(x.array)
    scale = (high - low) / 2
    scale = xp.expand_dims(xp.asarray(scale, dtype=np.float32), axis=0)
    mean = (high + low) / 2
    mean = xp.expand_dims(xp.asarray(mean, dtype=np.float32), axis=0)
    return F.tanh(x) * scale + mean

class DDQN_Adversary(DQN):
    """Deep Q-Network algorithm.
    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """
    saved_attributes = ('model','target_model', 'optimizer')

    def __init__(self, q_function, optimizer, replay_buffer, gamma,
                 explorer, gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__),
                 batch_states=batch_states):

        self.model = q_function
        self.adv_model = q_function # To load same weights
        self.q_function = q_function  # For backward compatibility

        if gpu is not None and gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu(device=gpu)

        self.xp = self.model.xp
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ('mean', 'sum')
        self.logger = logger
        self.batch_states = batch_states
        if episodic_update:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

        self.t = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()
        # For backward compatibility
        self.target_q_function = self.target_model
        self.average_q = 0
        self.average_q_decay = average_q_decay
        self.average_loss = 0
        self.average_loss_decay = average_loss_decay

        # Error checking
        if (self.replay_buffer.capacity is not None and
                self.replay_buffer.capacity <
                self.replay_updater.replay_start_size):
            raise ValueError(
                'Replay start size cannot exceed '
                'replay buffer capacity.')


    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch['next_state']

        with chainer.using_config('train', False), state_kept(self.q_function):
            next_qout = self.q_function(batch_next_state)

        target_next_qout = self.target_q_function(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def act_forward(self, obs, adv_action):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(self.batch_states([obs], self.xp, self.phi))
            q = float(action_value.max.array)
            action = cuda.to_cpu(action_value.greedy_actions.array)[0]

            #Q(s,a|theta) =  A(s,a|theta) + V(s|theta) as a NAF
            #V(s|theta) is constant
            #A(s,a|theta) = -1/2 (a - mu(s))^T P(s) (a - mu(s))
            #Only u is changing in the equation above to compute advantage

            v = action_value.v
            mat = action_value.mat
            mu = action_value.mu
            q_adv = QuadraticActionValue(mu ,mat ,v, [-1 -1], [1, 1]).evaluate_actions(adv_action)
            q_adv = q_adv.data[0]

        # Update stats
        self.average_q *= self.average_q_decay
        self.average_q += (1 - self.average_q_decay) * q

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        
        return action, q, q_adv

