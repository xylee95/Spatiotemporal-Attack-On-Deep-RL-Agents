"""An example of training PPO against OpenAI Gym Envs.
This script is an example of training a PPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.
To solve CartPole-v0, run:
    python train_ppo_gym.py --env CartPole-v0
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Sequential
import gym
import gym.wrappers
import numpy as np
import cupy as cp
from functools import partial

import chainerrl
from chainerrl.agents import a3c
from chainerrl.agents import PPO
from chainerrl import links
from chainerrl import distribution
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies
from chainerrl.initializers import LeCunNormal
from chainerrl.policy import Policy


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, obs_size, action_space,
                 n_hidden_layers=3, n_hidden_channels=512,
                 bound_mean=True):
        assert bound_mean in [False, True]
        super().__init__()
        hidden_sizes = (n_hidden_channels,) * n_hidden_layers
        with self.init_scope():
            self.pi = policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_size, action_space.low.size,
                n_hidden_layers, n_hidden_channels,
                var_type='diagonal', nonlinearity=F.tanh,
                bound_mean=bound_mean,
                min_action=action_space.low, max_action=action_space.high,
                mean_wscale=1e-2)
            self.v = links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

class ConvFCGaussianPolicyWithStateIndependentCovariance(chainer.Chain, Policy):
    """Gaussian policy that consists of FC layers with parametrized covariance.
    This model has one output layers: the mean layer.
    The mean of the Gaussian is computed in the same way as FCGaussianPolicy.
    Args:
        n_input_channels (int): Number of input channels.
        action_size (int): Number of dimensions of the action space.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        min_action (ndarray): Minimum action. Used only when bound_mean=True.
        max_action (ndarray): Maximum action. Used only when bound_mean=True.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        nonlinearity (callable): Nonlinearity placed between layers.
        mean_wscale (float): Scale of weight initialization of the mean layer.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(self, n_input_channels, action_size,
                 min_action=None, max_action=None, bound_mean=False,
                 var_type='spherical',
                 nonlinearity=F.relu,
                 pooling=F.max_pooling_2d,
                 mean_wscale=1,
                 var_func=F.softplus,
                 var_param_init=0,
                 ):

        self.n_input_channels = n_input_channels
        self.action_size = action_size
        self.min_action = min_action
        self.max_action = max_action
        self.bound_mean = bound_mean
        self.nonlinearity = nonlinearity
        self.var_func = var_func
        self.pooling = pooling
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]

        if self.bound_mean:
            layers.append(lambda x: bound_by_tanh(
                x, self.min_action, self.max_action))

        super().__init__()
        with self.init_scope():
            self.hidden_layers = Sequential(
                L.Convolution2D(3, 8, ksize=4),
                self.nonlinearity,
                partial(F.max_pooling_2d, ksize=2),

                L.Convolution2D(8, 16, ksize=3),
                self.nonlinearity,
                partial(F.max_pooling_2d, ksize=2),

                L.Convolution2D(16, 64, ksize=3),
                self.nonlinearity,
                L.Linear(None, 256),
                # self.nonlinearity,
                # L.Linear(512, 256),
                self.nonlinearity,
                L.Linear(256, action_size, initialW=LeCunNormal(mean_wscale))
                )
            self.var_param = chainer.Parameter(
                initializer=var_param_init, shape=(var_size,))

    def __call__(self, x):
        x = cp.rollaxis(x, 3, 1)
        mean = self.hidden_layers(x)
        var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        return distribution.GaussianDistribution(mean, var)

class ConvCritic(chainer.Chain): 

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(3, 32, ksize = 8, stride = 2, pad = 0)
            self.b1 = L.BatchNormalization(32)
            self.l2 = L.Convolution2D(32, 32, ksize = 4, stride = 1,  pad = 0)
            self.b2 = L.BatchNormalization(32)
            self.l3 = L.Convolution2D(32, 32, ksize = 2, stride = 1,  pad = 0)
            self.b3 = L.BatchNormalization(32)
            self.fc1 = L.Linear(None, 128) 
            self.fc2 = L.Linear(128, 1)

    def __call__(self, x, test=False):
        
        x = cp.rollaxis(x, 3, 1)

        h = self.l1(x)
        h = F.max_pooling_2d(h, ksize=2, stride=1)
        #h = self.b1(h)
        h = F.tanh(h)

        h = self.l2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=1)
        #h = self.b2(h)
        h = F.tanh(h)

        # h = self.l3(h)
        # h = self.b3(h)
        # h = F.tanh(h)

        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        return h

class A3CFFConvGaussian(chainer.Chain, a3c.A3CModel):

    def __init__(self, obs_size, action_space,
                 bound_mean=None):
        assert bound_mean in [False, True]
        super().__init__()
        with self.init_scope():
            self.pi = ConvFCGaussianPolicyWithStateIndependentCovariance(
                n_input_channels=obs_size, action_size=action_space.low.size,
                var_type='diagonal', nonlinearity=F.tanh,
                bound_mean=bound_mean,
                min_action=action_space.low, max_action=action_space.high,
                mean_wscale=1e-2)
            self.v = ConvCritic()

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--env', type=str, default='BipedalWalkerHardcore-v2') #MountainCarContinuous-v0, LunarLanderContinuous-v2, BipedalWalker-v2, BipedalWalkerHardcore-v2, CarRacing-v0
    parser.add_argument('--arch', type=str, default='FFGaussian',
                        choices=('FFSoftmax', 'FFMellowmax',
                                 'FFGaussian', 'FFConvGaussian'))
    parser.add_argument('--bound-mean', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--n_episodes', type=int, default=5000)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2) #default is 1e-2
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--load', type=str, default='') #PPOBipedalWalker-v2_Run2
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--update-interval', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.0275)
    parser.add_argument('--run_id', type=str, default='_Run1')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def make_env(args_env,test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        # if args.render:
        #     env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(args.env, test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    action_size = action_space.low.size

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    # Switch policy types accordingly to action space types
    if args.arch == 'FFSoftmax':
        model = A3CFFSoftmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A3CFFMellowmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFGaussian':
        model = A3CFFGaussian(obs_space.low.size, action_space,
                              bound_mean=args.bound_mean)
    elif args.arch == 'FFConvGaussian':
        model = A3CFFConvGaussian(obs_space.low.size, action_space,
                              bound_mean=args.bound_mean)
        obs_normalizer = None

    if args.gpu > -1:
        model.to_gpu(args.gpu)

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)

    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = PPO(model, opt,
                obs_normalizer=obs_normalizer, gpu=args.gpu,
                update_interval=args.update_interval, 
                minibatch_size=args.batchsize, epochs=args.epochs,
                clip_eps_vf=None, entropy_coef=args.entropy_coef,
                standardize_advantages=args.standardize_advantages,
                )
    if len(args.load) > 0:
        agent.load(args.load)

    print('Environment Time Step Limit', timestep_limit)
    for i in range(1, args.n_episodes + 1):
        obs = env.reset()
        reward = 0
        done = False
        R = 0  
        t = 0  
        while not done and t < timestep_limit:
            #env.render()
            action = agent.act_and_train(obs, reward)     
            obs, reward, done, _ = env.step(action)
            R += reward
            t += 1
        if i % 10 == 0:
            print('episode:', i,
                  'R:', R,
                  'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)
    print('Finished.')

    for i in range(10):
        obs = env.reset()
        done = False
        R = 0
        t = 0
        while not done and t < timestep_limit:
            env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
        print('test episode:', i, 'R:', R)
        agent.stop_episode()

    agent.save('PPO' + args.env + args.run_id)

    # if args.demo:
    #     env = make_env(True)
    #     eval_stats = experiments.eval_performance(
    #         env=env,
    #         agent=agent,
    #         n_steps=None,
    #         n_episodes=args.eval_n_runs,
    #         max_episode_len=timestep_limit)
    #     print('n_runs: {} mean: {} median: {} stdev {}'.format(
    #         args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
    #         eval_stats['stdev']))
    # else:
    #     # Linearly decay the learning rate to zero
    #     def lr_setter(env, agent, value):
    #         agent.optimizer.alpha = value

    #     lr_decay_hook = experiments.LinearInterpolationHook(
    #         args.steps, args.lr, 0, lr_setter)

    #     # Linearly decay the clipping parameter to zero
    #     def clip_eps_setter(env, agent, value):
    #         agent.clip_eps = max(value, 1e-8)

    #     clip_eps_decay_hook = experiments.LinearInterpolationHook(
    #         args.steps, 0.2, 0, clip_eps_setter)

    #     experiments.train_agent_with_evaluation(
    #         agent=agent,
    #         env=make_env(False),
    #         eval_env=make_env(True),
    #         outdir=args.outdir,
    #         steps=args.steps,
    #         eval_n_steps=None,
    #         eval_n_episodes=args.eval_n_runs,
    #         eval_interval=args.eval_interval,
    #         train_max_episode_len=timestep_limit,
    #         save_best_so_far_agent=False,
    #         step_hooks=[
    #             lr_decay_hook,
    #             clip_eps_decay_hook,
    #         ],
    #     )

if __name__ == '__main__':
    main()