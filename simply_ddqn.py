from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import os
import sys

import chainer
from chainer import optimizers
import gym
from gym import spaces
import gym.wrappers
import numpy as np
import cupy as cp

import chainerrl
from chainerrl.agents import DQN, DoubleDQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer

import norms

class ConvNet(chainer.Chain): 

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(3, 32, ksize =8, stride = 4)
            self.l2 = L.Convolution2D(32, 64, ksize =4 , stride = 2)
            self.l3 = L.Convolution2D(64, 64, ksize =3 , stride = 1)
            self.fc1 = L.Linear(None, 512) 
            self.fc2 = L.Linear(512, 5)
 
    def __call__(self, x, test=False):

        h = self.l1(x)
        h = F.relu(h)
        h = self.l2(h)
        h = F.relu(h)
        h = F.dropout(h, ratio = 0.5)
        h = self.l3(h)
        h = F.relu(h)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='BipedalWalkerHardcore-v2') #MountainCarContinuous-v0, BipedalWalker-v2, BipedalWalkerHardcore-v2,CarRacing-v0
    parser.add_argument('--seed', type=int, default=5,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--load', type=str, default=None) #DDQNBipedalWalker-v2_Run2
    parser.add_argument('--n_episodes', type=int, default=7000)
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--episodic-replay', type=bool, default=False)
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=500)#500 
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--n-hidden-channels', type=int, default=200)#128
    parser.add_argument('--n-hidden-layers', type=int, default=3) 
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4) #1e-4
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)#1e-2
    parser.add_argument('--run_id', type=str, default='_Run1')
    args = parser.parse_args()

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    def clip_action_filter(a):
        return cp.clip(a, action_space.low, action_space.high)

    def make_env(test):
        env = gym.make(args.env)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        # if args.render:
        #     env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space

    if isinstance(action_space, spaces.Box):
        action_size = action_space.low.size
        # Use NAF to apply DQN to continuous action spaces
        # q_func = q_functions.FCQuadraticStateQFunction(
        #     obs_size, action_size,
        #     n_hidden_channels=args.n_hidden_channels,
        #     n_hidden_layers=args.n_hidden_layers,
        #     action_space=action_space)

        q_func = q_functions.FCBNQuadraticStateQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            action_space=action_space)

        # q_func = q_functions.FCLSTMQuadraticStateQFunction(
        #     obs_size, action_size,
        #     n_hidden_channels=args.n_hidden_channels,
        #     n_hidden_layers=args.n_hidden_layers,
        #     action_space=action_space)
        # Use the Ornstein-Uhlenbeck process for exploration
        ou_sigma = (action_space.high - action_space.low) * 0.0015
        explorer = explorers.AdditiveOU(sigma=ou_sigma)
        #explorer = norms.DecayAdditiveOU(sigma=0.5, end_sigma=ou_sigma, decay_steps=1e4)

    else:
        n_actions = action_space.n
        q_func = q_functions.FCStateQFunctionWithDiscreteAction(
            obs_size, n_actions,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)

    q_func.to_gpu(args.gpu)
    opt = optimizers.Adam(args.lr)
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if args.episodic_replay:
        if args.minibatch_size is None:
            args.minibatch_size = 4
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.EpisodicReplayBuffer(rbuf_capacity)
    else:
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    agent = DoubleDQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_interval=args.target_update_interval,
                update_interval=args.update_interval,
                minibatch_size=args.minibatch_size,
                target_update_method=args.target_update_method,
                soft_update_tau=args.soft_update_tau,
                episodic_update=args.episodic_replay, episodic_update_len=16)
    
    if args.load is not None:
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
    agent.save('DDQN' + args.env + args.run_id)

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

    print("Finished")
 

if __name__ == '__main__':
    main()