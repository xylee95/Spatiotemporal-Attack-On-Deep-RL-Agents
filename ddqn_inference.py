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
from chainer import functions as F
from chainer import cuda
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
import ddqn_adversary
from ddqn_adversary import DDQN_Adversary
import norms
from operator import add

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env_id', type=str, default='MC')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--episodic-replay', action='store_true')
    parser.add_argument('--prioritized-replay', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1)
    parser.add_argument('--rollout', type=str, default='Random',
                        choices=('Nominal', 'Random','MAS','LAS'))
    parser.add_argument('--start_atk', type=float, default=1)
    parser.add_argument('--clip', type=bool, default=True,
                        help='If set to False, actions will be projected '
                             'based on unclipped nominal action ')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3)
    parser.add_argument('--save_experiments', type=bool, default=False)
    parser.add_argument('--horizon', type=int, default=25)
    parser.add_argument('--budget', type=float, default=1)
    parser.add_argument('--s', type=str, default='l2')
    parser.add_argument('--t', type=str, default='l2')

    args = parser.parse_args()

    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if args.env_id == 'LL':
        env_name = 'LunarLanderContinuous-v2'
        load = 'DDQNLunarLanderContinuous-v2_Run2'
        n_hidden_channels = 128
        n_hidden_layers = 4
    elif args.env_id == 'BW':
        # Not robust walker
        env_name = 'BipedalWalker-v2'
        load = 'DDQNBipedalWalker-v2_Run2'
        n_hidden_channels = 256
        n_hidden_layers = 3
    else:
        print('No model found')

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    def make_env(env_name):
        env = gym.make(env_name)
        # Use different random seeds for train and test envs
        env_seed = args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if isinstance(env.action_space, spaces.Box):
            misc.env_modifiers.make_action_filtered(env, clip_action_filter)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(env_name)
    spy_env = make_env(env_name)
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    obs_size = obs_space.low.size
    action_space = env.action_space
    action_size = action_space.low.size

    # Use NAF to apply DQN to continuous action spaces
    q_func = q_functions.FCQuadraticStateQFunction(
        obs_size, action_size,
        n_hidden_channels=n_hidden_channels,
        n_hidden_layers=n_hidden_layers,
        action_space=action_space)
    # Use the Ornstein-Uhlenbeck process for exploration
    ou_sigma = (action_space.high - action_space.low) * 0.2
    explorer = explorers.AdditiveOU(sigma=ou_sigma)

    q_func.to_gpu(args.gpu)
    opt = optimizers.Adam()
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

    agent = DDQN_Adversary(q_func, opt, rbuf, gpu=args.gpu, gamma=0.995,
                           explorer=explorer, replay_start_size=1000,
                           target_update_interval=100,
                           update_interval=1, minibatch_size=64,
                           target_update_method='hard', soft_update_tau=1e-2,
                           episodic_update=True, episodic_update_len=16)

    spy = DDQN_Adversary(q_func, opt, rbuf, gpu=args.gpu, gamma=0.995,
                         explorer=explorer, replay_start_size=1000,
                         target_update_interval=100,
                         update_interval=1, minibatch_size=64,
                         target_update_method='hard', soft_update_tau=1e-2,
                         episodic_update=True, episodic_update_len=16)

    agent.load(load)
    spy.load(load)

    def compute_grad(q, adv_q, action, adv_action):
        action = chainer.as_variable(cp.asarray(action.data))
        diff_q = (q - adv_q)
        diff_a = action - adv_action
        diff_a = diff_a.data[0]
        diff_a[diff_a == 0] = 1e-3
        diff_a = chainer.as_variable(diff_a)
        grad = diff_q / diff_a
        return grad

    def sample_adv_action():
        adv_action = cp.random.uniform(action_space.low, action_space.high, action_size)
        adv_action = chainer.as_variable(cp.expand_dims(adv_action.astype(cp.float32), axis=0))
        return adv_action

    print('Environment Time Step Limit', timestep_limit)

    epsilon = args.epsilon
    lr = args.lr
    budget = args.budget

    if args.rollout == 'Nominal':
        print('Running nominal inference')
        for i in range(args.n_episodes):
            done = False
            obs = env.reset()
            R = 0
            t = 0

            while not done and t < timestep_limit:
                env.render()
                adv_action = sample_adv_action()
                action, q, adv_q = agent.act_forward(obs, adv_action)
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
            agent.stop_episode()
            print('test episode:', i, 'R:', R)

    elif args.rollout == 'Random':
        print('Running random attacks')
        for i in range(args.n_episodes):
            done = False
            obs = env.reset()
            R = 0
            t = 0
            while not done and t < timestep_limit:
                env.render()
                adv_action = sample_adv_action()
                action, q, adv_q = agent.act_forward(obs, adv_action)
                if t % 1 == 0:
                    delta = norms.random_delta(n_dim=action_space.low.size, budget=args.budget)
                    action = action + delta
                    action = cuda.to_cpu(action)
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
            agent.stop_episode()
            print('test episode:', i, 'R:', R)

    elif args.rollout == 'MAS':
        print('Running MAS Attacks')
        for i in range(args.n_episodes):
            obs = env.reset()
            done = False
            R = 0
            t = 0
            while not done and t < timestep_limit:
                if t < args.start_atk:
                    # env.render()
                    adv_action = sample_adv_action()
                    action, q, adv_q = agent.act_forward(obs, adv_action)
                    if args.clip is True:
                        action = np.clip(action, -1, 1)
                    obs, r, done, _ = env.step(action)
                    R += r
                    t += 1
                else:
                    env.render()
                    # optimization loop here
                    adv_action = sample_adv_action()
                    action, q, adv_q = spy.act_forward(obs, adv_action)
                    if args.clip is True:
                        action = np.clip(action, -1, 1)

                    grad_a = compute_grad(q, adv_q, action, adv_action)
                    adv_action_new = adv_action - (lr * grad_a)

                    action, q, adv_q_new = spy.act_forward(obs, adv_action_new)
                    counter = 0

                    while np.absolute(adv_action.data - adv_action_new.data).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        grad_a = compute_grad(adv_q, adv_q_new, adv_action, adv_action_new)
                        adv_action = adv_action_new
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    delta = cuda.to_cpu(adv_action_new.data[0]) - action
                    if args.s == 'l2':
                        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
                    elif args.s == 'l1':
                        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

                    proj_action = action + proj_spatial_delta
                    proj_action = np.clip(proj_action, -1, 1)
                    proj_action_gpu = cuda.to_gpu(proj_action)

                    action, q, adv_q_new = spy.act_forward(obs, proj_action_gpu)
                    obs, r, done, _ = env.step(proj_action)

                    R += r
                    t += 1
            print('test episode:', i, 'R:', R)
            agent.stop_episode()

    elif args.rollout == 'LAS':
        print('LAS Attacks')
        global_budget = args.budget
        epsilon = args.epsilon
        for i in range(args.n_episodes):
            print('Episode:', i)
            env.seed(args.seed)

            exp_v = []
            total_R = []
            states = []
            actions = []
            adv_actions = []

            obs = env.reset()
            spy_obs = spy_env.reset()
            done = False
            R = 0
            t = 0

            while not done and t < timestep_limit:
                # env.render()
                #-----------virtual planning loop begins here-------------------------#
                v_actions = []
                v_values = []
                v_states = []
                spy_rewards = []
                deltas = []

                spy_done = False
                spy_R = 0
                spy_t = 0
                spy_env.seed(args.seed)
                spy_obs = spy_env.reset()

                #bring environment up to state s_t
                for l in range(t):
                    # spy_env.render()
                    spy_obs, spy_r, spy_done, _ = spy_env.step(adv_actions[l])

                spy_done = False

                #reset planning horizon is used up
                if t % args.horizon == 0:
                    horizon = args.horizon
                    global_budget = args.budget

                while not spy_done and spy_t < horizon:
                    # spy_env.render()
                    adv_action = sample_adv_action()
                    if spy_t == 0:
                        action, q, adv_q = spy.act_forward(obs, adv_action)
                    else:
                        action, q, adv_q = spy.act_forward(spy_obs, adv_action)

                    if args.clip is True:
                        action = np.clip(action, -1, 1)

                    grad_a = compute_grad(q, adv_q, action, adv_action)
                    adv_action_new = adv_action - (lr * grad_a)

                    action, q, adv_q_new = spy.act_forward(obs, adv_action_new)
                    if args.clip is True:
                        action = np.clip(action, -1, 1)

                    counter = 0
                    while np.absolute(adv_action.data - adv_action_new.data).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = cuda.to_cpu(adv_action.data[0])
                        grad_a = compute_grad(adv_q, adv_q_new, adv_action, adv_action_new)
                        adv_action = adv_action_new
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    v_actions.append(list(map(float, action)))
                    delta = np.squeeze(cp.asnumpy(adv_action_new.data) - action)
                    deltas.append(list(map(float, delta)))

                    spy_obs, spy_r, spy_done, _ = spy_env.step(action)
                    spy_R += spy_r
                    spy_t += 1

                    v_values.append(q)
                    spy_rewards.append(spy_R)
                    v_states.append(list(map(float, spy_obs)))
                
                # -----------virtual planning loop ends here-------------------------

                if args.s == 'l1':
                    delta_norms = [norms.l1_spatial_norm(delta) for delta in deltas]
                elif args.s == 'l2':
                    delta_norms = [norms.l2_spatial_norm(delta) for delta in deltas]

                if args.t == 'l1':
                    temporal_deltas = norms.l1_time_project2(delta_norms, global_budget)
                elif args.t == 'l2':
                    temporal_deltas = norms.l2_time_project(delta_norms, global_budget)

                spatial_deltas = []
                for a in range(len(temporal_deltas)):
                    if args.s == 'l1':
                        spatial_deltas.append(list(map(float, norms.l1_spatial_project2(deltas[a], temporal_deltas[a]))))
                    elif args.s == 'l2':
                        spatial_deltas.append(list(map(float, norms.l2_spatial_project(deltas[a], temporal_deltas[a]))))

                proj_actions = list(map(add, v_actions[0], spatial_deltas[0]))
                proj_actions = np.clip(proj_actions, -1, 1)

                obs, r, done, _ = env.step(proj_actions)
                R += r
                t += 1

                action, q, adv_q_new = spy.act_forward(obs, cp.asarray(proj_actions))

                total_R.append(R)
                exp_v.append((cp.asnumpy(adv_q_new)).tolist())
                states.append(list(map(float, obs)))
                actions.append(list(map(float, (np.clip(v_actions[0], -1, 1)))))
                adv_actions.append(list(map(float, np.clip(proj_actions, -1, 1))))

                #adaptive reduce budget here
                horizon = horizon - 1
                if args.t == 'l1':
                    used = temporal_deltas[0]
                    global_budget = norms.reducebudget_l1(used, global_budget)
                elif args.t == 'l2':
                    used = temporal_deltas[0]
                    global_budget =  norms.reducebudget_l2(used, global_budget)

            print('test episode:', i, 'R:', R)
            agent.stop_episode()

if __name__ == '__main__':
    path = main()
    chmodder(path) if path != None else None
