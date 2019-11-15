from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from chainerrl import policies
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import misc
from chainerrl import links
from chainerrl import experiments
from chainerrl.agents import PPO
from chainerrl.agents import a3c
import chainerrl
import matplotlib
import os
import norms
import ppo_adversary
from ppo_adversary import PPO_Adversary
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import json
import matplotlib.pyplot as plt
import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym.wrappers
from gym import spaces
import numpy as np
import cupy as cp
from operator import add

class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, obs_size, action_space,
                 n_hidden_layers=3, n_hidden_channels=64,
                 bound_mean=True):
        assert bound_mean in [False, True]
        super().__init__()
        hidden_sizes = (n_hidden_channels,) * n_hidden_layers
        # hidden_sizes = (128, 64) Run4
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--env_id', type=str, default='LL')
    parser.add_argument('--seed', type=int, default=0,  # default 0
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--rollout', type=str, default='Nominal',
                        choices=('Nominal', 'Random', 'MAS', 'LAS'))
    parser.add_argument('--start_atk', type=int, default='1')
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

    if args.env_id == 'LL':
        env_name = 'LunarLanderContinuous-v2'
        load = 'PPOLunarLanderContinuous-v2'
        n_hidden_channels = 128
        n_hidden_layers = 3

    elif args.env_id == 'BW':
        env_name = 'BipedalWalker-v2'
        load = 'PPOBipedalWalker-v2'
        n_hidden_channels = 64
        n_hidden_layers = 3

    elif args.env_id == 'Hopper':
        env_name = 'Hopper-v2'
        load = 'hopper/3177/2000000_finish'
        n_hidden_channels = 64
        n_hidden_layers = 3

    elif args.env_id == 'Walker':
        env_name = 'Walker2d-v2'
        load  = 'walker2D/3204/1600000_finish'
        n_hidden_channels = 64
        n_hidden_layers = 3

    elif args.env_id == 'HalfCheetah':
        env_name = 'HalfCheetah-v2'
        load = 'halfcheetah/2798/2500000_finish' 
        args.seed = 99
        n_hidden_channels = 64
        n_hidden_layers = 3
    else:
        print('No model found')

    def clip_action_filter(a):
        return np.clip(a, action_space.low, action_space.high)

    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs

    def make_env(env_name):
        env = gym.make(env_name)
        # Use different random seeds for train and test envs
        # process_seed = int(process_seeds[process_idx])
        #env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(args.seed)
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
    action_space = env.action_space

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5)

    # Switch policy types accordingly to action space types
    model = A3CFFGaussian(obs_space.low.size, action_space,
                              n_hidden_layers=n_hidden_layers,
                              n_hidden_channels=n_hidden_channels,
                              bound_mean=True)

    #For Mujoco Envs
    #################################################################
    winit = chainerrl.initializers.Orthogonal(1.)
    winit_last = chainerrl.initializers.Orthogonal(1e-2)
    action_size = action_space.low.size
    
    if args.env_id == 'Hopper' or args.env_id == 'Ant':
        policy = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, action_size, initialW=winit_last),
            chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )   
        vf = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 1, initialW=winit),
        )


    elif args.env_id == 'Walker' or args.env_id == 'HalfCheetah':
        policy = chainer.Sequential(
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, action_size, initialW=winit_last),
            chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
                action_size=action_size,
                var_type='diagonal',
                var_func=lambda x: F.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        )

        vf = chainer.Sequential(
            L.Linear(None, 128, initialW=winit),
            F.tanh,
            L.Linear(None, 64, initialW=winit),
            F.tanh,
            L.Linear(None, 1, initialW=winit),
        )


    model = chainerrl.links.Branched(policy, vf)
    opt = chainer.optimizers.Adam(alpha=3e-4, eps=1e-5)
    opt.setup(model)

    agent = PPO_Adversary(model, opt,
                          obs_normalizer=obs_normalizer,
                          gpu=args.gpu, update_interval=2048,
                          minibatch_size=64, epochs=10,
                          clip_eps_vf=None, entropy_coef=0.0,
                          standardize_advantages=True,
                          )

    spy = PPO_Adversary(model, opt,
                        obs_normalizer=obs_normalizer,
                        gpu=args.gpu, update_interval=2048,
                        minibatch_size=64, epochs=10,
                        clip_eps_vf=None, entropy_coef=0.0,
                        standardize_advantages=True,
                        )

    agent.load(load)
    spy.load(load)

    def compute_grad(action, means, std_devs):
        # compute analytical gradient
        coeff = -(action - means) / ((np.power(std_devs, 3) * (np.sqrt(2 * np.pi))))
        power = -(np.power((action - means), 2)) / (2 * np.power(std_devs, 2))
        exp = np.exp(power)
        grad_a = coeff * exp
        return grad_a

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
                # env.render()
                action, action_dist, vs_pred = agent.act_forward(obs)
                obs, r, done, _ = env.step(action)
                if args.clip is True:
                    action = np.clip(action, -1, 1)
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
                # env.render()
                action, action_dist, vs_pred = agent.act_forward(obs)
                if t % 1 == 0:
                    delta = norms.random_delta(n_dim=action_space.low.size, budget=args.budget)
                    action = action + delta
                obs, r, done, _ = env.step(action)
                R += r
                t += 1
            agent.stop_episode()
            print('test episode:', i, 'R:', R)

    elif args.rollout == 'MAS':
        print('Running MAS')
        for i in range(args.n_episodes):
            obs = env.reset()
            done = False
            R = 0
            t = 0
            while not done and t < timestep_limit:
                if t < args.start_atk:
                    print(t)
                    #env.render()
                    action, action_dist, vs_pred = agent.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, -1, 1)
                    obs, r, done, _ = env.step(action)
                    R += r
                    t += 1
                else:
                    #env.render()
                    action, action_dist, vs_pred = spy.act_forward(obs)
                    if args.clip is True:
                        action = np.clip(action, -1, 1)
                    means = []
                    std_devs = []
                    for k in range(len(action_dist.mean.data[0])):
                        means.append(cp.asnumpy(action_dist.mean[0][k].data))
                        var = np.exp(cp.asnumpy(action_dist.ln_var[0][k].data))
                        std_devs.append(np.sqrt(var))

                    grad_a = compute_grad(action, means, std_devs)
                    adv_action = action - (lr * grad_a)

                    grad_a = compute_grad(adv_action, means, std_devs)
                    adv_action_new = adv_action - (lr * grad_a)

                    counter = 0
                    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = adv_action_new
                        grad_a = compute_grad(adv_action, means, std_devs)
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    delta = adv_action_new - action
                    if args.s == 'l2':
                        proj_spatial_delta = norms.l2_spatial_project(delta, budget)
                    elif args.s == 'l1':
                        proj_spatial_delta = norms.l1_spatial_project2(delta, budget)

                    proj_action = action + proj_spatial_delta
                    proj_action = np.clip(proj_action, -1, 1)
                    obs, r, done, _ = env.step(proj_action)
                    R += r
                    t += 1

            print('test episode:', i, 'R:', R)
            agent.stop_episode()

    elif args.rollout == 'LAS':
        print('Running LAS')
        global_budget = args.budget
        epsilon = args.epsilon
        for i in range(args.n_episodes):
            print('Episode:', i)
            exp_v = []
            total_R = []
            states = []
            actions = []
            adv_actions = []

            obs = env.reset()
            done = False
            R = 0
            t = 0

            while not done and t < timestep_limit:
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

                #bring virtual environment up to state s_t
                for l in range(t):
                    #spy_env.render()
                    spy_obs, spy_r, spy_done, _ = spy_env.step(adv_actions[l])
                spy_done = False

                if t%args.horizon == 0:
                    horizon = args.horizon
                    global_budget = args.budget

                while not spy_done and spy_t < horizon:
                    #spy_env.render()

                    #forcing spy to act on real observation
                    if spy_t == 0:
                        action, action_dist, vs_pred = spy.act_forward(obs)
                    else:
                        action, action_dist, vs_pred = spy.act_forward(spy_obs)
                    
                    if args.clip is True:
                        action = np.clip(action, -1, 1)

                    means = []
                    std_devs = []
                    dims = len(action_dist.mean.data[0])
                    for j in range(dims):
                        means.append(cp.asnumpy(action_dist.mean[0][j].data))
                        var = np.exp(cp.asnumpy(action_dist.ln_var[0][j].data))
                        std_devs.append(np.sqrt(var))

                    grad_a = compute_grad(action, means, std_devs)
                    adv_action = action - (lr * grad_a)

                    grad_a = compute_grad(adv_action, means, std_devs)
                    adv_action_new = adv_action - (lr * grad_a)

                    counter = 0
                    while np.absolute(adv_action - adv_action_new).any() > epsilon and counter < 25:
                        # print('Optimizing')
                        adv_action = adv_action_new
                        grad_a = compute_grad(adv_action, means, std_devs)
                        adv_action_new = adv_action - (lr * grad_a)
                        counter += 1

                    v_actions.append(list(map(float, action)))
                    delta = adv_action_new - action
                    deltas.append(list(map(float, delta)))

                    spy_obs, spy_r, spy_done, _ = spy_env.step(action)
                    spy_R += spy_r
                    spy_t += 1

                    v_values.append(float(vs_pred.data[0][0]))
                    spy_rewards.append(spy_R)
                    v_states.append(list(map(float, spy_obs)))

                #-----------virtual planning loop ends here-------------------------#
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

                total_R.append(R)
                exp_v.append(float(vs_pred[0][0].data))
                states.append(list(map(float, obs)))
                actions.append(list(map(float, (np.clip(v_actions[0], -1, 1)))))
                adv_actions.append(list(map(float, np.clip(proj_actions, -1, 1))))

                #adaptive reduce budget and planning horiozn here
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
    main()
