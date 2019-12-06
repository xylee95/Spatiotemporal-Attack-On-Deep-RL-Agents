# Spatiotemporal-Attack-On-Deep-RL-Agents

## Brief overview
This post demonstrates some of the action space attack strategies on Deep RL agents presented in the the paper Spatiotemporally Constrained Action Space Attack on Deep RL Agents. 

The first attack strategy developed was the Myopic Action Space Attack (MAS). In MAS, we formulated the attack model as an optimization problem with decoupled constraints. The objective of the optmization is to craft a perturbation at every step that minimizes the Deep RL agent's immediate reward, subject to the constraint that the perturbation cannot exceed a certain budget. The constraints are decoupled since each constraint is only imposed upon the the immediate perturbation and are independent of the agent's trajectory. Hence, this can be thought of a myopic attack since we only aim to reduce immediate reward without taking into account future rewards. Given a small budget, these attacks might not have any apparent effect on the Deep RL agent as the agent might be able to recover in future steps.

The second attack strategy proposed was the Look-ahead Action Space Attack (LAS). In LAS, we formulated the attack model as an optimization problem with constraints that are coupled through the temporal dimension. The objective of the optimization is to craft a sequence of perturbation that minimizes the Deep RL agent's cumulative reward of a trajectory, subjected to the constraint that the total perturbations in the sequence cannot exceed a budget. By considering an overall budget of perturbations over a trajectory, the crafted perturbations are **less** myopic since they take into account the future states of the agent. Hence, a given budget of perturbations can be allocated more effectively to vulnerable states rather than being forced to expend all the perturbation on the immediate state.

## Results

As hypothesized, given the same budget, LAS proves to be a much stronger attack than MAS, which is in turn stronger than a random perturbation. Results are shown for rewards obtained by the RL agent across 10 different episodes.
![Distribution of rewards for PPO in Lunar Lander agent under different attacks](/images/PPO_LL_boxplot.png "Distribution of rewards for PPO in Lunar Lander agent under different attacks")

## Box2D environments
Trained PPO agent in Lunar Lander Environment
![PPO Agent in Lunar Lander Environment](/images/PPO_LL_nom.gif "PPO agent in Lunar Lander Environment")

Trained DDQN agent in Lunar Lander Environment
![DDQN Agent in Lunar Lander Environment](/images/DDQN_LL_nom.gif "DDQN agent in Lunar Lander Environment")

Implementation of LAS attacks on PPO agent in Lunar Lander Environment
![LAS attack on PPO](/images/PPO_LL_LAS_b4h5.gif "LAS attack on PPO")

Implementation of LAS attacks on DDQN agent trained in Lunar Lander Environment
![LAS attack on DDQN](/images/DDQN_LL_LAS_b5h5.gif "LAS attack on DDQN")

Trained PPO agent in Bipedal Walker Environment
![PPO Agent in Bipedal Walker Environment](/images/PPO_BW_nom.gif "PPO agent in Lunar Lander Environment")

Trained DDQN agent in Bipedal Walker Environment
![DDQN Agent in Bipedal Walker Environment](/images/DDQN_BW_nom.gif "PPO agent in Lunar Lander Environment")

Implementation of LAS attacks on PPO agent trained in Bipedal Walker Environment
![LAS attack on PPO Agent in Bipedal Walker Environment](/images/PPO_BW_LAS_b5h5.gif "PPO agent in Lunar Lander Environment")

Implementation of LAS attacks on DDQN agent trained in Bipedal Walker Environment
![LAS attack on DDQN Agent in Bipedal-Walker Environment](/images/DDQN_PPO_LAS_b5h5.gif "PPO agent in Lunar Lander Environment")

## MUJOCO Environments
Trained PPO agent in Walker-2D   
![PPO Agent in Walker-2D](/images/walker_nom.gif "PPO Agent in Walker-2D")

PPO agent under LAS attack in Walker-2D   
![LAS attack on PPO Agent in Walker-2D](/images/walker_fast.gif "LAS attack on PPO Agent in Walker-2D. Right animation illustrates virtual rollout of adversarial agent to craft an attack based on the agent's future dynamics")

Trained PPO agent in Half-Cheetah      
![PPO Agent in Half-Cheetah](/images/cheetah_nom.gif "PPO Agent in Half-Cheetah")

PPO agent under LAS attack in Half-Cheetah   
![LAS attack on PPO Agent in Half-Cheetah](/images/cheetah_fast.gif "LAS attack on PPO Agent in Half-Cheetah. Right animation illustrates virtual rollout of adversarial agent to craft an attack based on the agent's future dynamics")

Trained PPO agent in Hopper  
![PPO Agent in Hopper](/images/hopper_nom.gif "PPO Agent in Hopper")

PPO agent under LAS attack in Hopper
![LAS attack on PPO Agent in Hopper](/images/hopper_fast.gif "LAS attack on PPO Agent in Hopper. Right animation illustrates virtual rollout of adversarial agent to craft an attack based on the agent's future dynamics")

More detailed information and supplemental materials are available at https://arxiv.org/abs/1909.02583

---
## Implementation
### Pre-requisites 
This repository crafts the action space attacks on RL agents. The nominal agents were trained using ChainerRL library. Strictly speaking, the attacks does not require any specific libraries but the code in this repository utilizes Chainer variables and Cupy to accelerate the attacks. 

### Code Structure
1. (agent)_adversary.py contains class of agents that have been augmented to explicitly return Q-values, value functions and action probability distributions.
2. norms.py contains implementations of different norms and projections.
3. (agent)_inference.py contains implementations of attack algorithms on RL agent during inference.

### Running the code
1. To run inference using trained RL agent:
    * Run any one of the (agent)_inference.py with environment arguments 
         * --env_id **LL** for LunarLanderContinuous-v2
         * --env_id **BW** for BipedalWalker-v2
         * --env_id **Hopper** for Hopper-v2 
         * --env_id **Walker** for Walker2d-v2 
         * --env_id **HalfCheetah** for HalfCheetah-v2 
    * Attack argument
         * --rollout **Nominal** runs nominal agent for visualization
         * --rollout **MAS** runs a nominal agent and attacks the agent's action space at every step
         * --rollout **LAS** runs a nominal agent and attacks the agent's action space using attacks that are optimized and projected back to the spatial and temporal budget constraints.
    * Budget of attack
         * --budget any integer or float value
    * Type of spatial projection
         * --s **l1** for l1 projection of attacks onto action dimensions
         * --s **l2** for l2 projection of attacks onto action dimensions
    * Planning horizon (only for LAS)
         * --horizon any integer value
    * Type of temporal projection (only for LAS)
         * --t **l1** for l1 projection of attacks onto temporal dimensions
         * --t **l2** for l2 projection of attacks onto temporal dimensions

Example: To run LAS attack on a PPO agent in Lunar Lander environment with an allocated budget of 5 with a planning horizon of 5 steps using l2 temporal and spatial projections

`python ppo_inference.py --env_id LL --rollout LAS --budget 5 --horizon 5 --s l2 --t l2`

For a list of required packages, please refer to requirements.txt. 
