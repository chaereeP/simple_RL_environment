# simple_RL_framework_for simulation
Still working on revising the code for clarity.

Implementations of some basic RL algorithms with simple codes in Pytorch.
There are several variables:
1. algorithms,
2. continuous / discrete action
3. use different state for value and action function ( denoted as value_state and state, respectively.)

If you want to implement RL algorithms for other simulation environments, you just need to revise "Set_env" class in "environment.py file.

## Algorithms Implemented
| Command    | Description                                    |
| ---------- | ---------------------------------------------- |
| 1. PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)|
| 2. PPO-LSTM (recurrent PPO)   | |
| 3. SAC   | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) |

## Dependencies
Pytorch
Numpy


