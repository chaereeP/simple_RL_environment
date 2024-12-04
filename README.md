# simple_RL_framework_for simulation
Implementations of some basic RL algorithms with simple codes in Pytorch.

### Features:
- **Algorithms**:
  - Proximal Policy Optimization (PPO)
  - PPO with LSTM integration (PPO-LSTM)
  - Soft Actor-Critic (SAC)
- **Action Spaces**:
  - Supports both continuous and discrete action spaces.
- **Modularity**:
  - Easily adaptable to other simulation environments by modifying the `Set_env` class in `environment.py`.

## Run code
If you want to implement RL algorithms for other simulation environments, you just need to revise "Set_env" class in "environment.py" file.

 How to run code : 

    python main.py

## Algorithms Implemented
| Command    | Description                                    |
| ---------- | ---------------------------------------------- |
| 1. PPO | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)|
| 2. PPO-LSTM (recurrent PPO)   | |
| 3. SAC   | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) |

## Dependencies
Pytorch
Numpy


