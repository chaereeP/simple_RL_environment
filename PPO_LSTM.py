import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal,Normal
from torch.autograd import Variable
from abc import ABC, abstractmethod
import torch as th
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy

print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self,state_dim, value_state_dim,action_dim, hidden_size,max_size=int(5e3), recurrent=False):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.recurrent = recurrent
        self.states = numpy.zeros((self.max_size, state_dim))
        self.value_states = numpy.zeros((self.max_size, value_state_dim))
        self.actions = numpy.zeros((self.max_size, action_dim))
        self.logprobs = numpy.zeros((self.max_size, 1))
        self.rewards = numpy.zeros((self.max_size, 1))
        self.is_terminals = numpy.zeros((self.max_size, 1))

        if self.recurrent:
            self.c = numpy.zeros((self.max_size, hidden_size))
            self.h = numpy.zeros((self.max_size, hidden_size))
            self.v_c = numpy.zeros((self.max_size, hidden_size))
            self.v_h = numpy.zeros((self.max_size, hidden_size))

    def add(self, state,value_state, action,logprobs, reward, done):
        self.states[self.ptr] = state
        self.value_states[self.ptr] = value_state
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = reward
        self.is_terminals[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def on_policy_sample(self):
        ind = numpy.arange(0, self.size)

        # TODO: Clean up indexing. RNNs needs batch shape of
        # Batch size * Timesteps * Input size
        
        if not self.recurrent:
            return self._ff_sampling(ind)
        s = torch.FloatTensor(
            self.states[ind][:, None, :]).to(device)
        v_s = torch.FloatTensor(
            self.value_states[ind][:, None, :]).to(device)
        
        # reward and dones don't need to be "batched"
        a = torch.FloatTensor(
            self.actions[ind]).to(device)
        logprobs = torch.FloatTensor(
            self.logprobs[ind]).to(device)
        r = torch.FloatTensor(
            self.rewards[ind]).to(device)
        d = torch.FloatTensor(
            self.is_terminals[ind]).to(device)
        s=s.reshape(1,s.size(0),s.size(-1))
        v_s=v_s.reshape(1,v_s.size(0),v_s.size(-1))

        return s, v_s,a, logprobs, r, d #, actor_hidden, critic_hidden

    def _ff_sampling(self, ind):
        # FF only need Batch size * Input size, on_policy or not
        hidden = None
        next_hidden = None
        s = torch.FloatTensor(self.states[ind]).to(device)
        v_s = torch.FloatTensor(
            self.value_states[ind][:, None, :]).to(device)
        a = torch.FloatTensor(self.actions[ind]).to(device)
        logprobs = torch.FloatTensor(
            self.logprobs[ind][:, None, :]).to(device)
        r = torch.FloatTensor(self.rewards[ind]).to(device)
        d = torch.FloatTensor(self.is_terminals[ind]).to(device)
        return s,v_s, a,logprobs, r, d, hidden, next_hidden

    def clear(self):
        self.ptr=0
        self.size = 0

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,value_state_dim,has_continuous_action_space, action_std_init,is_recurrent,hidden_dim):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        self.recurrent = is_recurrent
        if self.recurrent:
            self.actor_1 = nn.LSTM(state_dim, hidden_dim, batch_first=True)
            self.critic_1 = nn.LSTM(value_state_dim, hidden_dim, batch_first=True)
            for name, param in self.actor_1.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, numpy.sqrt(2))
            for name, param in self.critic_1.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, numpy.sqrt(2))
        else:
            self.actor_1 = nn.Linear(state_dim, hidden_dim)
            self.critic_1 = nn.Linear(value_state_dim, hidden_dim)

        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.actor_2 = nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Tanh(),
                            nn.Linear(hidden_dim, 5),
                        )
        else :
            self.actor_2 = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, action_dim*11),
                        nn.Softmax(dim=-1)
                        )
        # critic
        self.critic_2 = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Linear(hidden_dim, 1)
                    )
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state ,actor_hidden,value_states, critic_hidden):
        if self.recurrent:
            self.actor_1.flatten_parameters()
            p, actor_hidden = self.actor_1(state, actor_hidden)
            p = p.squeeze(1)
            action_mean = self.actor_2(p.data)
            self.critic_1.flatten_parameters()
            v, critic_hidden = self.critic_1(value_states, critic_hidden)
        else:
            p = torch.tanh(self.actor_1(state))
            action_mean = self.actor_2(p)
        if self.has_continuous_action_space:
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean[0], cov_mat)
            action = dist.sample()
            action = action.clamp(min=-1,max=1)
        else:
            dist = MultiCategoricalDistribution(action_dims=[11,11,11,11,11] )
            action = dist.actions_from_params(action_mean)
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach(), actor_hidden, critic_hidden

    def evaluate(self, state, value_state , action):
        if self.recurrent:
            self.actor_1.flatten_parameters()
            self.critic_1.flatten_parameters()
            # print(actor_hidden.shape)
            actor_hidden, critic_hidden = self.get_initial_states(self.actor_1)
            p, h = self.actor_1(state, actor_hidden)
            p = p.squeeze(1)
            state_values, critic_hidden = self.critic_1(value_state, critic_hidden)
            state_values = state_values.squeeze(1)
            action_mean = self.actor_2(p.data)
            state_values = self.critic_2(state_values.data)
            if self.has_continuous_action_space:
                action_var = self.action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                dist = MultiCategoricalDistribution(action_dims=[11, 11, 11, 11, 11])
                dist.proba_distribution(action_mean)
            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            # state_values = state_values[..., None]
            action_logprobs = action_logprobs[..., None]
            dist_entropy = dist_entropy[..., None]
        else:
            p = torch.tanh(self.actor_1(state))
            state_values = torch.tanh(self.critic_1(value_state))
            action_mean = self.actor_2(p)
            state_values = self.critic_2(state_values)

            if self.has_continuous_action_space:
                action_var = self.action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            else :
                dist = MultiCategoricalDistribution(action_dims=[11,11,11,11,11])
                dist.proba_distribution(action_mean)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            action_logprobs = action_logprobs[..., None]
        return action_logprobs, state_values, dist_entropy

    def get_initial_states(self, policy):
        h_0, c_0 = None, None
        if self.recurrent:
            h_0 = Variable(
                torch.zeros((policy.num_layers, 1, policy.hidden_size),
                            dtype=torch.float).to(device=device), requires_grad=False)
            c_0 = Variable(torch.zeros((policy.num_layers, 1, policy.hidden_size),
                                       dtype=torch.float).to(device=device), requires_grad=False)
            h_1 = Variable(torch.zeros((policy.num_layers, 1, policy.hidden_size),
                                       dtype=torch.float).to(device=device), requires_grad=False)
            c_1 = Variable(torch.zeros((policy.num_layers, 1, policy.hidden_size),
                                       dtype=torch.float).to(device=device), requires_grad=False)

        return (h_0,c_0), (h_1,c_1)

class PPO_LSTM:
    def __init__(self, state_dim, action_dim, value_state_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,has_continuous_action_space, is_recurrent,hidden_dim,action_std_init=0.6):
        #action_std_init :  starting std for action distribution (Multivariate Normal)
        self.has_continuous_action_space = has_continuous_action_space
        self.recurrent=is_recurrent
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer(state_dim, value_state_dim,action_dim, hidden_dim,int(5e3), is_recurrent)

        self.policy = ActorCritic(state_dim, action_dim, value_state_dim,has_continuous_action_space, action_std_init, is_recurrent,hidden_dim).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor_1.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor_2.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic_1.parameters(), 'lr': lr_critic},
                        {'params': self.policy.critic_2.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim,value_state_dim, has_continuous_action_space, action_std_init,is_recurrent,hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_initial_states(self):
        return self.policy.get_initial_states(self.policy_old.actor_1)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state,actor_hidden, value_states, critic_hidden):
        with torch.no_grad():
            if self.recurrent:
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)[:, None, :]
                value_states = torch.FloatTensor(value_states.reshape(1, -1)).to(device)[:, None, :]
            else:
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                value_states = torch.FloatTensor(value_states.reshape(1, -1)).to(device)
            action, action_logprob,actor_hidden, critic_hidden = self.policy_old.act(state,actor_hidden, value_states,critic_hidden)
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten(),action_logprob.detach().cpu().numpy(), actor_hidden ,critic_hidden
        else:
            return action.detach().cpu().numpy() ,action_logprob.detach().cpu().numpy(),actor_hidden, critic_hidden

    def update(self):
        old_states, old_value_states, old_actions, old_logprobs, reward, is_terminals = self.buffer.on_policy_sample()
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(reward), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards[..., None]

        # convert list to tensor

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_value_states,old_actions)
            # match state_values tensor dimensions with rewards tensor
            state_values = state_values.squeeze(dim=0)
            logprobs = logprobs.squeeze(dim=0)
            dist_entropy = dist_entropy.squeeze(dim=0)

            # Finding the ratio (pi_theta / pi_theta__old)
            assert (state_values.shape == rewards.shape)
            assert logprobs.shape == old_logprobs.shape
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # 1) for policy
            advantages = rewards - state_values.detach()
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pol_surr1 = ratios * advantages
            pol_surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # # 2) for critic
            cri_surr1 = self.MseLoss(state_values, rewards)
            cri_surr2 = torch.clamp(cri_surr1,-self.eps_clip,self.eps_clip)

            # final loss of clipped objective PPO
            # cri_loss = torch.max(cri_surr1,cri_surr2)
            loss = -torch.min(pol_surr1, pol_surr2) + 0.5*torch.max(cri_surr1,cri_surr2)- 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # nn.utils.clip_grad_norm_([*self.actor_1.parameters(),*self.actor_2.parameters(), *self.critic_1.parameters(),*self.actor_2.parameters()], 0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        torch.cuda.empty_cache()
        self.buffer.clear()
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(self.optimizer.state_dict(),checkpoint_path + "_optimizer")
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_old.eval()
        self.policy.eval()


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = Categorical

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""


    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """


    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """


    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """


    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """


    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """


    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()


    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """


    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """

class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits


    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)


    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)


    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)


    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)


    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)


    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
