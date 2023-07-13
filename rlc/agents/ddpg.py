import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.core.defchararray import not_equal

from rlc.agents.base import Agent

# Using Cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

seed_number = 59
np.random.seed(seed_number)
random.seed(seed_number)
torch.manual_seed(seed_number)


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size, init_w=3e-5):
        super(PolicyNetwork, self).__init__()
        num_inputs = observation_space.shape[0]
        num_actions = action_space.shape[0]
        self.action_limit = action_space.high[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.relu_1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Get output of the policy

        Args:
            state (torch array): State of the dynamical system
        """
        x = self.relu_1(self.linear1(state))
        x = self.relu_2(self.linear2(x))
        x = self.sigmoid_mod(self.linear3(x))
        return x * self.action_limit

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        action_np = action.detach().cpu().numpy()[0]
        return action_np

    def tanh_mod(self, x, p=1):
        x = x.float()
        x = (2 / (1 + torch.exp(-2 * (x / 100)))) - 1
        x = x * p
        return x

    def sigmoid_mod(self, x, p=1.5):
        x = x.float()
        x = (2 / (1 + torch.exp(x) * 1) - 1) * -1
        x = x * p
        return x


class ValueNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        num_inputs = observation_space.shape[0]
        num_actions = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.relu_1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu_2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu_1(self.linear1(x))
        x = self.relu_2(self.linear2(x))
        x = self.linear3(x)
        return x


config_default = {
    "batch_size": 64,
    "hidden_dim": 32,
    "policy_net": PolicyNetwork,
    "value_net": ValueNetwork,
}


class DDPG(Agent):
    def __init__(
        self, action_space, observation_space, agent_config: config_default, params=None
    ):
        super().__init__(observation_space, action_space)
        self.algorithm_name = "DDPG"
        self.action_dim = action_space.shape[0]
        hidden_dim = agent_config["hidden_dim"]
        self.batch_size = agent_config["batch_size"]
        self.value_net = agent_config["value_net"](
            observation_space, action_space, hidden_dim
        ).to(device)
        self.policy_net = agent_config["policy_net"](
            observation_space, action_space, hidden_dim
        ).to(device)
        self.target_value_net = agent_config["value_net"](
            observation_space, action_space, hidden_dim
        ).to(device)
        self.target_policy_net = agent_config["policy_net"](
            observation_space, action_space, hidden_dim
        ).to(device)
        # Ornstein-Uhlenbeck Noise
        self.ou_noise = OUNoise(action_space)
        if not params:
            self.params = self.get_default_params()
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=self.params["value_lr"]
        )
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.params["policy_lr"]
        )
        self.value_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.params["replay_buffer_size"])
        # Copy original networks to target network weights (NOT SURE)
        self.create_target_nets()

    def get_algorithm_name(self):
        return self.algorithm_name

    def get_batch_size(self):
        return self.batch_size

    @staticmethod
    def get_default_params():
        params = {
            "value_lr": 1e-2,
            "policy_lr": 1e-3,
            "replay_buffer_size": 5000,
            "gamma": 0.99,
            "min_value": -np.inf,
            "max_value": np.inf,
            "soft_tau": 1e-2,
            "soft_update_frequency": 1,
        }
        return params

    def create_target_nets(self):
        for target_param, param in zip(
            self.target_value_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(param.data)

        for target_param, param in zip(
            self.target_policy_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def apply(self, state_dict: dict, step: int):
        # state = torch.FloatTensor(state).unsqueeze(0).to(device)
        state_diff = state_dict["state_ref"] - state_dict["state"]
        state_diff = torch.FloatTensor(state_diff).unsqueeze(0).to(device)
        action = self.policy_net.forward(state_diff)
        if torch.isnan(action):
            print("SOME SERIOUS PROBLEMS ENCOUNTERED IN action.cpu().data.numpy()")
            print("action : ", action)
            print("STATE : ", state_diff)
            return None
        # convert torch to np
        action_np = action.cpu().data.numpy()
        action_reshaped = action_np.reshape(self.action_dim)
        action_noisy = self.ou_noise.get_action(action_reshaped, step)

        return action_noisy

    def update_agent(self, episode_number):
        if self.batch_size < len(self.replay_buffer):
            value_loss, policy_loss = self._update_agent(episode_number)
            loss_dict = {}
            loss_dict["policy_loss"] = policy_loss.item()
            loss_dict["value_loss"] = value_loss.item()
            return loss_dict

    def _update_agent(self, episode_number):
        """Update agent weights by replay buffer rollout"""
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.params["gamma"] * target_value
        expected_value = torch.clamp(
            expected_value, self.params["min_value"], self.params["max_value"]
        )

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # # TODO : Add this to config
        # clipping_value = 1
        # torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), clipping_value)
        # torch.nn.utils.clip_grad_norm(self.value_net.parameters(), clipping_value)
        # torch.nn.utils.clip_grad_norm(self.target_policy_net.parameters(), clipping_value)
        # torch.nn.utils.clip_grad_norm(self.target_value_net.parameters(), clipping_value)

        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if episode_number % self.params["soft_update_frequency"] == 0:
            for target_param, param in zip(
                self.target_value_net.parameters(), self.value_net.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.params["soft_tau"])
                    + param.data * self.params["soft_tau"]
                )

            for target_param, param in zip(
                self.target_policy_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.params["soft_tau"])
                    + param.data * self.params["soft_tau"]
                )

        return value_loss, policy_loss

    def update_memory(self, state, action, reward, next_state, done):
        state = state.reshape(
            state.shape[0],
        )
        action = action.reshape(
            1,
        )
        next_state = next_state.reshape(
            state.shape[0],
        )
        self.replay_buffer.push(state, action, reward, next_state, done)

    def load(self, agent_weight_abs):
        """Loading Model"""
        try:
            self.policy_net.load_state_dict(torch.load(agent_weight_abs))
        except Exception as e:
            raise ("Network could not load : {0}".format(e))

    def save(self, agent_weight_abs):
        """Loading Model"""
        try:
            torch.save(self.policy_net.state_dict(), agent_weight_abs)
        except Exception as e:
            raise "Network could not load : {0}".format(e)

    def reset(self):
        self.ou_noise.reset()


class OUNoise(object):
    "Ornstein-Uhlenbeck process"

    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        """Adding time-correlated noise to the actions taken by the deterministic policy"""
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.action_space = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        # return np.clip(action + ou_state,-1,1)
        return np.clip(action + ou_state, self.action_space.low, self.action_space.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
