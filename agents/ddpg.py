import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from controller.fcnn import ValueNetwork,PolicyNetwork
from agents.base import Agent


# Using Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class DDPG(Agent):
    def __init__(self,
            action_dim, 
            state_dim,
            batch_size = 64,
            hidden_dim = 32,
            params=None):
        super().__init__(state_dim, action_dim)
        self.algorithm_name = "DDPG"
        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.batch_size = batch_size

        # Ornstein-Uhlenbeck Noise
        self.ou_noise = OUNoise(action_dim) 

        if not params:
            self.params = self.get_default_params()

        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=self.params['value_lr'])
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['policy_lr'])
        self.value_criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.params['replay_buffer_size'])

        # Copy original networks to target network weights (NOT SURE)
        self.create_target_nets()

    def get_algorithm_name(self):
        return self.algorithm_name

    def get_batch_size(self):
        return self.batch_size

    @staticmethod
    def get_default_params():
        params = {
            'value_lr' : 1e-2,
            'policy_lr' : 1e-3,
            'replay_buffer_size' : 5000,
            'gamma' : 0.99,
            'min_value' : -np.inf,
            'max_value' : np.inf,
            'soft_tau' : 1e-2,
            'soft_update_frequency' : 1
        }

    def create_target_nets(self):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

    def apply(self, state, step):
        action =  self.policy_net.forward(state)
        action = self.ou_noise.get_action(action, step)
        return action

    def update_agent(self,episode_number):
        """Update agent weights by replay buffer rollout"""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state) )
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) *  self.params['gamma'] * target_value
        expected_value = torch.clamp(expected_value, self.params['min_value'] , self.params['max_value'])

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        if( episode_number % self.params['soft_update_frequency'] == 0 ):
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.params['soft_tau']) + param.data * self.params['soft_tau']
                    )

            for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(
                        target_param.data * (1.0 - self.params['soft_tau']) + param.data * self.params['soft_tau']
                    )

        return value_loss,policy_loss

    def update_memory(self, state, action, reward, next_state, done):
        state = state.reshape(4,)
        action = action.reshape(1,)
        next_state = next_state.reshape(4,)
        self.replay_buffer.push(state, action, reward, next_state, done)

    def load(self,agent_weight_abs):
        """Loading Model"""
        try:
            self.policy_net.load_state_dict(torch.load(agent_weight_abs))
        except Exception as e:
            raise("Network could not load : {0}".format(e))

    def save(self,agent_weight_abs):
        """Loading Model"""
        try:
            torch.save(self.policy_net.state_dict(), agent_weight_abs)
        except Exception as e:
            raise("Network could not load : {0}".format(e))

    def reset(self):
        self.ou_noise.reset()

class OUNoise(object):
    "Ornstein-Uhlenbeck process"
    def __init__(self,action_dim, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        """ Adding time-correlated noise to the actions taken by the deterministic policy"""
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state,-35,35)

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









