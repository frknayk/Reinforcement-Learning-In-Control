import math
import random
import os
import pickle
import datetime
import csv

from gym import ENV
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

# Using Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
Action_dim = 1
"Ornstein-Uhlenbeck process"
# Adding time-correlated noise to the actions taken by the deterministic policy
class OUNoise(object):
    def __init__(self, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = Action_dim
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

# Critic Network
class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
    # def __init__(self, num_inputs, num_actions, hidden_size):

        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Actor Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size,init_w=3e-3):

        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        # x = self.tanh_mod(self.linear3(x),25)
        x = self.sigmoid_mod(self.linear3(x),10)
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

    def tanh_mod(self,x,p):
        x = x.float()
        x = ( 2 / ( 1 + torch.exp( -2*(x/100) ) ) ) - 1
        x = x * p
        return x

    def sigmoid_mod(self,x,p):
        x = x.float()
        x = 1 / (1 + torch.exp(-x)*10 )
        x = x * p
        return x

def ddpg_update(batch_size,
           gamma = 0.99,
           min_value=-np.inf,
           max_value=np.inf,
           soft_tau=1e-2):
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)


    policy_loss = value_net(state, policy_net(state) )
    policy_loss = -policy_loss.mean()

    next_action    = target_policy_net(next_state)
    target_value   = target_value_net(next_state, next_action.detach())
    expected_value = reward + (1.0 - done) * gamma * target_value
    expected_value = torch.clamp(expected_value, min_value, max_value)

    value = value_net(state, action)
    value_loss = value_criterion(value, expected_value.detach())


    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    if( eps % 1 == 0 ):
        for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

    return value_loss,policy_loss

# Non-Linear System Model is defined in "gym.py" file
env = ENV(1,0.5)

state_dim  = 4
action_dim = Action_dim # 1
hidden_dim = 32



value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)


value_lr  = 1e-2
policy_lr = 1e-3

value_optimizer  = optim.Adam(value_net.parameters(),  lr=value_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

value_criterion = nn.MSELoss()

replay_buffer_size = 50000
replay_buffer = ReplayBuffer(replay_buffer_size)

ou_noise = OUNoise()

max_steps   = 200
max_episodes = 250
batch_size  = 128


# Creating CSV File for All data
with open('log.csv','w') as f:
    writer = csv.writer(f)
# Creating CSV File for Episode Total Reward
with open('log_EpisodeReward.csv','w') as f:
    writer = csv.writer(f)

best_Reward = -2000

for eps in range(max_episodes):

    state = env.reset()
    ou_noise.reset()
    total_rewards = 0
    total_control_signal = 0

    for step in range(max_steps):
        action = policy_net.get_action(state)
        action = ou_noise.get_action(action, step)
        total_control_signal = total_control_signal + action
       
        next_state, reward, done = env.step(action,step)

        y1 = np.asscalar(env.y)
        u1 = np.asscalar(action[0])

        step_log = np.array([ [y1,u1,total_rewards] ]).tolist()

        # Saving to Log file (in csv format)
        with open('log.csv','a') as writeFile :
            writer = csv.writer(writeFile)
            writer.writerows(step_log)
        writeFile.close

        replay_buffer.push( state.reshape(4,) , action.reshape(1,) , reward, next_state.reshape(4,) , done)
        
        # print(str(env.y) + " *** " + str(env.e_dot) + " *** " + str(env.e_t1) + " *** " + str(env.r) )

        if(u1<-1):
            print("Control Signal : ",u1)
        # print(" h1' : [ {0} ] --- h2' : [ {1} ] \n".format(env.x[0],env.x[1]))
        # input()

        total_rewards = total_rewards + reward

        if len(replay_buffer) > batch_size:
            value_loss,policy_loss = ddpg_update(batch_size)

        state = next_state
    
    str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(eps+1,np.asscalar(env.y_set),np.asscalar(env.y),total_rewards)
    print(str1);print("\n*******************************\n")

    # Saving Model
    torch.save(policy_net.state_dict(), 'pd.weights')
    total_rewards_np = np.array( [[total_rewards]]).tolist()

    if(total_rewards > best_Reward) : 
        # Saving Model
        torch.save(policy_net.state_dict(), 'pd_Best.weights')
        best_Reward = total_rewards
    
    
    # Saving Epsiode Log file (in csv format)
    with open('log_EpisodeReward.csv','a') as writeFile :
        writer = csv.writer(writeFile)
        writer.writerows(total_rewards_np)
    writeFile.close

    
    
    