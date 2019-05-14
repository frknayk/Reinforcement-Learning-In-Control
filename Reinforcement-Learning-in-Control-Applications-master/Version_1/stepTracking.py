import math
import random
import pickle
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


# Nonlinear System Model is defined in 'gym.py' 

env = ENV(1,0.3)

state_dim  = 4
action_dim = 1 # 1
hidden_dim = 32


# Loading Model
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net.load_state_dict(torch.load('pd_Best.weights'))

system_responses1 = []
system_responses2 = []

max_steps = 500


# Creating CSV File
with open('log_stepTracking.csv','w') as f:
    writer = csv.writer(f)

state = env.reset()

references = [0.35,0.30,0.45,0.55,0.30]

for trail in range(len(references)):

    ref = references[trail]
    env.y_set = np.array([ [ref] ])
    # env.e_int = 0
    for step in range(max_steps):

        action = policy_net.get_action(state)

        next_state, reward, done = env.step(action,step)
        
        out1 = np.asscalar(env.y)

        # Update Lists
        system_responses1.append(out1)

        y1,u1 = out1,np.asscalar(action[0])
        best_of = np.array([ [y1,u1] ]).tolist()

        # Saving to Log file (in csv format)
        with open('log_stepTracking.csv','a') as writeFile :
            writer = csv.writer(writeFile)
            writer.writerows(best_of)
        writeFile.close

        state = next_state
    

plt.figure(1)
plt.plot(system_responses1, label="Output-1")
plt.legend(loc='upper left')
plt.show() 