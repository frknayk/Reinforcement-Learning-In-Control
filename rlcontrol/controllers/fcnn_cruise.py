import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# Using Cuda
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(0)

# Actor Network
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size,init_w=3e-5):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        """
        Get output of the policy

        Args:
            state (torch array): State of the dynamical system
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.sigmoid_mod(self.linear3(x))
        return x
    
    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        action_np = action.detach().cpu().numpy()[0]
        return action_np

    def tanh_mod(self,x,p=1):
        x = x.float()
        x = ( 2 / ( 1 + torch.exp( -2*(x/100) ) ) ) - 1
        x = x * p
        return x

    def sigmoid_mod(self,x,p=1.5):
        x = x.float()
        x = ( 2 / (1 + torch.exp(x)*1) - 1 ) * -1
        x = x * p
        return x

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    input_list = []
    output_list = []
    explist = np.arange(-5,5,0.1)
    for i in explist.tolist():
        x = i
        output = (2 / (1 + torch.exp(torch.tensor([x]))*1) - 1) * -1
        input_list.append(x)
        output_list.append(output)

    plt.plot(input_list,output_list)
    plt.show()