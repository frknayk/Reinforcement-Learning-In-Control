from rlcontrol import Organizer
# from gym_control.double_tank_old import ENV
from gym_control.linear_first_order import ENV
from agents.ddpg import DDPG
import numpy as np


# a = ENV()
# s = a.reset()
# while True:
#     s = a.step(np.matrix([[1.0]]))
#     print(s)
# Create training organizer
# It will create all log files,networks and environment for you
train_organizer = Organizer(
    env=ENV,
    agent_class=DDPG)

# Start train and log results to tensorboard
train_organizer.train()
