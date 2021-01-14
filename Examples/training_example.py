from rlcontrol import Organizer
from gym_control.double_tank_old import ENV
from agents.ddpg import DDPG

# Create training organizer
## It will create all log files,networks and environment for you
train_organizer = Organizer(
    env=ENV,
    agent_class=DDPG)

# Start train and log results to tensorboard
train_organizer.train()
