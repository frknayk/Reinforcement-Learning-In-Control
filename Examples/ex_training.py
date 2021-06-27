# Import system to be controlled
from rlcontrol.systems.double_tank_old import ENV
# Import RL algorithm
from rlcontrol.agents.ddpg import DDPG
# Import organizer to construct training/inference pipeline
from rlcontrol.rlcontrol import Organizer
train_organizer = Organizer(
    env=ENV,
    agent_class=DDPG)
train_organizer.train()
