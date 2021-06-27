# Import system to be controlled
from rlcontrol.systems.cruise_system import CruiseSystem
# Import RL algorithm
from rlcontrol.agents.ddpg import DDPG
from rlcontrol.controllers.fcnn_cruise import PolicyNetwork, ValueNetwork
# Import organizer to construct training/inference pipeline
from rlcontrol.rlcontrol import Organizer

ddpg_cruise_config = {
    'batch_size' : 64,
    'hidden_dim' : 32,
    'policy_net' : PolicyNetwork,
    'value_net' : ValueNetwork
}

train_organizer = Organizer(
    env=CruiseSystem,
    agent_class=DDPG,
    agent_config = ddpg_cruise_config)

train_config = train_organizer.get_default_training_config()
train_config['max_episode'] = 500
train_config['max_step'] = 1000
train_organizer.train(train_config)
