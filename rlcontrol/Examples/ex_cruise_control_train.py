# Import system to be controlled
# Import RL algorithm
from rlcontrol.agents.ddpg import DDPG
from rlcontrol.controllers.fcnn_cruise import PolicyNetwork, ValueNetwork

# Import organizer to construct training/inference pipeline
from rlcontrol.rlcontrol import Organizer
from rlcontrol.systems.cruise_system import CruiseSystem

ddpg_cruise_config = {
    "batch_size": 256,
    "hidden_dim": 32,
    "policy_net": PolicyNetwork,
    "value_net": ValueNetwork,
}

env_config = CruiseSystem.get_config_default()
env_config["random_state_init"] = True

train_organizer = Organizer(
    env=CruiseSystem,
    agent_class=DDPG,
    agent_config=ddpg_cruise_config,
    env_config=env_config,
)
train_config = train_organizer.get_default_training_config()
train_config["enable_log"] = True
train_config["max_episode"] = 150
train_config["algorithm_name"] = "DDPG"
train_config["max_step"] = 500
train_config["plotting"]["enable"] = True
train_config["plotting"]["freq"] = 10
train_organizer.train(train_config)
