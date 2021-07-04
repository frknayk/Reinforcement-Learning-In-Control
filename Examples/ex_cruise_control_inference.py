# Import system to be controlled
from rlcontrol.systems.cruise_system import CruiseSystem
# Import RL algorithm
from rlcontrol.agents.ddpg import DDPG
from rlcontrol.controllers.fcnn_cruise import PolicyNetwork, ValueNetwork
# Import organizer to construct training/inference pipeline
from rlcontrol.rlcontrol import Organizer
from rlcontrol.utils.utils_path import get_project_path

ddpg_cruise_config = {
    'batch_size' : 256,
    'hidden_dim' : 32,
    'policy_net' : PolicyNetwork,
    'value_net' : ValueNetwork
}

# In each time start with random state
env_config = CruiseSystem.get_config_default()
env_config['random_state_init'] = True

# Create organizer
organizer = Organizer(
    env=CruiseSystem,
    agent_class=DDPG,
    agent_config = ddpg_cruise_config,
    env_config=env_config)

# Trained agent path relative to project folder
agent_path= get_project_path() + "/Logs/Agents/DDPG_2021_7_4_18_20_39/agent_38.pth"

# Set inference config
config_inference = organizer.get_default_inference_config()
config_inference["max_episode"] = 5
config_inference["max_step"] = 1000
organizer.inference(agent_path,config_inference)

