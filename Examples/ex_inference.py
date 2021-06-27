# Import system to be controlled
from rlcontrol.systems.double_tank_old import ENV
# Import RL algorithm
from rlcontrol.agents.ddpg import DDPG
# Import organizer to construct training/inference pipeline
from rlcontrol.rlcontrol import Organizer
# Import path utils to auto-detect your homepath
from rlcontrol.utils.utils_path import get_project_path
inference_organizer = Organizer(
    env=ENV,
    agent_class=DDPG)
your_home_path = get_project_path() #"/home/anton/coding/repos/"
best_weight = your_home_path+"/Logs/Agents/DDPG_2021_1_14_23_32_47/agent_best.pth"
inference_organizer.inference(best_weight)
