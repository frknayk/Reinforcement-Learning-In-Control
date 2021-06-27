from gym_control.double_tank_old import ENV
from agents.ddpg import DDPG
train_organizer = Organizer(
    env=ENV,
    agent_class=DDPG)
# train_organizer.train()
your_home_path = "/home/anton/coding/repos/"
best_weight=your_home_path+"Reinforcement-Learning-In-Control/Logs/Agents/DDPG_2021_1_14_23_32_47/agent_best.pth"
train_organizer.inference(best_weight)
