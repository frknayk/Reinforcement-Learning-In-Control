import os
import sys
from typing import NewType
import numpy as np
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
from agents.base import Agent
from gym_control.base import ENV

import matplotlib.pyplot as plt

np.random.seed(59)


PROJECT_NAME = 'Reinforcement-Learning-In-Control'

def create_log_directories(algo_name):
    project_abs_path = get_project_path()
    algorithm_relative_name = get_algorithm_name_by_time(algo_name)
    results_dir = project_abs_path + "/Logs/Agents/" + algorithm_relative_name
    try:
        os.makedirs(results_dir)
    except Exception as e:
        logging.error("Could not created the agents folder : ",e)
    return project_abs_path, algorithm_relative_name

def get_project_path():
    main_path = os.path.abspath(__file__)
    index = main_path.find("/"+PROJECT_NAME)
    if index == -1:
        logging.error("Could not find the repository name, be sure not to renamed")
        return None
    main_dirname = main_path[:index]+"/"+PROJECT_NAME
    return main_dirname

def get_algorithm_name_by_time(algo_name:str):
    today = datetime.now()
    todays_date_full =  str(today.year)+"_"+str(today.month)+"_"+str(today.day)+"_"
    todays_date_full += str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    return algo_name + "_" + todays_date_full

#TODO : Rename 
class Organizer(object):
    def __init__(self, env:ENV, agent_class:Agent, batch_size=64):
        self.env = env()
        self.agent = agent_class(state_dim=self.env.state_dim, action_dim=self.env.action_dim)
        self.config = self.get_default_training_config()
        # Place to save traiend weights
        self.log_weight_dir = ""
        # Place to save tensorboard outputs
        self.log_tensorboard_dir = ""

    def set_log_directories(self):
        # algorithm_full_name = sself.agent.get_algorithm_name()
        algorithm_name = "DDPG"
        algorithm_relative_name = None
        project_abs_path = None
        project_abs_path, algorithm_relative_name = create_log_directories(algorithm_name)
        project_abs_path = project_abs_path + "/Logs/"
        self.log_weight_dir = project_abs_path + "Agents/" + algorithm_relative_name
        self.log_tensorboard_dir = project_abs_path+"runs/"+algorithm_relative_name
        # TODO: Add hyperparameters to tensorboard
        self.writer = SummaryWriter(comment="-"+algorithm_name,log_dir=self.log_tensorboard_dir+'/')

    @staticmethod
    def get_default_training_config():
        training_config = {
            'max_episode' : 10,
            'max_step' : 500,
            # Frequency of logging trained weights
            'freq_weight_log' : 50, 
            # Frequency of logging training stats to tensorboard
            'freq_tensorboard_log' : 50, 
        }
        return training_config

    def train(self, training_config=None):
        self.set_log_directories()
        self.config = self.get_default_training_config() if training_config is None else training_config         
        best_reward = -200000
        batch_size = self.agent.get_batch_size()
        for eps in range(self.config['max_episode']):
            
            self.agent.reset()

            episode_reward = 0
            episode_policy_loss = 0
            episode_value_loss = 0
            total_control_signal = 0
            total_output_signal = 0

            # DEBUG
            fig, axs = plt.subplots(3)
            output_list = []
            reward_list=[]
            reference_list=[]
            control_sig_list=[]

            state = self.env.reset()

            for step in range(self.config['max_step']):
                action = self.agent.apply(state, step)

                total_control_signal = total_control_signal + action
            
                next_state, reward, done = self.env.step(action)
                state = next_state
                y1 = np.asscalar(self.env.y)
                u1 = np.asscalar(action[0])
                
                # msg1 = "Y(t)/R(t): {0}/{1}".format(y1,self.env.y_set)
                # print(msg1)
                output_list.append(y1)
                reference_list.append(self.env.y_set)
                reward_list.append(reward)
                control_sig_list.append(action)
                # # PLOT
                # axs[0].clear()
                # axs[1].clear()
                # axs[2].clear()

                total_control_signal += u1
                total_output_signal += y1

                self.agent.update_memory(state, action, reward, next_state, done)
                
                episode_reward = episode_reward + reward

                if len(self.agent.replay_buffer) > batch_size:
                    value_loss,policy_loss = self.agent.update_agent(batch_size)
                    episode_policy_loss += policy_loss
                    episode_value_loss += value_loss

                if done:               
                    self.writer.add_scalar("Train/reward", episode_reward, eps)
                    self.writer.add_scalar("Train/policy_loss", episode_policy_loss, eps)
                    self.writer.add_scalar("Train/value_loss", episode_value_loss, eps)
                    self.writer.add_scalar("Train/mean_control_signal", np.mean(total_control_signal), eps)
                    self.writer.add_scalar("Train/mean_output_signal", np.mean(total_output_signal), eps)
                    break

            # axs[0].set_title("Output vs Reference")
            # axs[0].plot(output_list)
            # axs[0].plot(reference_list)
            # axs[1].set_title("Reward List")
            # axs[1].plot(reward_list)
            # axs[2].set_title("Control Signal List")
            # axs[2].plot(control_sig_list)
            # plt.show()

            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
                eps+1,
                self.env.y_set,
                np.asscalar(self.env.y),
                episode_reward)
            print(str1)
            print("\n*******************************\n")
            
            # Saving Model
            self.agent.save(self.log_weight_dir+'/agent_'+str(eps)+'.pth')


            # Save best model seperately
            if(episode_reward > best_reward) : 
                self.agent.save(self.log_weight_dir+'/agent_best.pth')
                best_reward = episode_reward

    def inference(self, agent_path, inference_config=None):
        self.config = self.get_default_training_config() if inference_config is None else inference_config 
        self.config['max_step'] = 5500
        self.config['max_episode'] = 100

        for eps in range(self.config['max_episode']):
            # Saving Model
            self.agent.load(agent_path)

            state = self.env.reset()
            self.agent.reset()

            episode_reward = 0
            total_control_signal = 0
            total_output_signal = 0

            # DEBUG
            fig, axs = plt.subplots(3)
            output_list = []
            reward_list=[]
            reference_list=[]
            control_sig_list=[]

            for step in range(self.config['max_step']):
                action = self.agent.apply(state, step)

                total_control_signal = total_control_signal + action
            
                next_state, reward, done = self.env.step(action)

                y1 = np.asscalar(self.env.y)
                u1 = np.asscalar(action[0])
                
                # msg1 = "Y(t)/R(t): {0}/{1}".format(y1,self.env.y_set)
                # print(msg1)
                output_list.append(y1)
                reference_list.append(self.env.y_set)
                reward_list.append(reward)
                control_sig_list.append(action)


                total_control_signal += u1
                total_output_signal += y1
                
                episode_reward = episode_reward + reward
                
                state = next_state
                if done:
                    break

            axs[0].set_title("Output vs Reference")
            axs[0].plot(output_list)
            axs[0].plot(reference_list)
            axs[1].set_title("Reward List")
            axs[1].plot(reward_list)
            axs[2].set_title("Control Signal List")
            axs[2].plot(control_sig_list)
            plt.show()

            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
                eps+1,
                self.env.y_set,
                np.asscalar(self.env.y),
                episode_reward)
            print(str1)
            print("\n*******************************\n")
            

if __name__ == "__main__":
    from gym_control.double_tank_old import ENV
    from agents.ddpg import DDPG
    train_organizer = Organizer(
        env=ENV,
        agent_class=DDPG)
    # train_organizer.train()
    your_home_path = "/home/anton/coding/repos/"
    best_weight=your_home_path+"Reinforcement-Learning-In-Control/Logs/Agents/DDPG_2021_1_14_23_32_47/agent_best.pth"
    train_organizer.inference(best_weight)
