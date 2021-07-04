import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
from tensorboardX import SummaryWriter
from rlcontrol.agents.base import Agent
from rlcontrol.systems.base_systems.base_env import ENV
from rlcontrol.utils.utils_path import create_log_directories
np.random.seed(59)


#TODO : Rename 
class Organizer(object):
    def __init__(self, 
        env:ENV, 
        agent_class:Agent,
        agent_config:dict,
        env_config:dict):
        self.env = env()
        self.agent = agent_class(
            state_dim=self.env.dimensions['state'],
            action_dim=self.env.dimensions['action'],
            agent_config = agent_config)
        self.config = self.get_default_training_config()
        # Place to save traiend weights
        self.log_weight_dir = ""
        # Place to save tensorboard outputs
        self.log_tensorboard_dir = ""

    def set_log_directories(self,algorithm_name="DDPG"):
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
        default_config_training = {
            'enable_log' : True, # Create log folders and log weights and plots
            'max_episode' : 10,
            'max_step' : 500,
            'freq_weight_log' : 50, # Frequency of logging trained weights 
            'freq_tensorboard_log' : 50, # Frequency of logging training stats to tensorboard
            'algorithm_name' : 'Unknown',
            'plotting' : {
                'freq' : 10, # Plot per episode
                'enable' : False}
        }
        return default_config_training

    @staticmethod
    def get_default_inference_config():
        default_config_inference = {
            'max_episode' : 10,
            'max_step' : 500,
        }
        return default_config_inference

    def train(self, training_config=None):
        self.config = self.get_default_training_config() if training_config is None else training_config        
 
        # TODO: Move this to logging class when created
        if self.config.get("enable_log") is True:
            self.set_log_directories(self.config.get("algorithm_name"))

        best_reward = -99999
        batch_size = self.agent.get_batch_size()

        for eps in range(self.config['max_episode']):    
            # Log plot list
            output_list = []
            reward_list=[]
            reference_list=[]
            control_sig_list=[]

            # Reset agent to random state
            self.agent.reset()

            # Zero log params 
            # TODO: Make this config
            episode_reward = 0
            episode_policy_loss = 0
            episode_value_loss = 0
            total_control_signal = 0
            total_output_signal = 0

            # Reset envrionment to random state
            state = self.env.reset()

            # One training loop
            for step in range(self.config['max_step']):
                action = self.agent.apply(state, step)
                if action is None:
                    print("NaN value detected, network is destroyed. Exiting training ..")
                    print("action : ",action)
                    if self.config.get("enable_log") is True: self.writer.close()
                    sys.exit()

                total_control_signal = total_control_signal + action
                next_state, reward, done = self.env.step(action)
                state = next_state
                output = self.env.get_info()['state'][0]

                if np.isnan(action) or np.isnan(state) or np.isnan(next_state) or np.isnan(done):
                    print("NaN value detected, network is destroyed. Exiting training ..")
                    print("state : ",state)
                    print("next state : ",next_state)
                    print("action : ",action)
                    print("reward : ",reward)
                    print("done :",done)
                    if self.config.get("enable_log") is True: self.writer.close()
                    sys.exit()

                y1 = np.asscalar(output)
                u1 = np.asscalar(action[0])
                
                output_list.append(y1)
                reference_list.append(self.env.get_info()['state_ref'][0])
                reward_list.append(reward)
                control_sig_list.append(action)

                total_control_signal += u1
                total_output_signal += y1

                self.agent.update_memory(state, action, reward, next_state, done)
                
                episode_reward = episode_reward + reward

                if len(self.agent.replay_buffer) > batch_size:
                    value_loss,policy_loss = self.agent.update_agent(batch_size)
                    episode_policy_loss += policy_loss
                    episode_value_loss += value_loss

                # TODO: Move this to logging class when created
                if done:
                    if self.config.get("enable_log") is True:
                        self.writer.add_scalar("Train/reward", episode_reward, eps)
                        self.writer.add_scalar("Train/policy_loss", episode_policy_loss, eps)
                        self.writer.add_scalar("Train/value_loss", episode_value_loss, eps)
                        self.writer.add_scalar("Train/mean_control_signal", np.mean(total_control_signal), eps)
                        self.writer.add_scalar("Train/mean_output_signal", np.mean(total_output_signal), eps)
                        self.writer.close()
                    break

            
            str_log = ""
            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
                eps+1,
                self.env.get_info()['state_ref'][0],
                np.asscalar(self.env.get_info()['state'][0]),
                episode_reward)
            str_log = str1 + " and lasted {0} steps".format(step)
            print(str_log)
            print("\n*******************************\n")

            # TODO: Move this to logging class when created
            # Saving Model
            if self.config.get("enable_log") is True:
                self.agent.save(self.log_weight_dir+'/agent_'+str(eps)+'.pth')

            if episode_reward > best_reward:
                best_reward = episode_reward
                # Save best model seperately
                if self.config.get("enable_log") is True: self.agent.save(self.log_weight_dir+'/agent_best.pth')

            # TODO: Create another class for plotting
            # Plot whithin some episodes
            if training_config['plotting']['enable'] is True and \
                eps % training_config['plotting']['freq'] == 0:
                fig, axs = plt.subplots(3)
                axs[0].set_title("Output vs Reference")
                axs[0].plot(output_list)
                axs[0].plot(reference_list)
                axs[1].set_title("Rewards")
                axs[1].plot(reward_list)
                axs[2].set_title("Control Signals")
                axs[2].plot(control_sig_list)
                plt.show()

    def inference(self, agent_path, inference_config=None):
        self.config = self.get_default_training_config() if inference_config is None else inference_config 

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

                y1 = np.asscalar(state[0])
                u1 = np.asscalar(action[0])
                ref = self.env.get_info().get("state_ref")
    
                output_list.append(y1)
                reference_list.append(ref)
                reward_list.append(reward)
                control_sig_list.append(action)


                total_control_signal += u1
                total_output_signal += y1
                
                episode_reward = episode_reward + reward
                
                state = next_state

            axs[0].set_title("Output vs Reference")
            axs[0].plot(output_list)
            axs[0].plot(reference_list)
            axs[1].set_title("Rewards")
            axs[1].plot(reward_list)
            axs[2].set_title("Control Signals")
            axs[2].plot(control_sig_list)
            plt.show()

            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
                eps+1,
                self.env.get_info().get("state_ref"),
                np.asscalar(self.env.get_info().get("state")[0]),
                episode_reward)
            print(str1)
            print("\n*******************************\n")
