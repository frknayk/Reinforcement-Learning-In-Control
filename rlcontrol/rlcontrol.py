import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import Inf
from tensorboardX import SummaryWriter
from rlcontrol.agents.base import Agent
from rlcontrol.systems.base_systems.base_env import ENV
from rlcontrol.logger.logger import Logger

np.random.seed(59)


#TODO : Rename 
class Organizer(object):
    def __init__(self, 
        env:ENV, 
        agent_class:Agent,
        agent_config:dict,
        env_config:dict):
        self.env = env(env_config)
        self.agent = agent_class(
            state_dim=self.env.dimensions['state'],
            action_dim=self.env.dimensions['action'],
            agent_config = agent_config)
        self.config = self.get_default_training_config()
        self.logger = Logger()

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
            self.logger.set(self.config.get("algorithm_name"))

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
                    if self.config.get("enable_log") is True: self.logger.close()
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
                    if self.config.get("enable_log") is True: self.logger.close()
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

                if done:
                    if self.config.get("enable_log") is True:
                        train_dict = {
                            "Train/reward" : episode_reward,
                            "Train/policy_loss" : episode_policy_loss,
                            "Train/value_loss" : episode_value_loss,
                            "Train/mean_control_signal" : np.mean(total_control_signal),
                            "Train/mean_output_signal" :  np.mean(total_output_signal)}
                        self.logger.log_tensorboard_train(train_dict, eps)
                    break
            
            # Print progress
            train_print_dict = {
                'eps' : eps+1,
                'state_reference' : self.env.get_info()['state_ref'][0],
                'state' : np.asscalar(self.env.get_info()['state'][0]),
                'reward' : episode_reward,
                'step' : step,
            }
            self.logger.print_progress(train_print_dict)

            # TODO: Move this to logging class when created
            # Saving Model
            if self.config.get("enable_log") is True:
                self.agent.save(self.logger.log_weight_dir+'/agent_'+str(eps)+'.pth')

            # Save best model seperately
            if episode_reward > best_reward:
                best_reward = episode_reward
                if self.config.get("enable_log") is True: self.agent.save(
                    self.logger.log_weight_dir+'/agent_best.pth')

            # TODO: Create another class for plotting
            # Plot whithin some episodes
            if training_config['plotting']['enable'] is True and \
                eps % training_config['plotting']['freq'] == 0 and \
                eps != 0:
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
