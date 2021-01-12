
import numpy as np
from numpy.lib.function_base import select
from tensorboardX import SummaryWriter

#TODO : Rename 
class Organizer(object):
    def __init__(self, env, agent, controller, log_dir, batch_size=64):
        self.env = env
        self.agent = agent
        self.controller = controller
        self.log_dir = log_dir
        self.config = self.get_default_training_config()
        # Place to save traiend weights
        self.log_weight_dir = ""
        # Place to save tensorboard outputs
        self.log_tensorboard_dir = ""

    def set_log_directories(self):
        algorithm_full_name = self.agent.algorithm_name
        algorithm_relative_name = None
        project_abs_path = None
        # TODO: Add hyperparameters to tensorboard
        self.writer = SummaryWriter(comment="-"+algorithm_full_name,log_dir=project_abs_path+"/runs/"+algorithm_relative_name+'/')

    @staticmethod
    def get_default_training_config():
        training_config = {
            'max_episode' : 5000,
            'max_step' : 500,
            # Frequency of logging trained weights
            'freq_weight_log' : 50, 
            # Frequency of logging training stats to tensorboard
            'freq_tensorboard_log' : 50, 
        }
        return training_config

    def train(self, training_config=None):
        if not training_config:
            self.config = training_config
        
        best_reward = -2000
        batch_size = self.agent.get_batch_size()

        for eps in range(self.config['max_episode']):
            state = self.env.reset()
            self.agent.reset()

            episode_reward = 0
            episode_policy_loss = 0
            episode_value_loss = 0
            total_control_signal = 0
            total_output_signal = 0

            for step in range(self.config['max_step']):
                action = self.agent.apply(state, step)

                total_control_signal = total_control_signal + action
            
                next_state, reward, done = self.env.step(action,step)

                y1 = np.asscalar(self.env.y)
                u1 = np.asscalar(action[0])
                
                total_control_signal += u1
                total_output_signal += y1


                self.agent.update_memory(state, action, reward, next_state, done)
                
                episode_reward = episode_reward + reward

                if len(self.agent.replay_buffer) > batch_size:
                    value_loss,policy_loss = self.agent.update_agent(batch_size)
                    episode_policy_loss += policy_loss
                    episode_value_loss += value_loss

                if done:
                    state = self.env.reset()                    
                    self.writer.add_scalar("Train/reward", episode_reward, eps)
                    self.writer.add_scalar("Train/policy_loss", episode_policy_loss, eps)
                    self.writer.add_scalar("Train/value_loss", episode_value_loss, eps)
                    self.writer.add_scalar("Train/mean_control_signal", np.mean(total_control_signal), eps)
                    self.writer.add_scalar("Train/mean_output_signal", np.mean(total_output_signal), eps)

                state = next_state
            
            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(eps+1,np.asscalar(env.y_set),np.asscalar(env.y),episode_reward)
            print(str1);print("\n*******************************\n")
            
            
            # Saving Model
            self.agent.save(self.log_weight_dir+'agent_'+eps+'.pth')


            # Save best model seperately
            if(episode_reward > best_reward) : 
                self.agent.save(self.log_weight_dir+'agent_best.pth')
                best_reward = episode_reward

    def inference(self, inference_config=None):
        raise(NotImplementedError)
    