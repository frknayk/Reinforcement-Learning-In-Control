import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

from rlc.agents.base import Agent
from rlc.logger.logger import Logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.random.seed(59)


# TODO : Rename
class Trainer(object):
    def __init__(
        self, env: gym.Env, agent_class: Agent, agent_config: dict, env_config: dict
    ):
        self.env = env(env_config)
        self.agent = agent_class(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            agent_config=agent_config,
        )
        self.config = self.get_default_training_config()
        self.logger = Logger()

    @staticmethod
    def get_default_training_config():
        default_config_training = {
            "enable_log": True,  # Create log folders and log weights and plots
            "max_episode": 10,
            "max_step": 500,
            "freq_weight_log": 50,  # Frequency of logging trained weights
            "freq_tensorboard_log": 50,  # Frequency of logging training stats to tensorboard
            "algorithm_name": "",
            "plotting": {"freq": 1, "enable": False},  # Plot per episode
            "freq_print_console": 1,
            "save_checkpoints": True,
        }
        return default_config_training

    @staticmethod
    def get_default_inference_config():
        default_config_inference = {
            "max_episode": 10,
            "max_step": 500,
        }
        return default_config_inference

    def train(self, training_config=None):
        self.config = (
            self.get_default_training_config()
            if training_config is None
            else training_config
        )

        # TODO: Move this to logging class when created
        if self.config.get("enable_log") is True:
            self.logger.set(self.config.get("algorithm_name"))

        best_reward = -99999
        batch_size = self.agent.get_batch_size()

        for eps in range(self.config["max_episode"]):
            # Reset agent to random observation
            self.agent.reset()

            # TODO: Make this dict
            episode_reward = 0
            episode_policy_loss = 0
            episode_value_loss = 0
            total_control_signal = 0
            total_output_signal = 0
            step_total = 0
            # Reset environment to random observation
            observation, info = self.env.reset()
            # Log plot list
            output_list = [float(observation)]
            reward_list = [0]
            reference_list = [self.env.env_config["y_ref"]]
            control_sig_list = [np.array([0])]

            # One training loop
            for step in range(self.config["max_step"]):
                state_dict = {
                    "state": observation,
                    "state_ref": np.array(self.env.env_config["y_ref"]),
                }
                action = self.agent.apply(state_dict=state_dict, step=step)
                if action is None:
                    print(
                        "NaN value detected, network is destroyed. Exiting training .."
                    )
                    print("action : ", action)
                    if self.config.get("enable_log") is True:
                        self.logger.close()
                    sys.exit()
                total_control_signal = total_control_signal + action
                if action[0] > 25:
                    dbg = True
                next_observation, reward, done, _, info = self.env.step(action)
                observation = next_observation
                # output = self.env._get_info()['states'][0]
                output = next_observation[0]

                if (
                    np.isnan(action)
                    or np.isnan(observation)
                    or np.isnan(next_observation)
                    or np.isnan(done)
                ):
                    print(
                        "NaN value detected, network is destroyed. Exiting training .."
                    )
                    print("observation : ", observation)
                    print("next observation : ", next_observation)
                    print("action : ", action)
                    print("reward : ", reward)
                    print("done :", done)
                    if self.config.get("enable_log") is True:
                        self.logger.close()
                    sys.exit()

                y1 = float(output)
                u1 = float(action[0])

                output_list.append(y1)
                # reference_list.append(self.env.get_info()['state_ref'][0])
                reference_list.append(self.env.env_config["y_ref"])
                reward_list.append(reward)
                control_sig_list.append(action)
                step_total += 1
                total_control_signal += u1
                total_output_signal += y1

                self.agent.update_memory(
                    observation, action, reward, next_observation, done
                )

                episode_reward = episode_reward + reward

                if len(self.agent.replay_buffer) > batch_size:
                    value_loss, policy_loss = self.agent.update_agent(batch_size)
                    episode_policy_loss += policy_loss.item()
                    episode_value_loss += value_loss.item()

                train_dict = {
                    "Train/reward": episode_reward,
                    "Train/policy_loss": episode_policy_loss,
                    "Train/value_loss": episode_value_loss,
                    "Train/mean_control_signal": np.mean(control_sig_list),
                    "Train/max_control_signal": np.max(control_sig_list),
                    "Train/min_control_signal": np.min(control_sig_list),
                    "Train/mean_output_signal": np.mean(output_list),
                    "Train/max_output": np.max(output_list),
                    "Train/min_output": np.min(output_list),
                }

                if done:
                    if self.config.get("enable_log") is True:
                        self.logger.log_tensorboard_train(train_dict, eps)
                    break
            if step_total == 1:
                dbg = True
            # Print progress
            train_print_dict = {
                "eps": eps + 1,
                # 'state_reference' : self.env.get_info()['state_ref'][0],
                "state_reference": self.env.env_config["y_ref"],
                "state": self.env._get_obs()[0],
                "reward": episode_reward,
                "step": step_total,
                "train_info": train_dict,
            }
            if step % self.config["freq_print_console"] == 0:
                self.logger.print_progress(train_print_dict)

            # TODO: Move this to logging class when created
            # Saving Model
            if self.config.get("save_checkpoints") is True:
                self.agent.save(
                    self.logger.log_weight_dir + "/agent_" + str(eps) + ".pth"
                )

            # Save best model seperately
            is_new_best = False
            if episode_reward > best_reward:
                best_reward = episode_reward
                if self.config.get("enable_log") is True:
                    self.agent.save(self.logger.log_weight_dir + "/agent_best.pth")
                is_new_best = True

            # TODO: Create another class for plotting
            # Plot whithin some episodes
            if is_new_best or (
                training_config["plotting"]["enable"] is True
                and eps % training_config["plotting"]["freq"] == 0
                and eps != 0
            ):
                fig, axs = plt.subplots(3, sharex=True)
                axs[0].set_title("Output vs Reference")
                axs[0].plot(output_list)
                axs[0].plot(reference_list)
                axs[1].set_title("Rewards")
                axs[1].plot(reward_list)
                axs[2].set_title("Control Signals")
                axs[2].plot(control_sig_list)
                plt.show()

        return True

    def inference(self, agent_path, inference_config=None):
        self.config = (
            self.get_default_training_config()
            if inference_config is None
            else inference_config
        )

        for eps in range(self.config["max_episode"]):
            # Saving Model
            self.agent.load(agent_path)

            observation = self.env.reset()
            self.agent.reset()

            episode_reward = 0
            total_control_signal = 0
            total_output_signal = 0

            # DEBUG
            fig, axs = plt.subplots(3)
            output_list = []
            reward_list = []
            reference_list = []
            control_sig_list = []

            for step in range(self.config["max_step"]):
                action = self.agent.apply(observation, step)

                total_control_signal = total_control_signal + action

                next_observation, reward, done = self.env.step(action)

                y1 = np.asscalar(observation[0])
                u1 = np.asscalar(action[0])
                ref = self.env.get_info().get("state_ref")

                output_list.append(y1)
                reference_list.append(ref)
                reward_list.append(reward)
                control_sig_list.append(action)

                total_control_signal += u1
                total_output_signal += y1

                episode_reward = episode_reward + reward

                observation = next_observation

            axs[0].set_title("Output vs Reference")
            axs[0].plot(output_list)
            axs[0].plot(reference_list)
            axs[1].set_title("Rewards")
            axs[1].plot(reward_list)
            axs[2].set_title("Control Signals")
            axs[2].plot(control_sig_list)
            plt.show()

            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
                eps + 1,
                self.env.get_info().get("state_ref"),
                np.asscalar(self.env.get_info().get("observation")[0]),
                episode_reward,
            )
            print(str1)
            print("\n*******************************\n")


if __name__ == "__main__":
    import gym_control
    from gym_control.envs import LinearSISOEnv
    from rlc.agents.ddpg import DDPG, PolicyNetwork, ValueNetwork

    agent_config = {
        "batch_size": 256,
        "hidden_dim": 32,
        "policy_net": PolicyNetwork,
        "value_net": ValueNetwork,
    }

    env_config = {
        "action_space": [0, 50],
        "obs_space": [0, 10],
        "num": [1],
        "den": [1, 10, 20],
        "x_0": [0],
        "dt": 0.1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 100,
        "y_ref": 1,
        "steady_state_indicator": 30,
    }

    train_organizer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )

    train_config = train_organizer.get_default_training_config()
    train_config["enable_log"] = True
    train_config["max_episode"] = 2000
    train_config["algorithm_name"] = "DDPG"
    train_config["max_step"] = int(env_config["t_end"] / env_config["dt"])
    train_config["plotting"]["enable"] = False
    train_config["plotting"]["freq"] = 2
    train_config["freq_print_console"] = 1
    train_config["save_checkpoints"] = False
    train_organizer.train(train_config)
