import os
import sys
from typing import List

import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rlc.agents.base import Agent
from rlc.logger.logger import Logger
from rlc.utils.plot import plot_matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

np.random.seed(59)

# Default 1-episode output structure
episode_result_dict_default = {
    "episode_reward": float,
    "episode_policy_loss": float,
    "episode_value_loss": float,
    "total_control_signal": float,
    "total_output_signal": float,
    "step_total": int,
    "output_list": List[float],
    "reward_list": List[float],
    "reference_list": List[float],
    "control_sig_list": List[float],
    "sim_time": List[float],
}


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
            "checkpoints": {"enable": True, "freq": 1},
        }
        return default_config_training

    @staticmethod
    def get_default_inference_config():
        default_config_inference = {
            "max_episode": 10,
            "max_step": 500,
        }
        return default_config_inference

    def set_training_config(self, train_config: dict):
        self.config = train_config
        self.logger.set(train_config["algorithm_name"])

    def train_one_episode(self, test_mode: bool = False):
        """Run simulation for one episode and train by default.

        Parameters
        ----------
        test_mode : bool, optional
            If True, agent's parameters will not be updated(inference mode), \
                by default False

        Returns
        -------
        _type_
            _description_
        """

        self.agent.reset()
        observation, _ = self.env.reset()
        agent_loss_dict = None
        episode_result_dict = episode_result_dict_default.copy()
        episode_result_dict["output_list"] = []
        episode_result_dict["reward_list"] = []
        episode_result_dict["reference_list"] = []
        episode_result_dict["control_sig_list"] = []
        episode_result_dict["step_total"] = 0
        episode_result_dict["episode_reward"] = 0
        episode_result_dict["episode_policy_loss"] = 0
        episode_result_dict["episode_value_loss"] = 0
        episode_result_dict["total_control_signal"] = 0
        episode_result_dict["total_output_signal"] = 0
        # Reset environment to random observation
        observation, _ = self.env.reset()
        for step in range(self.config["max_step"]):
            state_dict = {
                "state": observation,
                "state_ref": np.array(self.env.env_config["y_ref"]),
            }
            action = self.agent.apply(state_dict=state_dict, step=step)
            next_observation, reward, done, _, _ = self.env.step(action)
            self._check_output(action, observation, next_observation, done)
            observation = next_observation
            if not test_mode:
                self.agent.update_memory(
                    observation, action, reward, next_observation, done
                )
                agent_loss_dict = self.agent.update_agent(step)
            episode_result_dict = self._update_iter_dict(
                episode_result_dict,
                next_observation,
                reward,
                action,
                self.env.env_config["y_ref"],
                agent_loss_dict,
            )
            if done:
                break
        sim_results = np.array(self.env.sim_results)
        episode_result_dict["sim_time"] = list(sim_results[:, 0])
        return episode_result_dict

    def train_step(self):
        yield self.train_one_episode()

    def test_best_agent(self):
        self.load_best_agent()
        return self.train_one_episode(test_mode=True)

    def run(self, training_config: dict):
        self.set_training_config(training_config)
        # TODO: Move this to logging class when created
        for eps in tqdm(range(self.config["max_episode"]), "Agent Learning Progress: "):
            episode_result_dict = self.train_one_episode()
            self.log_train_iter(episode_result_dict, eps)
            plot_matplotlib(episode_result_dict, eps)

    def log_train_iter(self, episode_result_dict, eps):
        # Print progress
        train_print_dict = {
            "eps": eps + 1,
            "state_reference": self.env.env_config["y_ref"],
            "state": self.env._get_obs()[0],
            "reward": episode_result_dict["episode_reward"],
            "step": episode_result_dict["step_total"],
        }
        self.logger.print_progress(
            train_print_dict, self.config["freq_print_console"], eps
        )
        self.logger.log_tensorboard_train(
            episode_result_dict, eps, self.config["enable_log"]
        )
        # Saving Model
        if self.config["checkpoints"]["enable"]:
            if frequency_check(self.config["checkpoints"]["freq"], eps):
                self.agent.save(
                    self.logger.log_weight_dir + "/agent_" + str(eps) + ".pth"
                )
        # Save best model seperately
        if episode_result_dict["episode_reward"] > self.agent.best_reward:
            self.agent.best_reward = episode_result_dict["episode_reward"]
            if self.config.get("enable_log") is True:
                self.agent.save(self.logger.log_weight_dir + "/agent_best.pth")
        if self.config["plotting"]["enable"]:
            if frequency_check(self.config["plotting"]["freq"], eps):
                plot_matplotlib(episode_result_dict, eps)

    def _check_output(self, action, observation, next_observation, done):
        if (
            np.isnan(action)
            or np.isnan(observation)
            or np.isnan(next_observation)
            or np.isnan(done)
        ):
            self.logger.logger_console.error(
                "NaN value detected, network is destroyed. Exiting training .."
            )
            if self.config.get("enable_log") is True:
                self.logger.close()
            sys.exit()

    def _update_iter_dict(
        self,
        episode_result_dict,
        next_observation,
        reward,
        action,
        y_ref,
        agent_loss_dict,
    ):
        episode_result_dict["output_list"].append(float(next_observation[0]))
        episode_result_dict["reference_list"].append(y_ref)
        episode_result_dict["reward_list"].append(reward)
        episode_result_dict["control_sig_list"].append(action[0])
        episode_result_dict["step_total"] = episode_result_dict["step_total"] + 1
        episode_result_dict["total_output_signal"] = episode_result_dict[
            "total_output_signal"
        ] + float(next_observation[0])
        episode_result_dict["total_control_signal"] = episode_result_dict[
            "total_control_signal"
        ] + float(action[0])
        episode_result_dict["episode_reward"] = (
            episode_result_dict["episode_reward"] + reward
        )
        if agent_loss_dict is not None:
            episode_result_dict["episode_policy_loss"] = agent_loss_dict["policy_loss"]
            episode_result_dict["episode_value_loss"] = agent_loss_dict["value_loss"]
        return episode_result_dict

    def load_best_agent(self):
        self.agent.load(self.logger.log_weight_dir + "/agent_best.pth")


def frequency_check(freq, eps):
    if eps % freq == 0 and eps != 0:
        return True
    return False


def inference(self, agent_path, inference_config=None):
    self.config = (
        self.get_default_training_config()
        if inference_config is None
        else inference_config
    )
    self.agent.load(agent_path)
    for eps in range(self.config["max_episode"]):
        self.agent.reset()
        observation = self.env.reset()
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
    train_config["plot_library"] = "streamlit"  # or Matplotlib
    train_config["max_episode"] = 2000
    train_config["algorithm_name"] = "DDPG"
    train_config["max_step"] = int(env_config["t_end"] / env_config["dt"])
    train_config["plotting"]["enable"] = True
    train_config["plotting"]["freq"] = 2
    train_config["freq_print_console"] = 1
    train_config["save_checkpoints"] = False
    train_config["checkpoints"]["enable"] = False
    train_config["checkpoints"]["freq"] = 1
    train_organizer.run(train_config)
