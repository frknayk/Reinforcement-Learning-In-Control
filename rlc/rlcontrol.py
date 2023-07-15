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
            "enable_log_tensorboard": True,  # Create log folders and log weights and plots
            "max_episode": 10,
            "max_step": 500,
            "freq_weight_log": 50,  # Frequency of logging trained weights
            "freq_tensorboard_log": 50,  # Frequency of logging training stats to tensorboard
            "algorithm_name": "",
            "plotting": {"freq": 1, "enable": False},  # Plot per episode
            "freq_print_console": 1,
            "checkpoints": {"enable": True, "freq": 1},
            "plot_library": "streamlit",  # or Matplotlib
        }
        return default_config_training

    @staticmethod
    def get_default_inference_config():
        default_config_inference = {
            "enable_log_tensorboard": False,  # Create log folders and log weights and plots
            "max_episode": 10,
            "max_step": 500,
            "freq_weight_log": 50,  # Frequency of logging trained weights
            "freq_tensorboard_log": 50,  # Frequency of logging training stats to tensorboard
            "algorithm_name": "",
            "plotting": {"freq": 1, "enable": False},  # Plot per episode
            "freq_print_console": 1,
            "checkpoints": {"enable": False, "freq": 1},
            "plot_library": "streamlit",  # or Matplotlib
        }
        return default_config_inference

    def set_trainer_config(self, trainer_config: dict):
        self.config = trainer_config
        if trainer_config["checkpoints"]["enable"]:
            self.logger.set_checkpoints_dir(trainer_config["algorithm_name"])
            self.logger.save_experiment_config(
                env_config=self.env.env_config,
                agent_config=self.agent.config,
                train_config=trainer_config,
            )
            if trainer_config["enable_log_tensorboard"]:
                self.logger.set_tensorboard_dir(trainer_config["algorithm_name"])

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

    def test_agent(self):
        """Test agent for one episode. This mode for inference."""
        self.load_checkpoint()
        return self.train_one_episode(test_mode=True)

    def run(self, trainer_config: dict):
        self.set_trainer_config(trainer_config)
        for eps in tqdm(range(self.config["max_episode"]), "Agent Learning Progress: "):
            episode_result_dict = self.train_one_episode()
            self.log_train_iter(episode_result_dict, eps)
            if trainer_config["plot_library"] == "matplotlib":
                plot_matplotlib(episode_result_dict, eps)
            # if trainer_config["plot_library"] == 'streamlit':
            #     plot_streamlit(episode_result_dict, eps)

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
        # Saving Model
        if self.config["checkpoints"]["enable"]:
            if self.config["enable_log_tensorboard"]:
                self.logger.log_tensorboard_train(episode_result_dict, eps)
            if frequency_check(self.config["checkpoints"]["freq"], eps):
                self.agent.save(
                    self.logger.log_weight_dir + "/agent_" + str(eps) + ".pth"
                )
        # Save best model seperately
        if episode_result_dict["episode_reward"] > self.agent.best_reward:
            self.agent.best_reward = episode_result_dict["episode_reward"]
            if self.config["checkpoints"]["enable"] is True:
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
            if self.config.get("enable_log_tensorboard") is True:
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

    def load_checkpoint(self, checkpoint_dir=""):
        """Load agent

        Parameters
        ----------
        path : str, optional
            _description_, by default ""
        """
        if checkpoint_dir == "":
            checkpoint_dir = self.logger.log_weight_dir + "/agent_best.pth"
        self.agent.load(checkpoint_dir)


def frequency_check(freq, eps):
    if eps % freq == 0 and eps != 0:
        return True
    return False


if __name__ == "__main__":
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
        "dt": 1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 50,
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
    train_config["enable_log_tensorboard"] = True
    train_config["plot_library"] = "streamlit"  # or Matplotlib
    train_config["max_episode"] = 10
    train_config["algorithm_name"] = "DDPG"
    train_config["max_step"] = int(env_config["t_end"] / env_config["dt"])
    train_config["plotting"]["enable"] = True
    train_config["plotting"]["freq"] = 2
    train_config["freq_print_console"] = 1
    train_config["checkpoints"]["enable"] = True
    train_config["checkpoints"]["freq"] = 1
    train_organizer.run(train_config)
