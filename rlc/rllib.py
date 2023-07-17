import os
import pathlib
from datetime import date

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from gymnasium.envs.registration import register
from ray import air, tune
from ray.air.config import CheckpointConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.schedulers import AsyncHyperBandScheduler

import gym_control
from gym_control.envs import LinearSISOEnv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


env_cfg = {
    "action_space": [-1, 1],
    "obs_space": [-10, 10],
    "num": [1],
    "den": [1, 10, 20],
    "x_0": [0],
    "dt": 0.01,
    "y_0": 0,
    "t_0": 0,
    "t_end": 5,
    "y_ref": 1,
}


register(
    id="gym_control/LinearSISOEnv-v0",
    entry_point="gym_control.envs:LinearSISOEnv",
    max_episode_steps=3000,
)

env = gymnasium.make("gym_control/LinearSISOEnv-v0", env_config=env_cfg)


def env_creator(env_config):
    return LinearSISOEnv(env_config)


register_env("LinearSISOEnv", lambda config: LinearSISOEnv(env_cfg))


ray.init()


def train_ppo():
    config = PPOConfig().training(lr=0.1)
    config.environment(env=LinearSISOEnv, env_config=env_cfg)
    config.rollouts(num_rollout_workers=1)
    config.framework("torch")
    config.checkpointing(export_native_model_files=True)
    config.resources(num_gpus=1)
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={"episode_reward_mean": -1.1},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
        param_space=config,
    )
    results_ppo_tuner = tuner.fit()
    return results_ppo_tuner


def train_ddpg():
    config = DDPGConfig().training(lr=0.1)
    config.environment(env=LinearSISOEnv, env_config=env_cfg)
    config.rollouts(num_rollout_workers=1)
    config.framework("torch")
    config.checkpointing(export_native_model_files=True)
    config.resources(num_gpus=1)

    scheduler = AsyncHyperBandScheduler(time_attr="training_iteration")

    tuner = tune.Tuner(
        "DDPG",
        run_config=air.RunConfig(
            stop={"episode_reward_mean": -1.1},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=1, checkpoint_at_end=True
            ),
        ),
        param_space=config,
        # # For 3 iteration if any progress does not happen in the metric, stop exp
        # tune_config=tune.TuneConfig(
        #         scheduler=scheduler, num_samples=3, metric="episode_reward_mean", mode="max"),
    )

    results_ddpg_tuner = tuner.fit()
    return results_ddpg_tuner


# train_ppo()
train_ddpg()
