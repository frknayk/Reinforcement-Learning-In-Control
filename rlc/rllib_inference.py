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
from ray.rllib.algorithms.algorithm import Algorithm, ExportFormat
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.schedulers import AsyncHyperBandScheduler

import gym_control
from gym_control.envs import LinearSISOEnv

######## ENV CONFIG #############
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

####### CREATE ENV ###########
register(
    id="gym_control/LinearSISOEnv-v0",
    entry_point="gym_control.envs:LinearSISOEnv",
    max_episode_steps=3000,
)
env = gymnasium.make("gym_control/LinearSISOEnv-v0", env_config=env_cfg)


def env_creator(env_config):
    return LinearSISOEnv(env_config)


register_env("LinearSISOEnv", lambda config: LinearSISOEnv(env_cfg))


def inference(checkpoint_dir: str, max_eps: int = 100):
    algo = None
    try:
        algo = Algorithm.from_checkpoint(checkpoint_dir)
    except Exception as e:
        raise (f"Could not load checkpoint: {e}")
        return
    # Simulate
    obs, info = env.reset()
    for x in range(max_eps):
        # obs_cuda = torch.from_numpy(obs).cuda(0)
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
        print(obs)


# Will be read from outside
checkpoint_dir = r"\Users\Furka\ray_results\DDPG\DDPG_LinearSISOEnv_1bbda_00000_0_2023-07-11_00-24-09\checkpoint_000004"
checkpoint_dir = str(pathlib.PurePath(checkpoint_dir))
inference(checkpoint_dir)

# algo_policy = Algorithm.from_checkpoint(checkpoint_dir)
# path_cwd = pathlib.Path(__file__).parent.resolve()
# export_dir = pathlib.Path(path_cwd , "exported_policies")
# export_dir_dict = algo_policy.export_model(export_formats=ExportFormat.MODEL,export_dir=export_dir)
