import json
import logging
import pathlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from tensorboardX import SummaryWriter

from rlc.utils.utils_path import (
    create_dir,
    create_log_directories,
    create_tensorboard_log_dir,
    get_algorithm_name_by_time,
    get_project_path,
)

np.random.seed(59)


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def create_console_logger(loger_name: str):
    # create logger with 'spam_application'
    logger = logging.getLogger(loger_name)
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


train_dict_default = {
    "Train/reward": list,
    "Train/policy_loss": list,
    "Train/value_loss": list,
    "Train/mean_control_signal": list,
    "Train/max_control_signal": list,
    "Train/min_control_signal": list,
    "Train/mean_output_signal": list,
    "Train/max_output": list,
    "Train/min_output": list,
}

train_progress_default = {
    "eps": int,
    "state_reference": float,
    "state": float,
    "reward": int,
    "step": int,
}


class Logger(object):
    def __init__(self, algorithm_name="DDPG"):
        self.algorithm_name = algorithm_name
        self.log_weight_dir = ""  # Directory of experiment checkpoints
        self.log_tensorboard_dir = ""  # Directory of tensorboard runs
        self.experiment_dir = ""
        self.logger_console = create_console_logger("logger_rlc_logger")
        self.project_abs_path = get_project_path()
        self.experiment_name = get_algorithm_name_by_time(algorithm_name)
        self.experiment_dir = str(
            pathlib.Path(self.project_abs_path, "Runs", self.experiment_name)
        )
        self.writer = None  # Tensorboard writer object

    def set_checkpoints_dir(self, algorithm_name):
        self.algorithm_name = algorithm_name
        create_log_directories(algorithm_name)
        self.log_weight_dir = str(pathlib.Path(self.experiment_dir, "checkpoints"))

    def set_tensorboard_dir(self, algorithm_name):
        self.algorithm_name = algorithm_name
        create_tensorboard_log_dir(algorithm_name)
        self.log_tensorboard_dir = str(pathlib.Path(self.experiment_dir, "runs"))
        # TODO: Add hyperparameters to tensorboard
        self.writer = SummaryWriter(
            comment="-" + self.algorithm_name, log_dir=self.log_tensorboard_dir
        )

    def get_experiment_pickle_dir(self):
        experiment_config_dir = self.get_logger_relative_path(
            "experiment_config.pickle"
        )
        return experiment_config_dir

    def get_logger_relative_path(self, path: str):
        return str(pathlib.Path(self.experiment_dir, path))

    def save_experiment_config(self, env_config, agent_config, train_config):
        experiment_config = {
            "env_config": env_config,
            "train_config": train_config,
            "agent_config": agent_config,
            "path_best_agent": str(
                pathlib.Path(self.experiment_dir, "checkpoints", "agent_best.pth")
            ),
        }
        experiment_config_dir = self.get_experiment_pickle_dir()
        with open(experiment_config_dir, "wb") as handle:
            pickle.dump(experiment_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def log_tensorboard_train(self, train_dict: train_dict_default, eps: int):
        for key in train_dict:
            value = train_dict[key]
            if type(value) != int or type(value) != float:
                continue
            self.writer.add_scalar(key, value, eps)
        self.writer.close()

    def print_progress(self, train_progress: train_progress_default, print_freq, step):
        if step % print_freq == 0:
            str_log = ""
            str1 = "Episode : [ {0} ] =>>> Ref/Out:[ {1} ] / OUT-1 : [ {2} ] \
                Total Reward : [ {3} ] -- Mean Reward: [ {4} ]".format(
                train_progress["eps"],
                train_progress["state_reference"],
                train_progress["state"],
                train_progress["reward"],
                train_progress["reward_mean"],
            )
            str_log = str1 + " and lasted {0} steps".format(train_progress["step"])
            str_log += "\n*******************************\n"
            self.logger_console.info(str_log)

    def close(self):
        self.writer.close()
