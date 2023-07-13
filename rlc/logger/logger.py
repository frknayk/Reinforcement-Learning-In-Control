import logging
import pathlib

import numpy as np
import pandas as pd
import streamlit as st
from tensorboardX import SummaryWriter

from rlc.utils.utils_path import create_dir, create_log_directories

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
        # Place to save traiend weights
        self.log_weight_dir = ""
        # Place to save tensorboard outputs
        self.log_tensorboard_dir = ""
        self.writer = None
        self.experiment_dir = ""
        self.logger_console = create_console_logger("RLC")

    def set(self, algorithm_name):
        project_abs_path, experiment_name = create_log_directories(algorithm_name)
        self.experiment_dir = str(
            pathlib.Path(project_abs_path, "Runs", experiment_name)
        )
        self.log_weight_dir = str(pathlib.Path(self.experiment_dir, "checkpoints"))
        self.log_tensorboard_dir = str(pathlib.Path(self.experiment_dir, "runs"))
        # TODO: Add hyperparameters to tensorboard
        self.writer = SummaryWriter(
            comment="-" + algorithm_name, log_dir=self.log_tensorboard_dir
        )

    def log_tensorboard_train(
        self, train_dict: train_dict_default, eps: int, enable=False
    ):
        for key in train_dict:
            value = train_dict[key]
            if type(value) != int or type(value) != float:
                continue
            self.writer.add_scalar(key, value, eps)
        self.writer.close()

    def print_progress(self, train_progress: train_progress_default, print_freq, step):
        if step % print_freq == 0:
            str_log = ""
            str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : \
                [ {2} ]\nEpisode Reward : [ {3} ]".format(
                train_progress["eps"],
                train_progress["state_reference"],
                train_progress["state"],
                train_progress["reward"],
            )
            str_log = str1 + " and lasted {0} steps".format(train_progress["step"])
            print(str_log)
            # asd = train_progress["train_info"]
            # print(f"---- train info ----\n{asd}")
            print("\n*******************************\n")

    def close(self):
        self.writer.close()
