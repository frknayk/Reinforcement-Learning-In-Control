import pathlib
import sys

import numpy as np
from tensorboardX import SummaryWriter

from rlc.utils.utils_path import create_dir, create_log_directories

np.random.seed(59)

train_dict_default = {
    "reward": None,
    "loss_policy": None,
    "loss_value": None,
    "mean_control_signal": None,
    "mean_output_signal": None,
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

    def log_tensorboard_train(self, train_dict: train_dict_default, eps: int):
        for key in train_dict:
            value = train_dict[key]
            self.writer.add_scalar(key, value, eps)
        self.writer.close()

    def print_progress(self, train_progress: train_progress_default):
        str_log = ""
        str1 = "Trial : [ {0} ] is completed with reference : [ {1} ]\nOUT-1 : [ {2} ]\nEpisode Reward : [ {3} ]".format(
            train_progress["eps"],
            train_progress["state_reference"],
            train_progress["state"],
            train_progress["reward"],
        )
        str_log = str1 + " and lasted {0} steps".format(train_progress["step"])
        print(str_log)
        asd = train_progress["train_info"]
        print(f"---- train info ----\n{asd}")
        print("\n*******************************\n")

    def close(self):
        self.writer.close()
