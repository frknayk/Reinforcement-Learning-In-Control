import logging
import os
import pathlib
import sys
from datetime import datetime


def create_log_directories(algo_name: str):
    project_abs_path = get_project_path()
    experiment_name = get_algorithm_name_by_time(algo_name)
    create_dir(str(pathlib.Path(project_abs_path, "Runs")))
    create_dir(str(pathlib.Path(project_abs_path, "Runs", experiment_name)))
    create_dir(
        str(pathlib.Path(project_abs_path, "Runs", experiment_name, "checkpoints"))
    )
    create_dir(str(pathlib.Path(project_abs_path, "Runs", experiment_name, "runs")))


def create_tensorboard_log_dir(algo_name: str):
    project_abs_path = get_project_path()
    experiment_name = get_algorithm_name_by_time(algo_name)
    create_dir(str(pathlib.Path(project_abs_path, "Runs", experiment_name, "runs")))


def get_project_path():
    return str(pathlib.Path(__file__).parent.resolve().parent.parent)


def get_algorithm_name_by_time(algo_name: str):
    today = datetime.now()
    todays_date_full = (
        str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "_"
    )
    todays_date_full += (
        str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    )
    return algo_name + "_" + todays_date_full


def create_dir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        logging.info("Folder is already exist : ", path)
    except Exception as e:
        logging.error("Could not created the agents folder : ", e)
