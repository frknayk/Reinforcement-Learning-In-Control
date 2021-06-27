import os
import sys
from datetime import datetime

PROJECT_NAME = 'Reinforcement-Learning-In-Control'

def create_log_directories(algo_name):
    project_abs_path = get_project_path()
    algorithm_relative_name = get_algorithm_name_by_time(algo_name)
    results_dir = project_abs_path + "/Logs/Agents/" + algorithm_relative_name
    try:
        os.makedirs(results_dir)
    except Exception as e:
        logging.error("Could not created the agents folder : ",e)
    return project_abs_path, algorithm_relative_name

def get_project_path():
    main_path = os.path.abspath(__file__)
    index = main_path.find("/"+PROJECT_NAME)
    if index == -1:
        logging.error("Could not find the repository name, be sure not to renamed")
        return None
    main_dirname = main_path[:index]+"/"+PROJECT_NAME
    return main_dirname

def get_algorithm_name_by_time(algo_name:str):
    today = datetime.now()
    todays_date_full =  str(today.year)+"_"+str(today.month)+"_"+str(today.day)+"_"
    todays_date_full += str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    return algo_name + "_" + todays_date_full