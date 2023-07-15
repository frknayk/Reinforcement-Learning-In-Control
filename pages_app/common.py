import json

import numpy as np
import streamlit as st

import gym_control
from gym_control.envs import LinearSISOEnv
from rlc.agents.ddpg import PolicyNetwork, ValueNetwork
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer

logger = create_console_logger("logger_inference")


def select_algorithm():
    algorithm_selected = st.selectbox("Select the algorithm", ("DDPG", "PPO", "DQN"))
    if algorithm_selected != "DDPG":
        logger.error("Only DDPG is available for now")
        st.error("Only DDPG is available for now")
    return algorithm_selected


def convert_np_array(text_list):
    return np.float32(json.loads(text_list))


def create_tab_tf(tab_tf):
    """Create transfer function tab"""
    numerator = None
    denum = None
    with tab_tf:
        numerator = st.text_input(
            "Numerator of transfer function (Enter as list of numbers: [1,2,3]) ",
            value="[1]",
        )
        denum = st.text_input(
            "Denumerator of transfer function (Enter as list of numbers: [1,2,3]) ",
            value="[1, 10, 20]",
        )
    return numerator, denum


# TODO: Make this flexible with different algorithms like PPO
def create_tab_agent(tab_agent):
    agent_config = None
    with tab_agent:
        agent_config = {}
        batch_size = st.number_input(
            "batch_size", value=128, min_value=0, max_value=2048
        )
        hidden_dim = st.number_input(
            "hidden_dim", value=64, min_value=16, max_value=1024
        )
        agent_config = {
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "policy_net": PolicyNetwork,
            "value_net": ValueNetwork,
        }
    return agent_config


def create_tab_env(tab_env, numerator, denum):
    """Create gymnasium environment"""
    env_config = {}
    with tab_env:
        with st.expander("Action/Observation Space Bounds"):
            action_space_low = st.number_input("action_space_low", step=1, value=-1)
            action_space_high = st.number_input("action_space_high", step=1, value=50)
            assert action_space_high > action_space_low
            obs_space_low = st.number_input("obs_space_low", step=1, value=-10)
            obs_space_high = st.number_input("obs_space_high", step=1, value=10)
            assert obs_space_high > obs_space_low
        with st.expander("Set Initial Conditions"):
            x_0 = st.number_input("x_0", step=1, value=0)
            y_0 = st.number_input("y_0", step=1, value=0)
            dt = st.number_input("dt", step=0.1, value=0.1)
            t_0 = st.number_input("t_0", step=1, value=0)
            t_end = st.number_input("t_end", step=10, value=20)
            y_ref = st.number_input("y_ref", value=1)
            steady_state_indicator = st.number_input(
                "steady_state_counter", step=5, value=30
            )
        env_config = {
            "action_space": [action_space_low, action_space_high],
            "obs_space": [obs_space_low, obs_space_high],
            "num": convert_np_array(numerator),
            "den": convert_np_array(denum),
            "x_0": [x_0],
            "dt": dt,
            "y_0": y_0,
            "t_0": t_0,
            "t_end": t_end,
            "y_ref": y_ref,
            "steady_state_indicator": steady_state_indicator,
        }
    return env_config


def create_tab_trainer(tab_training, env_config, algorithm_selected):
    train_config = None
    with tab_training:
        max_episode = st.number_input("Max Episodes", step=100, value=10)
        plotting_freq = st.number_input("Frequency of Plotting", value=1, step=1)
        printint_freq = st.number_input("Frequency of Console Logging", value=1, step=1)
        enable_log_tensorboard = st.checkbox("enable_log_tensorboard", value=False)
        save_checkpoints = st.checkbox("save_checkpoints", value=True)
        train_config = Trainer.get_default_training_config()
        train_config["enable_log_tensorboard"] = enable_log_tensorboard
        train_config["max_episode"] = max_episode
        train_config["algorithm_name"] = algorithm_selected
        train_config["max_step"] = int(env_config["t_end"] / env_config["dt"])
        train_config["plotting"]["enable"] = False  # plotting_enable
        train_config["plotting"]["freq"] = plotting_freq
        train_config["freq_print_console"] = printint_freq
        train_config["checkpoints"]["enable"] = save_checkpoints
        train_config["checkpoints"]["freq"] = 1
        train_config["plot_library"] = "streamlit"
    return train_config
