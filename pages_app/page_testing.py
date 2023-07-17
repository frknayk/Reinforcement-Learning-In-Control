import io
import json
import pickle
import pprint
from io import StringIO

import streamlit as st
from tqdm import tqdm

from gym_control.envs import LinearSISOEnv
from pages_app.common import create_tab_agent, create_tab_env, create_tab_tf
from pages_app.plot_functions import plot_test_results
from rlc.agents.ddpg import DDPG
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer

logger = create_console_logger("logger_training")
import torch


def test_agent(trainer):
    with st.spinner("Analyzing the best policy"):
        if st.button("Analyze and Test Best Policy", key="button_analyze"):
            episode_result_dict = trainer.train_one_episode(test_mode=True)
            placeholder = st.empty()
            plot_test_results(episode_result_dict, placeholder, 10)
            st.success("Test of best agent is completed!")


def load_experiment_config():
    experiment_config = None
    experiment_config_file = st.file_uploader("Load Experiment Config")
    if experiment_config_file is not None:
        experiment_config = pickle.loads(experiment_config_file.read())
    return experiment_config


def load_agent_ckpt():
    uploaded_file = st.file_uploader("Load Checkpoint")
    if uploaded_file is not None:
        buffer = io.BytesIO(uploaded_file.read())
        return torch.load(buffer)


def page_testing():
    st.title("Test Agent 🕶️🕶️🕶️")
    st.sidebar.markdown("Run Inference On Trained Agent🎉")
    st.sidebar.text("Select environment and load agent")
    st.title("Configure Agent,System and Training Parameters")
    experiment_config = load_experiment_config()
    if experiment_config is None:
        return
    checkpoint = load_agent_ckpt()
    if checkpoint == "" or checkpoint is None:
        return
    env_config = dict(experiment_config["env_config"])
    trainer_config = dict(experiment_config["train_config"])
    agent_config = dict(experiment_config["agent_config"])
    pprint.pprint(env_config)
    pprint.pprint(trainer_config)
    pprint.pprint(agent_config)

    trainer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )
    trainer.set_trainer_config(trainer_config)
    trainer.load_checkpoint_binary(checkpoint)
    test_agent(trainer)
