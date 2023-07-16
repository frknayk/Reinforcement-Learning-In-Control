import io
import json
import pickle
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


def test_agent():
    with st.spinner("Analyzing the best policy"):
        if st.button("Analyze and Test Best Policy", key="button_analyze"):
            # episode_result_dict = trainer.test_agent()
            pass
        st.success("Test of best agent is completed!")


def load_experiment_config():
    experiment_config = None
    experiment_config_file = st.file_uploader("Load Experiment Config")
    if experiment_config_file is not None:
        experiment_config = pickle.loads(experiment_config_file.read())
    return experiment_config


def load_agent_ckpt():
    uploaded_file = None
    uploaded_file = st.file_uploader("Load Checkpoint")
    return uploaded_file


def page_testing():
    st.title("🕶️🕶️🕶️ Test Agent 🕶️🕶️🕶️")
    st.sidebar.markdown("Run Inference On Trained Agent🎉")
    st.sidebar.text("Select environment and load agent")
    st.title("Configure Agent,System and Training Parameters")
    experiment_config = load_experiment_config()
    if experiment_config is None:
        return
    env_config = dict(experiment_config["env_config"])
    trainer_config = dict(experiment_config["train_config"])
    agent_config = dict(experiment_config["agent_config"])
    trainer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )
    trainer.set_trainer_config(trainer_config)
    checkpoint = load_agent_ckpt()
    if checkpoint == "":
        return
    buffer = io.BytesIO(checkpoint.read())
    checkpoint = torch.load(buffer)
    # print(checkpoint)
    trainer.load_checkpoint_binary(checkpoint)
    # episode_result_dict = trainer.test_agent()
    # plot_test_results(episode_result_dict,10)
