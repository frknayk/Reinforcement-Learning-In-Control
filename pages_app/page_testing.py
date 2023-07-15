import numpy as np
import streamlit as st
from tqdm import tqdm

# import gym_control
from gym_control.envs import LinearSISOEnv
from pages_app.common import create_tab_agent, create_tab_env, create_tab_tf
from pages_app.plot_functions import plot_test_results
from rlc.agents.ddpg import DDPG
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer

logger = create_console_logger("logger_training")


def test_agent():
    with st.spinner("Analyzing the best policy"):
        if st.button("Analyze and Test Best Policy", key="button_analyze"):
            # episode_result_dict = trainer.test_agent()
            print("asdasd")
        st.success("Test of best agent is completed!")


def load_experiment_config():
    experiment_config = st.file_uploader("Load Experiment Config")
    return experiment_config


def load_agent():
    checkpoint_dir = ""
    uploaded_file = st.file_uploader("Choose a CSV file")
    print(uploaded_file)
    return checkpoint_dir


def page_testing():
    st.title("🕶️🕶️🕶️ Test Agent 🕶️🕶️🕶️")
    st.sidebar.markdown("Run Inference On Trained Agent🎉")
    st.sidebar.text("Select environment and load agent")
    st.title("Configure Agent,System and Training Parameters")
    tab_tf, tab_env = st.tabs(["Transfer Function", "Agent", "Environment"])
    numerator, denum = create_tab_tf(tab_tf)
    assert numerator is not None
    assert denum is not None
    experiment_config = load_experiment_config()
    print(experiment_config)
    assert experiment_config is not None
    env_config = create_tab_env(tab_env, numerator, denum)
    # assert env_config is not None
    # trainer_config = Trainer.get_default_training_config()
    # trainer = Trainer(
    #     env=LinearSISOEnv,
    #     agent_class=DDPG,
    #     agent_config=agent_config,
    #     env_config=env_config,
    # )
    # assert trainer_config is not None
    # trainer.set_trainer_config(trainer_config)
    # checkpoint_dir = load_agent()
    # if checkpoint_dir == "":
    #     return
    # # trainer.load_checkpoint(checkpoint_dir)
    # # episode_result_dict = trainer.test_agent()
    # # plot_test_results(episode_result_dict,eps)
