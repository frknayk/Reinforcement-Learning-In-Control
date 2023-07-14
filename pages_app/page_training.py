import numpy as np
import streamlit as st
from tqdm import tqdm

# import gym_control
from gym_control.envs import LinearSISOEnv
from pages_app.common import (
    create_tab_agent,
    create_tab_env,
    create_tab_tf,
    create_tab_trainer,
    select_algorithm,
)
from pages_app.plot_functions import plot_training_episode
from rlc.agents.ddpg import DDPG
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer

logger = create_console_logger("logger_training")


def train_agent(trainer: Trainer):
    placeholder = st.empty()
    episode_reward_list = []
    with st.spinner("Wait Until Training Finished"):
        if st.button("Train", key="button_train"):
            max_eps = trainer.config["max_episode"]
            st_pbar = st.progress(0)
            for eps in tqdm(range(max_eps), "Agent Learning Progress: "):
                percentage = int(100 * (eps + 1) / max_eps)
                st_pbar.progress(percentage, text=f"Training Progress: %{percentage}")
                episode_result_dict = trainer.train_one_episode()
                episode_reward_list.append(episode_result_dict["episode_reward"])
                trainer.log_train_iter(episode_result_dict, eps)
                plot_training_episode(episode_result_dict, placeholder, eps)
            st.success("Training is completed!")


def page_training():
    st.header("Control LTI Systems with Deep Reinforcement Learning")
    st.sidebar.markdown("# Training Page 🎈")
    st.sidebar.text("Configure environment,agent\n and training parameters.")
    algorithm_selected = select_algorithm()
    st.title("Configure Agent,System and Training Parameters")
    tab_tf, tab_agent, tab_env, tab_training = st.tabs(
        ["Transfer Function", "Agent", "Environment", "Training"]
    )
    numerator, denum = create_tab_tf(tab_tf)
    assert numerator is not None
    assert denum is not None
    agent_config = create_tab_agent(tab_agent)
    assert agent_config is not None
    env_config = create_tab_env(tab_env, numerator, denum)
    assert env_config is not None
    train_config = create_tab_trainer(tab_training, env_config, algorithm_selected)
    trainer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )
    assert train_config is not None
    trainer.set_trainer_config(train_config)
    train_agent(trainer)
    agent_path = trainer.logger.log_weight_dir
    st.info(
        f"Trained agent can be found at path \
        (if save_checkpoints is checked) : {agent_path}"
    )
