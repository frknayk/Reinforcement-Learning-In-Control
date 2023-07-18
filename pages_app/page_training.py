from os.path import basename
from zipfile import ZipFile

import streamlit as st
from tqdm import tqdm

from gym_control.envs import LinearSISOEnv
from pages_app.common import (
    create_tab_agent,
    create_tab_algorithm,
    create_tab_env,
    create_tab_tf,
    create_tab_trainer,
)
from pages_app.plot_functions import plot_training_episode
from rlc.agents.ddpg import DDPG
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer

logger = create_console_logger("logger_training")


def train_agent(trainer: Trainer, trainer_config):
    placeholder = st.empty()
    episode_reward_list = []
    with st.spinner("Wait Until Training Finished"):
        if st.button("Train", key="button_train"):
            trainer.set_trainer_config(trainer_config)
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
            return True
    return False


def download_experiment(_trainer, is_train_completed):
    if not is_train_completed:
        return
    ckpt_best_dir = _trainer.logger.log_weight_dir + "/agent_best.pth"
    experiment_config_dir = _trainer.logger.get_experiment_pickle_dir()
    zip_path = _trainer.logger.get_logger_relative_path("rlcontrol_result.zip")
    with st.spinner("Results are zipping..."):
        with ZipFile(zip_path, "w") as zip:
            zip.write(ckpt_best_dir, basename(ckpt_best_dir))
            zip.write(experiment_config_dir, basename(experiment_config_dir))
        st.success("Training files zipped succesfully 🔥")
        with open(zip_path, "rb") as f:
            st.download_button(
                "Download Result as Zip", f, file_name="rlcontrol_result.zip"
            )


def page_training():
    st.title("Control LTI Systems with Deep Reinforcement Learning")
    st.header("Configure Agent,System and Training Parameters")
    st.sidebar.markdown("# Training Page 🎈")
    st.sidebar.text("Configure environment,agent\n and training parameters.")
    tab_algorithm, tab_tf, tab_agent, tab_env, tab_training = st.tabs(
        ["Algorithm", "Transfer Function", "Agent", "Environment", "Training"]
    )
    algorithm_selected = create_tab_algorithm(tab_algorithm)
    numerator, denum = create_tab_tf(tab_tf)
    assert numerator is not None
    assert denum is not None
    agent_config = create_tab_agent(tab_agent, DDPG.get_default_params())
    assert agent_config is not None
    env_config = create_tab_env(tab_env, numerator, denum)
    assert env_config is not None
    trainer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )
    trainer_config = create_tab_trainer(tab_training, env_config, algorithm_selected)
    assert trainer_config is not None
    is_train_completed = False
    try:
        is_train_completed = train_agent(trainer, trainer_config)
    except Exception as e:
        print(e)
    agent_path = trainer.logger.log_weight_dir
    st.info(
        f"Trained agent can be found at path \
        (use it when inference) : {agent_path}"
    )
    download_experiment(trainer, is_train_completed)
