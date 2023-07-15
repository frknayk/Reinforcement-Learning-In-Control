import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from tqdm import tqdm

import gym_control
from gym_control.envs import LinearSISOEnv
from rlc.agents.ddpg import DDPG, PolicyNetwork, ValueNetwork
from rlc.logger.logger import create_console_logger
from rlc.rlcontrol import Trainer
from rlc.utils.plot import plot_streamlit

logger = create_console_logger("rlcontrolApp")


st.set_page_config(
    page_title="Control LTI Systems with Deep Reinforcement Learning",
    page_icon="✅",
    layout="wide",
)
# st.header("Control LTI Systems with Deep Reinforcement Learning")


def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")


def page_training():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")


def page_testing():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")


page_names_to_funcs = {
    "Main Page": main_page,
    "Train": page_training,
    "Inference": page_testing,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

# Algorithms
algorithm_selected = st.selectbox("Select the algorithm", ("DDPG", "PPO", "DQN"))
if algorithm_selected != "DDPG":
    logger.error("Only DDPG is available for now")
    st.error("Only DDPG is available for now")
st.title("Configure Agent,System and Training Parameters")
tab_tf, tab_agent, tab_env, tab_training = st.tabs(
    ["Transfer Function", "Agent", "Environment", "Training"]
)

with tab_tf:
    # st.text("Enter Transfer Function (in laplace domain)")
    def convert_np_array(text_list):
        return np.float32(json.loads(text_list))
        # return json.loads(text_list)

    numerator = st.text_input(
        "Numerator of transfer function (Enter as list of numbers: [1,2,3]) ",
        value="[1]",
    )
    denum = st.text_input(
        "Denumerator of transfer function (Enter as list of numbers: [1,2,3]) ",
        value="[1, 10, 20]",
    )

with tab_agent:
    # with st.expander("Agent Config"):
    # st.text(" ==== Agent Config ====")
    agent_config = {}
    batch_size = st.number_input("batch_size", value=128, min_value=0, max_value=2048)
    hidden_dim = st.number_input("hidden_dim", value=64, min_value=16, max_value=1024)
    agent_config = {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "policy_net": PolicyNetwork,
        "value_net": ValueNetwork,
    }

with tab_env:
    # st.text(" ==== Environment Config ====")
    with st.expander("Action/Observation Space Bounds"):
        action_space_low = st.number_input("action_space_low", step=1, value=-1)
        action_space_high = st.number_input("action_space_high", step=1, value=50)
        assert action_space_high > action_space_low
        obs_space_low = st.number_input("obs_space_low", step=1, value=-10)
        obs_space_high = st.number_input("obs_space_high", step=1, value=10)
        assert obs_space_high > obs_space_low
    with st.expander("Initial Conditions:"):
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

with tab_training:
    trainer = Trainer(
        env=LinearSISOEnv,
        agent_class=DDPG,
        agent_config=agent_config,
        env_config=env_config,
    )
    # st.text(" ==== Training Config ====")
    max_episode = st.number_input("Max Episodes", step=100, value=10)
    plotting_freq = st.number_input("Frequency of Plotting", value=1, step=1)
    printint_freq = st.number_input("Frequency of Console Logging", value=1, step=1)
    enable_log_tensorboard = st.checkbox("enable_log_tensorboard", value=True)
    # plotting_enable = st.checkbox("plotting_enable", value=True)
    save_checkpoints = st.checkbox("save_checkpoints", value=False)

    train_config = trainer.get_default_training_config()
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


placeholder = st.empty()
episode_reward_list = []
is_training_completed = False
with st.spinner("Wait for it..."):
    if st.button("Train", key="button_train"):
        trainer.set_training_config(train_config)
        for eps in tqdm(
            range(trainer.config["max_episode"]), "Agent Learning Progress: "
        ):
            episode_result_dict = trainer.train_one_episode()
            trainer.log_train_iter(episode_result_dict, eps)
            with placeholder.container():
                fig_col1, empty_col, fig_col2 = st.columns(3)
                with fig_col1:
                    st.markdown("### First Chart")
                    fig = make_subplots(
                        rows=3,
                        cols=1,
                        shared_xaxes=True,
                        x_title="time[s]",
                        subplot_titles=(
                            "Reference vs Output",
                            "Reward",
                            "Control Signal",
                        ),
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=episode_result_dict["sim_time"],
                            y=episode_result_dict["reference_list"],
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=episode_result_dict["sim_time"],
                            y=episode_result_dict["output_list"],
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=episode_result_dict["sim_time"],
                            y=episode_result_dict["reward_list"],
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=episode_result_dict["sim_time"],
                            y=episode_result_dict["control_sig_list"],
                        ),
                        row=3,
                        col=1,
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=600,
                        width=800,
                        title_text=f"Output vs Reference (Episode:{eps})",
                    )
                    st.write(fig)
                with empty_col:
                    st.markdown("============")
                with fig_col2:
                    st.markdown("### Metrics of the Episode ###")
                    episode_reward = episode_result_dict["episode_reward"]
                    episode_reward_list.append(episode_reward)
                    step_total = episode_result_dict["step_total"]
                    episode_policy_loss = episode_result_dict["episode_policy_loss"]
                    episode_value_loss = episode_result_dict["episode_value_loss"]
                    total_control_signal = episode_result_dict["total_control_signal"]
                    total_output_signal = episode_result_dict["total_output_signal"]
                    st.metric(label="Reward(Total)", value=f"{episode_reward}")
                    st.metric(label="Episode Length", value=f"{step_total}")
                    st.metric(label="Policy Loss", value=f"{episode_policy_loss}")
                    st.metric(label="Value Loss", value=f"{episode_value_loss}")
                    st.metric(
                        label="Integral of Control Signal",
                        value=f"{total_control_signal}",
                    )
                    st.metric(
                        label="Integral of Output Signal",
                        value=f"{total_output_signal}",
                    )
        is_training_completed = True
        st.success("Training is completed!")

# if is_training_completed:
#     with st.spinner("Analyzing the best policy"):
#         if st.button("Analyze and Test Best Policy", key="button_analyze"):
#             episode_result_dict = trainer.test_agent()
#             with placeholder.container():
#                 fig_col1, empty_col, fig_col2 = st.columns(3)
#                 with fig_col1:
#                     st.markdown("### INFERENCE PLOT")
#                     fig = make_subplots(
#                         rows=3,
#                         cols=1,
#                         shared_xaxes=True,
#                         x_title="time[s]",
#                         subplot_titles=(
#                             "Reference vs Output",
#                             "Reward",
#                             "Control Signal",
#                         ),
#                     )
#                     fig.add_trace(
#                         go.Scatter(
#                             x=episode_result_dict["sim_time"],
#                             y=episode_result_dict["reference_list"],
#                         ),
#                         row=1,
#                         col=1,
#                     )
#                     fig.add_trace(
#                         go.Scatter(
#                             x=episode_result_dict["sim_time"],
#                             y=episode_result_dict["output_list"],
#                         ),
#                         row=1,
#                         col=1,
#                     )
#                     fig.add_trace(
#                         go.Scatter(
#                             x=episode_result_dict["sim_time"],
#                             y=episode_result_dict["reward_list"],
#                         ),
#                         row=2,
#                         col=1,
#                     )
#                     fig.add_trace(
#                         go.Scatter(
#                             x=episode_result_dict["sim_time"],
#                             y=episode_result_dict["control_sig_list"],
#                         ),
#                         row=3,
#                         col=1,
#                     )
#                     fig.update_layout(
#                         showlegend=False,
#                         height=600,
#                         width=800,
#                         title_text=f"Output vs Reference (Episode:{eps})",
#                     )
#                     st.write(fig)
#                 with empty_col:
#                     st.markdown("============")
#                 with fig_col2:
#                     st.markdown("### Metrics of the TEST ###")
#                     episode_reward = episode_result_dict["episode_reward"]
#                     episode_reward_list.append(episode_reward)
#                     step_total = episode_result_dict["step_total"]
#                     episode_policy_loss = episode_result_dict["episode_policy_loss"]
#                     episode_value_loss = episode_result_dict["episode_value_loss"]
#                     total_control_signal = episode_result_dict["total_control_signal"]
#                     total_output_signal = episode_result_dict["total_output_signal"]
#                     st.metric(label="Reward(Total)", value=f"{episode_reward}")
#                     st.metric(label="Episode Length", value=f"{step_total}")
#                     st.metric(label="Policy Loss", value=f"{episode_policy_loss}")
#                     st.metric(label="Value Loss", value=f"{episode_value_loss}")
#                     st.metric(
#                         label="Integral of Control Signal",
#                         value=f"{total_control_signal}",
#                     )
#                     st.metric(
#                         label="Integral of Output Signal",
#                         value=f"{total_output_signal}",
#                     )
#         st.success("Test of best agent is completed!")
