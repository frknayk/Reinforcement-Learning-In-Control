import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def plot_training_episode(episode_result_dict, placeholder, eps):
    with placeholder.container():
        fig_col1, empty_col, fig_col2 = st.columns(3)
        with fig_col1:
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
            st.markdown("                            ")
        with fig_col2:
            st.markdown("### Metrics of the Episode ###")
            episode_reward = episode_result_dict["episode_reward"]
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


def plot_test_results(episode_result_dict, placeholder):
    with placeholder.container():
        fig_col1, empty_col, fig_col2 = st.columns(3)
        with fig_col1:
            st.markdown("### INFERENCE PLOT")
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
            st.markdown("### Metrics of the TEST ###")
            episode_reward = episode_result_dict["episode_reward"]
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
