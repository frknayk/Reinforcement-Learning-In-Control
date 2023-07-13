import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# fig_plotly_pyplot, axs_plotly_pyplot = plt.subplots(3, sharex=True)
# fig_plotly_trace = make_subplots(rows=3, cols=1,
#             shared_xaxes=True, x_title="time[s]",
#             subplot_titles=("Reference vs Output","Reward", "Control Signal"))
# fig_plt, axs_plt = plt.subplots(3, sharex=True)


def plot_streamlit(episode_result_dict, eps: int, method="pyplot"):
    # df = pd.DataFrame(
    #     {"ref":episode_result_dict["reference_list"],
    #     "out":episode_result_dict["output_list"],
    #     "time":episode_result_dict["sim_time"],
    #     ""
    #     })
    if method == "pyplot":
        fig, axs = plt.subplots(3, sharex=True)
        axs[0].set_title(f"Output vs Reference (Episode:{eps})")
        axs[0].plot(episode_result_dict["sim_time"], episode_result_dict["output_list"])
        axs[0].plot(
            episode_result_dict["sim_time"], episode_result_dict["reference_list"]
        )
        axs[1].set_title("Rewards")
        axs[1].plot(episode_result_dict["sim_time"], episode_result_dict["reward_list"])
        axs[2].set_title("Control Signals(System Inputs)")
        axs[2].plot(
            episode_result_dict["sim_time"], episode_result_dict["control_sig_list"]
        )
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        return

    if method == "trace":
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            x_title="time[s]",
            subplot_titles=("Reference vs Output", "Reward", "Control Signal"),
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
                x=episode_result_dict["sim_time"], y=episode_result_dict["output_list"]
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=episode_result_dict["sim_time"], y=episode_result_dict["reward_list"]
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
        st.plotly_chart(fig, use_container_width=True)
        return


def plot_matplotlib(episode_result_dict: dict, eps: int):
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].set_title(f"Output vs Reference (Episode:{eps})")
    axs[0].plot(episode_result_dict["sim_time"], episode_result_dict["output_list"])
    axs[0].plot(episode_result_dict["sim_time"], episode_result_dict["reference_list"])
    axs[1].set_title("Rewards")
    axs[1].plot(episode_result_dict["sim_time"], episode_result_dict["reward_list"])
    axs[2].set_title("Control Signals(System Inputs)")
    axs[2].plot(
        episode_result_dict["sim_time"], episode_result_dict["control_sig_list"]
    )
    fig.show()
