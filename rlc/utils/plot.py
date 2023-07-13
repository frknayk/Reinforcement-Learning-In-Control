import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def plot_episode_result(episode_result_dict: dict, plot_library: str = "matplotlib"):
    def plot_matplotlib(episode_result_dict: dict):
        fig, axs = plt.subplots(3, sharex=True)
        axs[0].set_title("Output vs Reference")
        axs[0].plot(episode_result_dict["system_output"])
        axs[0].plot(episode_result_dict["system_reference"])
        axs[1].set_title("Rewards")
        axs[1].plot(episode_result_dict["reward_list"])
        axs[2].set_title("Control Signals(System Inputs)")
        axs[2].plot(episode_result_dict["Control Signals"])
        plt.show()

    def plot_streamlit(episode_result_dict):
        print("st line  chart")

        # chart_data = pd.DataFrame(
        #     {"ref":episode_result_dict["system_reference"].reshape(-1),
        #     "out":episode_result_dict["system_output"].reshape(-1)})
        # chart_data = pd.DataFrame(
        #     {"ref":episode_result_dict["system_reference"],
        #     "out":episode_result_dict["system_output"]})
        # st.line_chart(chart_data,x="Time[s]",y="Response",clear_figure=True)

        fig, axs = plt.subplots(3, sharex=True)
        axs[0].set_title("Output vs Reference")
        axs[0].plot(episode_result_dict["system_output"])
        axs[0].plot(episode_result_dict["system_reference"])
        axs[1].set_title("Rewards")
        axs[1].plot(episode_result_dict["reward_list"])
        axs[2].set_title("Control Signals(System Inputs)")
        axs[2].plot(episode_result_dict["Control Signals"])
        st.pyplot(fig)

        if episode_result_dict["plot_library"] == "matplotlib":
            plot_matplotlib(episode_result_dict)
        if episode_result_dict["plot_library"] == "streamlit":
            plot_streamlit(episode_result_dict)
