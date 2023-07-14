import streamlit as st

from pages_app.page_testing import page_testing
from pages_app.page_training import page_training

st.set_page_config(
    page_title="Control Dynamical Systems With Deep Reinforcement Learning",
    page_icon="🕹️",
    layout="wide",
)


page_names_to_funcs = {
    "Training": page_training,
    "Inference": page_testing,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
