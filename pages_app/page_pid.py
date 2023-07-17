import control as ct
import numpy as np
import streamlit as st

from gym_control.envs import LinearSISOEnv
from pages_app.common import create_tab_env, create_tab_tf
from pages_app.plot_functions import plot_pid
from rlc.control.pid import PIDController


def simulate_system(env: LinearSISOEnv, pid: PIDController):
    env.reset()
    pid.reset()
    max_step = int(env.env_config["t_end"] / env.env_config["dt"])
    for _ in range(max_step):
        output = pid.compute(env.y)
        env.step(pid.compute(env.y))
    return np.array(env.sim_results)


def page_pid():
    st.title("Design PID Controller")
    st.sidebar.markdown(
        "Play with parameters to see how easy to control \
        the system with good-old PID controller"
    )
    tab_tf, tab_env = st.tabs(["Transfer Function", "Simulation Params"])
    numerator, denum = create_tab_tf(tab_tf)
    assert numerator is not None
    assert denum is not None
    env_config = create_tab_env(tab_env, numerator, denum)
    kp = st.slider("Proportional(P)", 0.0, 1000.0, value=10.0)
    ki = st.slider("Integral(I)", 0.0, 1000.0, value=1.0)
    kd = st.slider("Derivative(D)", 0.0, 1000.0, value=5.0)
    env = LinearSISOEnv(env_config)
    pid = PIDController(kp, ki, kd, setpoint=env.y_ref)
    env.reset()
    pid.reset()
    with st.spinner("Simulation in Progress.."):
        if st.button("Control", key="button_pid"):
            sim_results = simulate_system(env, pid)
            plot_pid(sim_results, env)
