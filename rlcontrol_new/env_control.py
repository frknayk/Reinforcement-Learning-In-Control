from copy import deepcopy
from dataclasses import dataclass
from typing import List

import control as ct
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import Env
from gym.spaces import Box


@dataclass
class ConfigSISO:
    action_space: List[float]  # lower/upper limits
    obs_space: List[float]  # lower/upper limits
    num: List[int]
    den: List[int]
    x_0: List[float]
    dt: float
    y_0: float
    t_0: float
    t_end: float
    y_ref: float


def check_controllability(tf):
    """Check if system is being contrallable by controllability matrix's rank

    Parameters
    ----------
    tf : _type_
        _description_
    """
    controllability_matrix = ct.ctrb(tf.A, tf.B)
    rank = np.linalg.matrix_rank(controllability_matrix)
    if rank == controllability_matrix.shape[0]:
        return True
    return False


def create_pid(kp, ki, kd=0):
    """Create PID controller

    Parameters
    ----------
    kp : float
        Proportional Term
    ki : float
        Integral Term
    kd : float
        Derivative Term

    Returns
    -------
    _type_
        Transfer Function of PID controller in s-domain
    """
    s = ct.TransferFunction.s
    d_term = kd
    if kd != 0:
        d_term = kd * (s + 0.000001) / s
    return kp + ki * (1 / s) + d_term


class LinearSISOEnv(Env):
    """Linear Time-Invariant Single Input-Output Dynamic System"""

    def __init__(self, env_config: ConfigSISO):
        super(LinearSISOEnv, self).__init__()
        self.env_config = env_config
        self.action_space = Box(
            low=env_config.action_space[0],
            high=env_config.action_space[1],
            dtype=np.float32,
        )  # action space limits
        self.observation_space = Box(
            low=env_config.obs_space[0], high=env_config.obs_space[1], dtype=np.float32
        )  # observation space limits
        # self.sys = ct.tf2ss(env_config.num, env_config.den)  # Transfer function
        sys_tf = ct.tf(env_config.num, env_config.den)
        self.sys = ct.LinearIOSystem(sys_tf, inputs="u", outputs="y")
        if not check_controllability(self.sys):
            raise Exception("System is not controllable!, Try another system.")
        self.collected_reward = -1  # Total collected reward [scalar]
        self.t_all = np.arange(
            env_config.t_0, env_config.t_end + env_config.dt, env_config.dt
        )
        self.tick_sim = 0  # Simulation time index
        self.x = np.zeros(self.sys.nstates)  # system state
        self.y = None
        self.sim_results = []

    def reset(self, start_random: bool = False):
        self.x = np.zeros(self.sys.nstates)
        self.y = self.env_config.y_0
        obs = [self.x, self.y]
        info = {}
        self.sim_results = []
        self.tick_sim = 0  # Simulation time index
        return obs

    def __check_steady_state(self):
        """If system response is achieved to steady-state or not"""
        epsilon = 0.01
        diff = np.sum(np.absolute(self.y - self.env_config.y_ref))
        if (diff < epsilon) or self.y <= 0:
            return True
        return False

    @staticmethod
    def __calculate_reward(y_ref, y):
        diff = y_ref - y
        tuner = y_ref
        if diff < 0.01 * tuner:
            reward = 5
        if (diff < 0.05 * tuner) and (diff > 0.01 * tuner):
            reward = 0.5
        if (diff < 0.1 * tuner) and (diff > 0.05 * tuner):
            reward = 0.1
        else:
            reward = -diff
        reward = -np.abs(diff)
        return reward

    def step(self, action: float):
        """Apply control signal to system

        Parameters
        ----------
        action : float
            Control signal coming from outside loop

        Returns
        -------
        _type_
            _description_
        """
        done = False
        obs = None
        info = {}
        ###### One step simulation ######
        T_sim = [self.t_all[self.tick_sim], self.t_all[self.tick_sim + 1]]
        self.tick_sim += 1
        T_sim, Y_sim, X_sim = ct.input_output_response(
            self.sys, T_sim, U=action, X0=self.x, return_x=True
        )
        # Z^-1: unit-delay
        self.x = X_sim[:, 1]
        self.y = Y_sim[1]
        done = self.__check_steady_state()
        reward = self.__calculate_reward(self.env_config.y_ref, self.y)
        info = {"time_current": T_sim}
        obs = [X_sim, Y_sim]
        self.sim_results.append([T_sim[-1], action, Y_sim[-1]])
        return obs, reward, done, info

    def open_loop_step_response(self, action: float = 1.0):
        num_sim = int(
            (self.env_config.t_end - self.env_config.t_0) / self.env_config.dt
        )
        for _ in range(num_sim):
            self.step(action)

    def closed_loop_step_response(self, action: float = 1.0):
        self.sys = ct.feedback(self.sys, 1)
        num_sim = int(
            (self.env_config.t_end - self.env_config.t_0) / self.env_config.dt
        )
        for _ in range(num_sim):
            self.step(action)

    def render(self):
        if self.sim_results.__len__() == 0:
            print("No simulation is run, please call run simulation first")
            return
        sim_results = np.array(self.sim_results)
        t = sim_results[:, 0]
        u = sim_results[:, 1]
        y = sim_results[:, 2]
        # Plot inputs and outputs
        plt.title("System Response")
        plt.subplot(2, 1, 1)
        plt.plot(t, y, "o-")
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.step(t, u, where="post")
        plt.xlabel("t")
        plt.ylabel("u(t)")
        plt.grid()
        plt.show()


def example_open_loop():
    """Apply control signal (1) 100 times to system and render"""
    # Instantiate the env
    config_siso = ConfigSISO(
        action_space=[-1, 1],
        obs_space=[-10, 10],
        num=[1],
        den=[1, 1, 1],
        x_0=[0],
        dt=0.1,
        y_0=0,
        t_0=0,
        t_end=10,
        y_ref=5,
    )
    env = LinearSISOEnv(config_siso)
    env.open_loop_step_response(1)
    env.render()


def example_pid_control(kp, ki, kd):
    # Masss-Spring-Damper system
    config_siso = ConfigSISO(
        action_space=[-1, 1],
        obs_space=[-10, 10],
        num=[1],
        den=[1, 10, 20],
        x_0=[0],
        dt=0.01,
        y_0=0,
        t_0=0,
        t_end=5,
        y_ref=5,
    )
    env = LinearSISOEnv(config_siso)
    env.reset()
    c_p = create_pid(kp, 0, kd)
    env.sys = ct.series(env.sys, c_p)
    env.closed_loop_step_response()
    env.render()


if __name__ == "__main__":
    # example_open_loop()
    example_pid_control(300, 5)
