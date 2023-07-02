import random
from dataclasses import dataclass
from typing import List

import control as ct
import gym
import matplotlib.pyplot as plt
import numpy as np
from con_sys import check_controllability
from gym import Env
from gym.spaces import Box, Discrete


@dataclass
class ConfigSISO:
    action_space: List[float]
    obs_space: List[float]
    num: List[int]
    den: List[int]
    x_0: List[float]
    dt: float
    y_0: float
    t_0: float
    t_end: float
    y_ref: float


class LinearSISOEnv(Env):
    """Linear Time-Invariant Single Input-Output Dynamic System"""

    def __init__(self, env_config: ConfigSISO):
        self.env_config = env_config
        self.action_space = env_config.action_space  # action space limits
        self.observation_space = env_config.obs_space  # observation space limits
        self.sys = ct.tf2ss(env_config.num, env_config.den)  # Transfer function
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
        return obs

    def __check_steady_state(self):
        """If system response is achieved to steady-state or not"""
        epsilon = 0.01
        diff = np.sum(np.absolute(self.y - self.y_set))
        if (diff < epsilon) or self.y <= 0:
            return True
        return False

    def __calculate_reward(self, y_ref, y):
        diff = y_ref - y
        diff = np.abs(np.asscalar(diff[0]))
        tuner = np.asscalar(y_ref)
        if diff < 0.01 * tuner:
            reward = 5
        if (diff < 0.05 * tuner) and (diff > 0.01 * tuner):
            reward = 0.5
        if (diff < 0.1 * tuner) and (diff > 0.05 * tuner):
            reward = 0.1
        else:
            reward = -diff
        reward = -np.abs(diff)
        self.collected_reward += reward
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
        self.y = Y_sim[:, 1]
        done = self.__check_steady_state()
        reward = self.__calculate_reward(self.env_config.ref, self.y)
        info = {"time_current": T_sim}
        obs = [X_sim, Y_sim]
        return obs, reward, done, info

    def render(self):
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
