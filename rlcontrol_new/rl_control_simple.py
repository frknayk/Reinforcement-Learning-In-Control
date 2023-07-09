import os
from typing import List

import control as ct
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import ray
from gymnasium.spaces import Box, Discrete
from ray.tune.registry import register_env

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LinearSISOEnv(gym.Env):
    """Linear Time-Invariant Single Input-Output Dynamic System"""

    def __init__(self, env_config: dict):
        super(LinearSISOEnv, self).__init__()
        self.env_config = env_config
        # action space limits
        self.action_space = Box(
            low=env_config["action_space"][0],
            high=env_config["action_space"][1],
            dtype=np.float32,
        )
        # observation space limits
        self.observation_space = Box(
            low=env_config["obs_space"][0],
            high=env_config["obs_space"][1],
            dtype=np.float32,
        )
        sys_tf = ct.tf(env_config["num"], env_config["den"])
        # self.sys = ct.LinearIOSystem(sys_tf, inputs="u", outputs="y", dt=env_config["dt"])
        self.sys = ct.LinearIOSystem(sys_tf, inputs="u", outputs="y")
        if not check_controllability(self.sys):
            raise Exception("System is not controllable!, Try another system.")
        self.collected_reward = -1  # Total collected reward [scalar]
        self.t_all = np.arange(
            env_config["t_0"], env_config["t_end"] + env_config["dt"], env_config["dt"]
        )
        self.tick_sim = 0  # Simulation time index
        self.x = np.zeros(self.sys.nstates)  # system state
        self.y = env_config["y_0"]
        self.sim_results = []

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.x = np.zeros(self.sys.nstates)
        self.y = self.env_config["y_0"]
        self.sim_results = []
        self.tick_sim = 0  # Simulation time index
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _get_obs(self):
        return np.array([self.y], dtype=np.float32)

    def _get_info(self):
        return {"reward_total": self.collected_reward}

    def __check_steady_state(self):
        """If system response is achieved to steady-state or not"""
        epsilon = 0.01
        diff = np.sum(np.absolute(self.y - self.env_config["y_ref"]))
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

    def step(self, action):
        """Apply control signal to system

        Parameters
        ----------
        action : numpy.ndarray
            Control signal coming from outside loop

        Returns
        -------
        _type_
            _description_
        """
        ###### One step simulation ######
        T_sim = [self.t_all[self.tick_sim], self.t_all[self.tick_sim + 1]]
        self.tick_sim += 1
        if type(action) == np.ndarray:
            action = action[0]
        T_sim, Y_sim, X_sim = ct.input_output_response(
            self.sys, T_sim, U=action, X0=self.x, return_x=True
        )
        # Z^-1: unit-delay
        self.x = X_sim[:, 1]
        self.y = Y_sim[1]
        done = self.__check_steady_state()
        reward = self.__calculate_reward(self.env_config["y_ref"], self.y)
        info = {"time_current": T_sim}
        obs = self._get_obs()
        info = self._get_info()
        self.sim_results.append([T_sim[-1], action, Y_sim[-1]])
        # observation, reward, terminated, False, info
        return obs, reward, done, False, info

    def open_loop_step_response(self, action: float = 1.0):
        num_sim = int(
            (self.env_config["t_end"] - self.env_config["t_0"]) / self.env_config["dt"]
        )
        for _ in range(num_sim):
            self.step(action)

    def closed_loop_step_response(self, action: float = 1.0):
        self.sys = ct.feedback(self.sys, 1)
        num_sim = int(
            (self.env_config["t_end"] - self.env_config["t_0"]) / self.env_config["dt"]
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

    @staticmethod
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

    @staticmethod
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


env_config = {
    "action_space": [-1, 1],
    "obs_space": [-10, 10],
    "num": [1],
    "den": [1, 10, 20],
    "x_0": [0],
    "dt": 0.01,
    "y_0": 0,
    "t_0": 0,
    "t_end": 5,
    "y_ref": 5,
}

## Check custom environment
# from ray.rllib.utils import check_env
# check_env(LinearSISOEnv(env_config))


def train_ppo_ex():
    from ray.rllib.algorithms import ppo

    ray.init()
    algo = ppo.PPO(env=LinearSISOEnv, config={"env_config": env_config})
    while True:
        print(algo.train())


from ray.rllib.algorithms import ddpg

ray.init()
algo = ddpg.DDPG(env=LinearSISOEnv, config={"env_config": env_config})

train_result = algo.train()
print(train_result)
