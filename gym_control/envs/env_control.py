from typing import List

import control as ct
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.spaces import Box

ConfigSISO = {
    # lower/upper limits
    "action_space": List[float],
    # lower/upper limits
    "obs_space": List[float],
    "num": List[int],
    "den": List[int],
    "x_0": List[float],
    "dt": float,
    "y_0": float,
    "t_0": float,
    "t_end": float,
    "y_ref": float,
    # How many seconds will be accepted as achieved ss
    "steady_state_indicator": float,
}


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
        if not self.check_controllability(self.sys):
            raise Exception("System is not controllable!, Try another system.")
        self.collected_reward = -1  # Total collected reward [scalar]
        self.t_all = np.arange(
            env_config["t_0"], env_config["t_end"] + env_config["dt"], env_config["dt"]
        )
        self.tick_sim = 0  # Simulation time index
        self.x = np.zeros(self.sys.nstates)  # system state
        self.y = env_config["y_0"]
        self.y_ref = env_config["y_ref"]
        self.sim_results = []
        self.counter_ss = 0

    def _get_obs(self):
        return np.array([self.y], dtype=np.float32)

    def _get_info(self):
        return {"reward_total": self.collected_reward, "states": self.x}

    def __check_steady_state(self):
        """If system response is achieved to steady-state or not"""
        if self.y <= 0:
            return False
        if (self.y_ref - self.y) / self.y_ref < 0.02:
            self.counter_ss += 1
            if self.counter_ss >= self.env_config["steady_state_indicator"]:
                return True
            return False
        # if the difference is not lie around 2% of reference consecutively,
        # reset the counter
        self.counter_ss = 0
        return False

    @staticmethod
    def __calculate_reward(y_ref, y):
        # Assuming y_ref : positive
        e_t = y_ref - y  # e(t)
        e_t_normalized = e_t / y_ref
        if y < 0:
            return -e_t_normalized
        if y > y_ref:
            return e_t_normalized
        return -e_t_normalized

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

    def step(self, action):
        """Apply control signal to system

        Parameters
        ----------
        action : numpy.ndarray
            Control signal coming from outside loop

        Returns
        -------
        Tuple
            obs, reward, done, truncated, info
        """
        ###### One step simulation ######
        if self.tick_sim >= self.t_all.shape[0] - 1:
            print(f"tick:{self.tick_sim} ? t_all:{self.t_all.shape[0]}")
            obs, info = self.reset()
            done = True
            reward = -self.env_config["y_ref"]
            return obs, reward, done, False, info
        T_sim = [self.t_all[self.tick_sim], self.t_all[self.tick_sim + 1]]
        self.tick_sim += 1
        if type(action) == np.ndarray:
            action = action[0]
        # If action exceeds limits, saturate it.
        action = self._saturate(action)
        T_sim, Y_sim, X_sim = ct.input_output_response(
            self.sys, T_sim, U=action, X0=self.x, return_x=True
        )
        # Z^-1: unit-delay
        self.x = X_sim[:, 1]
        self.y = Y_sim[1]
        done = self.__check_steady_state()
        # if done:
        #     yref = self.env_config["y_ref"]
        #     # print(f"DONE TRIGGERED!: y_ref:{yref} - y:{self.y}")
        reward = self.__calculate_reward(self.env_config["y_ref"], self.y)
        info = {"time_current": T_sim}
        obs = self._get_obs()
        info = self._get_info()
        self.sim_results.append([T_sim[-1], action, Y_sim[-1]])
        return obs, reward, done, False, info

    def _saturate(self, action):
        if action < self.action_space.low[0]:
            return float(self.action_space.low[0])
        if action > self.action_space.high[0]:
            return float(self.action_space.high[0])
        return action

    def open_loop_step_response(self, action: float = 1.0):
        num_sim = int(
            (self.env_config["t_end"] - self.env_config["t_0"]) / self.env_config["dt"]
        )
        for _ in range(num_sim):
            self.step(action)

    def closed_loop_step_response(self, action: float = 1.0):
        # sys = C(s)*G(s) -> TF(s)= C(s)G(s)/1+C(s)G(s) where TF:closed-loop tf fnc.
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


def example_open_loop():
    """Apply control signal (1) 100 times to system and render"""
    # Instantiate the env
    config_siso = {
        "action_space": [-5, 5],
        "obs_space": [-10, 10],
        "num": [1],
        "den": [1, 10, 20],
        "x_0": [0],
        "dt": 1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 500,
        "y_ref": 1,
        # number of ticks
        "steady_state_indicator": 10,
    }
    env = LinearSISOEnv(config_siso)
    env.open_loop_step_response(1)
    env.render()


def example_pid_control(kp, ki, kd):
    # Masss-Spring-Damper system
    config_siso = {
        "action_space": [-5, 5],
        "obs_space": [-10, 10],
        "num": [1],
        "den": [1, 10, 20],
        "x_0": [0],
        "dt": 1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 500,
        "y_ref": 1,
        # number of ticks
        "steady_state_indicator": 10,
    }
    env = LinearSISOEnv(config_siso)
    env.reset()
    c_p = env.create_pid(kp, ki, kd)
    env.sys = ct.series(env.sys, c_p)
    env.closed_loop_step_response()
    env.render()


def example_pid_control_2(kp, ki, kd):
    # Masss-Spring-Damper system
    env_config = {
        "action_space": [-5, 5],
        "obs_space": [-10, 10],
        "num": [1],
        "den": [1, 10, 20],
        "x_0": [0],
        "dt": 1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 500,
        "y_ref": 1,
        # number of ticks
        "steady_state_indicator": 10,
    }
    env = LinearSISOEnv(env_config)
    sys_controller = env.create_pid(kp, ki, kd)
    # u(t) = C(e(t))
    num_sim = int((env_config["t_end"] - env_config["t_0"]) / env_config["dt"])
    obs, info = env.reset()
    for iter_idx in range(num_sim):
        e_t = env_config["y_ref"] - obs[0]

        pass

    env.render()


def example_custom_signal():
    # Masss-Spring-Damper system
    config_siso = {
        "action_space": [-5, 5],
        "obs_space": [-10, 10],
        "num": [1],
        "den": [1, 10, 20],
        "x_0": [0],
        "dt": 1,
        "y_0": 0,
        "t_0": 0,
        "t_end": 500,
        "y_ref": 1,
        # number of ticks
        "steady_state_indicator": 10,
    }
    env = LinearSISOEnv(config_siso)
    obs, info = env.reset()
    total_ticks = int(config_siso["t_end"] / config_siso["dt"])
    for x in range(total_ticks):
        # env.step(config_siso["action_space"][1])
        env.step(25)
    env.render()


if __name__ == "__main__":
    # example_pid_control_2(300, 10, 5)
    example_custom_signal()
