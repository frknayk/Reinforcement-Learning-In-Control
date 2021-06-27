import numpy as np
import copy
import math
import random
from rlcontrol.systems.base_systems.base_env import ENV
from rlcontrol.systems.base_systems.base_dynamics_linear import DynamicsLinear

class CruiseSystem(ENV):
    """Vehicle longitudinal dynamics for cruise control
    --------------------------------
    src : https://ctms.engin.umich.edu/CTMS/index.php?example=CruiseControl&section=ControlStateSpace
    --------------------------------
    v' = [-b/m]*v + [1/m]*u

    where,
        m : vehicle mass, 1000 kg
        b : damping coefficient, 50 N.s/m
        u : control force, N
        v : longitudinal velocity
    --------------------------------
    Args:
        ENV (class): Base environment class
    """
    def __init__(self, config=None):
        self.dimensions = {'state':1,'action':1}
        self.config = self.get_config_default()
        self.__set_point = self.config['set_point']
        self.__state = None
        self.__done = False
        self.rew = 0        
        self.model = DynamicsLinear(self.config['dynamics_config'])

    def set_config(self, config):
        self.config = self.get_config_default() if config is None else config
    
    def get_info(self):
        """Get state,reward and elapsed time like info as dict to outside"""
        info = {
            'state': self.__state,
            'state_ref': self.__set_point,
            'done' : self.__done,
            'reward' : self.rew,
            'sim_time': self.model.sim_time}
        return info

    @staticmethod
    def get_config_default():
        # System parameters
        b = 50.0 # damping_coeff
        m = 1000.0 # mass
        dynamics_config = {
            'system_matrix':np.array([ -b/m ]),
            'input_matrix' : np.array([1/m]),
            'output_matrix' : [1],
            'x0': np.array([0]),
            'sampling_time' : 0.1
        }
        default_config = {
            'dynamics_config' : dynamics_config,
            'set_point' : np.array([33]), #m/s
            'steady_state_tresh' : 0.5
        }
        return default_config

    def step(self, u:np.ndarray):
        nominal_control_force = 500
        u = u*nominal_control_force
        self.__state = self.model.forward(u)
        self.reward()
        is_done = self.is_done()
        return self.__state.copy(), self.rew, is_done 

    def reward(self):
        """Reward is absolute distance from speed to reference speed"""
        diff = np.sum( np.abs(self.__state - self.__set_point) )/self.__set_point
        self.rew = diff[0]

    def reset(self, x0:np.ndarray=None):
        self.rew = 0
        random_state = x0.copy() if x0 is not None else np.random.rand(1)*5
        self.model.x = random_state.copy()
        self.__state = random_state.copy()
        self.__set_point = 30 + np.random.rand(1)*10
        return self.__state

    def is_done(self):
        if np.abs(self.__set_point - self.__state) < 1e-3:
            self.model.steady_state_count += 1
            if self.model.steady_state_count*self.ts >= self.config['steady_state_tresh']:
                return True
            return False
        return False

def example_poc(num_eps=3):
    import matplotlib.pyplot as plt
    long_model = CruiseSystem()
    nominal_control_force = 500
    u = np.array([1])*nominal_control_force

    for eps in range(num_eps):

        fig, axs = plt.subplots(3)
        output_list = []
        reward_list=[]
        reference_list=[]
        control_sig_list=[]

        reward = 0
        state = np.random.rand(1)*10
        print("Simulation is started with {0} m/s velocity".format(state[0]))
        for x in range(1000):
            state, reward, done = long_model.step(u)

            sim_info = long_model.get_info()

            output_list.append(sim_info['state'])
            reference_list.append(sim_info['state_ref'])
            reward_list.append(sim_info['reward'])
            control_sig_list.append(u[0])

            if done:
                print("EPS: {0} is DONE!".format(eps))
                random_state = np.random.rand(1)*10
                long_model.reset(random_state)
                break

        model_info = long_model.get_info()

        print("Vehicle speed : {0} km/h".format(model_info['state'][0]*3.6))
        print("Total accumulated reward : {0}".format(model_info['reward']))
        print("======================")

        axs[0].set_title("Output vs Reference")
        axs[0].plot(output_list)
        axs[0].plot(reference_list)
        axs[1].set_title("Reward List")
        axs[1].plot(reward_list)
        axs[2].set_title("Control Signal List")
        axs[2].plot(control_sig_list)
        plt.show()


if __name__ == '__main__':
    example_poc()