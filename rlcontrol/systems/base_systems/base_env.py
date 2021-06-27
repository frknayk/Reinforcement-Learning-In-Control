"""
This file aims to model any transfer function 
given in state-space form
"""
# Lib
import numpy as np
import math 


class ENV(object):
    def __init__(self,env_config):
        """Base environment constructor"""
        self.dimensions = {'state':int,'action':int}
        self.__state = None
        self.__set_point = None
        self.__done = bool
        self.rew = 0
        self.config = {'dynamics_config':None,'set_point':None}
        self.model = None

    @staticmethod
    def get_config_default():
        """Get default environment config"""
        return {'dynamics_config':None,'set_point':None}

    def step(self,action):
        """Pass RL agent's decision to environment and get next states"""
        next_state = None
        reward = None
        done = None
        return next_state, reward, done

    def reward(self):
        """Calculate and return reward"""
        pass

    def reset(self, x0=None):
        """Reset RL states with given initial conditions"""
        pass

    def is_done(self):
        """Check if the episode is done or not"""
        pass