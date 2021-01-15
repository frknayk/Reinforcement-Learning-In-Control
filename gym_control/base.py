"""
This file aims to model any transfer function 
given in state-space form
"""
# Lib
import numpy as np
import math 


class ENV(object):
    # x' = Ax + Bu
    # y = Cx + Du
    def __init__(self,env_config):
        self.state_dim = None
        self.action_dim = None
        self.state = None
        self.set_point = None
        self.done = False
        self.reward = 0

    @staticmethod
    def get_default_env_config():
        default_config = {}
        return default_config

    def step(self,action):
        next_state = None
        reward = None
        done = None
        return next_state, reward, done

    def calc_reward(self):
        pass

    def reset(self): 
        pass

    def is_done(self):
        pass