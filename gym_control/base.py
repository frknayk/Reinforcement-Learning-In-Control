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

    def __init__(self,initial_conditions):
        self.state = None
        self.next_state = None
        self.set_point = None
        self.done = False
        self.reward = 0

    def step(self,action):
        next_state = None
        reward = None
        done = None
        return next_state, reward, done

    def calc_reward(self,step):
        pass

    def reset(self): 
        pass

    def is_done(self):
        pass