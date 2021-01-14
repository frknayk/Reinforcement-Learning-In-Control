from numpy.lib.function_base import select
from .base import ENV
import numpy as np
import copy
import math
import random


class ENV(object):
    # x' = Ax + Bu
    # y = Cx + Du
    def __init__(self, config=None):
        # States of RL agent
        self.state_dim = 4
        self.action_dim = 1
        self.done = False
        self.reward = 0

        # Time Constant
        self.tau = 0.01

        self.x = np.zeros((2, 1))
        self.xdot = np.zeros((2, 1))
        self.y = np.zeros((1, 1))
        self.y_set = np.zeros((1, 1))
        self.set_config(config)

        self.A = np.matrix([
            [0.0, 1.0],
            [-0.4, -0.5]
        ])

        self.B = np.matrix([
            [0.0],
            [0.3]
        ])
        self.C = np.matrix([
            [1.0, 0.0]
        ])
        self.D = 0.0

    
        self.epsilon = 0.01
        self.e_t0 = self.y_set - self.y
        self.e_t1 = 0
        self.e_dot = 0
        self.e_int = 0


    def set_config(self, config):
        if config is None:
            config = self.get_default_env_config()
        self.tau = config['tau']
        self.y_set = config['y_set']
        self.x = config['x_0'].copy()
        self.xdot = config['xdot_0'].copy()
        self.y = config['y_0'].copy()

    @staticmethod
    def get_default_env_config():
        default_config = {
            'tau': 0.1,
            'y_set': 0.5,
            # Initial cond. of X
            'x_0': np.array([[0.5], [0]]),
            # Initial cond. of x'
            'xdot_0': np.array([[0], [0]]),
            # Initial cond of output
            'y_0': np.array([[0]])
        }
        return default_config

    def step(self, u):
        u = u.transpose()

        xdot = self.A * self.x + self.B * u
        self.y = self.C * self.x + self.D * u


        x = self.x + xdot * self.tau
        
        # self.xdot = copy.deepcopy(xdot)
        # self.x = copy.deepcopy(x)
        self.x = x
        self.xdot = xdot

        self.calc_reward()
        self.if_done()
        self.error_calc()

        next_state = np.random.rand(1, 4)
        next_state[0, 0] = np.asscalar(self.y_set)
        next_state[0, 1:3] = self.x.reshape(1, 2)
        next_state[0, 3] = self.e_int
        return next_state, self.reward, self.done

    def error_calc(self):
        self.e_int += (self.y_set - self.y) * self.tau

    def calc_reward(self):
        diff = self.y_set - self.y
        diff = math.fabs(np.asscalar(diff[0]))
        tuner = np.asscalar(self.y_set)
        if(diff < 0.01*tuner):
            self.reward = 5
        if(diff < 0.05*tuner) and (diff > 0.01*tuner):
            self.reward = 0.5
        if(diff < 0.1*tuner) and (diff > 0.05*tuner):
            self.reward = 0.1
        else:
            self.reward = -diff
        self.reward = -math.fabs(diff)
        
    def reset(self):
        # Init system output randomly in every episode.
        self.y = np.array([[0]])
        self.x = np.array([[0.5], [0]])  # np.array([ [h1],[h2] ])
        self.e_int = 0.0
        self.y_set = np.array([[random.random()*0.4 + 0.1]])
        out = np.random.rand(1, 4)
        out[0, 0] = np.asscalar(self.y_set)
        out[0, 1:3] = self.x.reshape(1, 2)
        out[0, 3] = self.e_int
        return out

    def if_done(self):
        diff = np.sum(np.absolute(self.y-self.y_set))
        if(diff < self.epsilon*0.1):
            self.done = True
        elif self.y <= 0:
            self.done = True
        else:
            self.done = False
