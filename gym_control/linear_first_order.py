from numpy.lib.function_base import select
from base import ENV
import numpy as np
import copy

class StateSpace(ENV):
    # x' = Ax + Bu
    # y = Cx + Du

    def __init__(self, initial_conditions, A, B, C, D, Ts, y_set, epsilon=0.98):
        super(StateSpace, self).__init__(initial_conditions)
        self.initial_conditions = initial_conditions
        self.dt = Ts
        self.set_point = y_set
        self.epsilon= epsilon

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.reset()

    def step(self,action):
        next_state = None
        done = False

        next_state = self.A * self.state + self.B * action
        y_ = (self.C * self.state + self.D * action).item(0)
        error = self.set_point - y_
        reward = -error

        # print error
        if abs(error) < abs((1.0 - self.epsilon) * y_):
            done = self.done = True

        self.state += copy.deepcopy(next_state) * self.dt
        return self.state.item(0), reward, done

    def calc_reward(self,step):
        pass

    def reset(self): 
        self.state = self.initial_conditions["x"]
        self.next_state = self.initial_conditions["xdot"]
        self.done = False
        self.reward = 0

    def is_done(self):
        pass

    @staticmethod
    def create_basic():
        A = np.matrix([
            [0.0, 1.0], 
            [-0.4, -0.5]
            ])

        B = np.matrix([
            [0.0],
            [0.3]
            ])
        C = np.matrix([
            [1.0, 0.0]
        ])
        D = 0.0

        dt = 0.01
        reference = 5.0
        epsilon = 0.98

        x = np.matrix([[0.0], [0.0]])
        xdot = np.matrix([[0.0], [0.0]])
        initial_conditions = {
            "x": x,
            "xdot": xdot
        }
        return StateSpace(initial_conditions, A, B, C, D, dt, reference, epsilon=epsilon)

if __name__ == "__main__":
    
    simtime = 0.0
    env = StateSpace.create_basic()
    while simtime < 10.0:
        env.step(7.09)
        simtime += env.dt