
from numpy.lib.function_base import select
from gym_control.base import ENV

class LinearFirstOrder(ENV):
    def __init__(self, initial_conditions, Ts, y_set, epsilon=0.98):
        super().__init__(initial_conditions)
        self.tau = pass
    
    