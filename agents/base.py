
from numpy.lib.function_base import select


class Agent(object):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 64
        self.algorithm_name = ""

    def apply(self, state):
        action = None
        return action
    
    def load(self, agent_weight_abs):
        """Load trained agent weights"""
        pass

    def save(self, agent_weight_abs):
        """Save trained agent weights"""
        pass

    def get_batch_size(self):
        return self.batch_size

    def update_agent(self):
        pass

    def update_memory(self, state,action,reward,next_state,done):
        pass
    
    def reset(self):
        pass