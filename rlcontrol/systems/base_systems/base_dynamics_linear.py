import numpy as np

class DynamicsLinear(object):
    """
        ================================
        Create linear dynamical systems.
            x' = A.x + B.u 
        --------------------------------
        where,
            A : System matrix,
            B : Input matrix,
            x : System state vector,
            u : System input vector
    """
    def __init__(self, dynamics_config):
        """Constructor for linear dynamical systems
        ==========================================
        Args:
            - dynamics_config(dynamics_config_dict): Dictionary of dynamical system configuration as 
            follows:
             - system_matrix (np.ndarray): System matrix (A)
             - input_matrix (np.ndarray): Input matrix (B)
             - output_matrix (np.ndarray): Output matrix (C)
             - sampling_time (float): Sampling time in seconds
             - x0 (np.ndarray) : Vector of initial conditions
        """
        self.A = dynamics_config['system_matrix']
        self.B = dynamics_config['input_matrix']
        self.C = dynamics_config['output_matrix']
        self.x0 = dynamics_config['x0']
        self.ts = dynamics_config['sampling_time']
        self.x = np.zeros((self.A.shape[0],1)) if dynamics_config['x0'] is None else dynamics_config['x0']
        self.sim_time = 0
        self.steady_state_count = 0

    def forward(self, input_vector:np.ndarray):
        # Return none if input dimension mismatch
        if input_vector.shape[0] != self.B.shape[0]:
            return None
        # Calculate next state : x' = Ax + Bu
        self.x = self.x + self.ts*(np.matmul(self.A,self.x) + np.matmul(self.B,input_vector))
        # Calculate sim elapsed time
        self.sim_time += self.ts 
        return self.x.copy()

    def reset(self, initial_conditions):
        if initial_conditions.shape != self.x.shape:
            return None
        self.x = initial_conditions