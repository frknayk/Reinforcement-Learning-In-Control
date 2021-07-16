import numpy as np
from rlcontrol.systems.base_systems.base_env import ENV
from rlcontrol.systems.base_systems.base_dynamics_linear import DynamicsLinear

np.random.seed(59)

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
        self.config = config if config is not None else self.get_config_default()
        self.__set_point = self.config['set_point']
        self.__state = None
        self.__done = False
        self.rew = 0
        self.nominal_control_force = 2000 #Throttle to acceleration ratio
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
            'steady_state_tresh' : 0.5,
            'random_state_init' : False, #TODO: Specify range for randomness
            'saturation_limit': [-2,2],
            'init_state' : np.array([0]), # If state is not initated randomly use this as state
        }
        return default_config

    def step(self, u:np.ndarray):
        u = np.clip(u, self.config.get('saturation_limit')[0], self.config.get('saturation_limit')[1])
        u = u*self.nominal_control_force 
        new_state = self.model.forward(u)
        if np.isnan(self.__state): 
            return None, None,None, True
        self.__state = new_state
        self.reward()
        is_done = self.is_done()
        return self.__state.copy(), self.rew, is_done 

    def reward(self):
        """Reward is minus absolute distance from speed to reference speed"""
        # BUG: Current reward logic causes high rewards for early stoppage. 
        if self.__state[0]<0 or self.__state[0]>45:
            self.rew = -100
            return

        # Percentage difference in set point 
        diff = np.sum( np.abs(self.__state - self.__set_point) )/self.__set_point
        diff = diff[0]

        # scalar = 0.1
        # if      0  <=  diff*100 < 1: scalar = 0.01
        # elif    1  <=  diff*100 < 5: scalar = 0.10
        # elif    5  <= diff*100 < 15: scalar = 0.30
        # elif    15 <= diff*100 < 35: scalar = 0.45
        # elif    35 <= diff*100 < 55: scalar = 0.6
        # elif    55 <= diff*100 < 75: scalar = 0.8
        # elif    75 <= diff*100 < 85: scalar = 1.2
        # elif    85 <= diff*100:      scalar = 1.5
        scalar = 1 

        self.rew = diff*scalar*-1

    def reset(self, x0:np.ndarray=None):
        self.rew = 0

        new_state = self.config['init_state']
        if self.config['random_state_init'] is True:
            new_state = x0.copy() if x0 is not None else np.random.rand(1)*5
        
        self.__state = new_state.copy()

        # Reset dynamic system
        self.model.reset(new_state.copy())

        # self.__set_point = 30 + np.random.rand(1)*10 if self.config['random_state_init'] is True else \
        #     self.config['set_point']
        self.__set_point = self.config['set_point']

        return self.__state

    def is_done(self):
        # In general, settling time is considered as %98 percent of the set point
        if np.abs(self.__set_point*0.98 - self.__state) < 1e-3:
            self.model.steady_state_count += 1
            if self.model.steady_state_count*self.model.ts >= self.config['steady_state_tresh']:
                print("Achived Steady-STATE!")
                return True

        # TODO: Add these to config as state finish boundaries
        # TODO: Use same boundaries for both is_done() and reward() functions
        elif self.__state[0]<0 or self.__state[0]>40:
            return True

        return False

def example_cruise_dynamics(num_eps=3):
    import matplotlib.pyplot as plt
    long_model = CruiseSystem()
    u = np.array([1])
    for eps in range(num_eps):

        fig, axs = plt.subplots(3)
        output_list = []
        reward_list=[]
        reference_list=[]
        control_sig_list=[]

        state = long_model.reset()

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

        print("Vehicle speed : {0} km/h and Reference speed : {1} km/h".format(model_info['state'][0],
            model_info['state_ref'][0]))
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
    example_cruise_dynamics()