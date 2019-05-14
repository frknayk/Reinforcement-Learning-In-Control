"""
This file aims to model any transfer function 
given in state-space form
"""
# Lib
import numpy as np
import math 
import random 


class ENV(object):
    # x' = Ax + Bu
    # y = Cx + Du

    def __init__(self,tau,Y_set):
        self.tau = tau
        self.x = np.array([ [0.5],[0] ])
        self.xdot = np.array([ [0],[0] ])
        self.y = np.array([ [0] ])
        self.y_set = np.array([ [Y_set] ])
        self.epsilon = 0.01
        self.done = False
        self.r = 0
        self.e_t0 = self.y_set - self.y
        self.e_t1 = 0
        self.e_dot = 0
        self.e_int = 0
        self.c1 = 1.035454*math.pow(10,-2)
        self.c2 = 1.143*math.pow(10,-3)
        self.c3 = 1.035454*math.pow(10,-2)

    def step(self,u,step):
        u = u.transpose()
        h1 = np.asscalar(self.x[0])
        h2 = np.asscalar(self.x[1])
        dh = h1 - h2
        h1_dot = ( ( -self.c1 )*np.sign(dh)* math.sqrt(math.fabs(dh)) ) + ( u*self.c2 )
        h2_dot = ( ( self.c1  )*np.sign(dh)*  math.sqrt(math.fabs(dh)) ) - ( self.c3*math.sqrt( h2 ) )
        # h2_dot = ( ( self.c1  )*np.sign(dh)*  math.sqrt(math.fabs(dh)) ) - ( self.c3*math.sqrt( math.fabs(h2) ) )
        h1 = h1 + h1_dot*self.tau 
        h2 = h2 + h2_dot*self.tau 
        self.xdot[0] = h1_dot
        self.xdot[1] = h2_dot
        self.x[0] = h1
        self.x[1] = h2
        self.y = self.x[1]
        self.reward(step)
        self.if_done()
        self.error_calc()
        out = np.random.rand(1,4)
        out[0,0] = np.asscalar(self.y_set)
        out[0,1:3] = self.x.reshape(1,2)
        out[0,3] = self.e_int
        return out, self.r, self.done

    def error_calc(self):
        self.e_int = self.e_int + self.y_set - self.y

    def reward(self,step):
        diff = self.y_set - self.y
        diff = math.fabs( np.asscalar(diff[0]) )
        tuner = np.asscalar(self.y_set)
        if( diff < 0.01*tuner ):
            self.r = 5
        if(diff < 0.05*tuner) and (diff>0.01*tuner):
            self.r = 0.5
        if(diff < 0.1*tuner ) and (diff>0.05*tuner):
            self.r = 0.1
        else:
            self.r = -diff

    def reset(self): 
        # Init system output randomly in every episode.
        self.y = np.array([0])
        self.x = np.array([ [0.5],[0] ]) # np.array([ [h1],[h2] ]) 
        self.e_int = 0
        out = np.random.rand(1,4)
        out[0,0] = np.asscalar(self.y_set)
        out[0,1:3] = self.x.reshape(1,2)
        out[0,3] = self.e_int
        return out

    def if_done(self):
        diff = np.sum( np.absolute(self.y-self.y_set) )
        if ( diff < self.epsilon*0.1) : self.done = True
        else : self.done = False