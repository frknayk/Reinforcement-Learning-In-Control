# Reinforcement-Learning-in-Control-Applications
Investigating applicability of SOTA RL algorithms in linear/nonlinear control systems

***************************************************
'gym.py' file, mathematical model of linear/nonlinear system is given. You can change it for any system.

'train.py' contains all of DDPG algorithm code for training.

During the training all of data is recorded to csv files to plot reward,control signal, output changes
during the training.

'inference.py' simulates step response

'stepTracking.py' simulates step tracking with given reference

***************************************************

To plot csv files I used MATLAB. 
(You can use free version of MATLAB, octave to plot graphs)

To see reward change and most reward maximizing episode's step respnse,control signal change,
use 'readLog.m'.Before this run inference.py file.

In read_StepTrack.m is used to plot step tracking performance. Before this run stepTracking.py file.

***************************************************
Version_1 includes training for onyl one reference point.

Version_2 includes training for difference reference points.
