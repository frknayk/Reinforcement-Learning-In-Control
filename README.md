# Designing Controllers via Deep Reinforcement Learning


## About
Apply state-of-the-art RL algorithms to control linear/nonlinear dynamical systems


## Installing
<!-- Create conda environment -->
<!-- conda env export > environment.yml --no-builds -->
'conda env create --name rlcontrol --file=environment.yml'

Install gym_control environment: `pip install -e .`
## Training
- Look for raylib
* Track experiments via: tensorboard --logdir=~/ray_results


<img width=640px height=480px src="images\result.png" alt="Project logo">

## TODO
    - Train environemnt with rllib
    - Integrate integral error to rlcontrol
    - Replace all prints, with logger
    - Streamlit integration(enter dynamical system from app)
    - Dockerize

## DONE
    - Create custom gym env for simulating dynamical system responses
    - Integrate custom gym env to rllib
