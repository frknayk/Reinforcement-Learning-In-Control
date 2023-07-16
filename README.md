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

## Run App
streamlit run app.py


<img width=640px height=480px src="images\result.png" alt="Project logo">

## TODO
    - Add RL algorithm params to training page.
    - Track experiments from db (postgresql)
    - Integrate integral error to rlcontrol
    - Train environemnt with rllib
    - Add class diagram of project
    - Dockerize

## DONE
    - Add Training and Inference Pages. And Inference page must be able to load
    latest training's best agent in selected folder to test agent.
    - Do not create folder unless checkpoint saving is enabled.
    - Add Experiment name to Introduction page
    - Make app multiple pages
    - Make Trainer runnable by episode by episode, and plot graphs to streamlit.
    - Create custom gym env for simulating dynamical system responses
    - Integrate custom gym env to rllib
    - Streamlit integration(enter dynamical system from app)
    - Replace all prints, with logger
