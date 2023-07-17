from gymnasium.envs.registration import register

register(
    id="LinearSISOEnv-v0",
    entry_point="gym_control.envs:LinearSISOEnv",
    max_episode_steps=1000000,
)
