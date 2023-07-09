import gym
import ray
from env_control import ConfigSISO, asdict
from ray.rllib.algorithms.ddpg import DDPGConfig, ddpg
from ray.tune.registry import register_env

config_siso = ConfigSISO(
    action_space=[-1, 1],
    obs_space=[-10, 10],
    num=[1],
    den=[1, 10, 20],
    x_0=[0],
    dt=0.01,
    y_0=0,
    t_0=0,
    t_end=5,
    y_ref=5,
)
# env = LinearSISOEnv(config)
env_config = asdict(config_siso)


def env_creator(env_name):
    if env_name == "siso-v0":
        from env_control import LinearSISOEnv as env
    else:
        raise NotImplementedError
    return env


env = env_creator("siso-v0")
register_env("siso-v0", env_creator("siso-v0"))


if __name__ == "__main__":
    from ray import tune

    ray.init()
    tune.run(
        "DDPG",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": "siso-v0",  # <--- This works fine!
            "env_config": env_config,
        },
    )
