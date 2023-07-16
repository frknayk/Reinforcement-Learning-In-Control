from typing import List

agent_config_default = {"batch_size": int, "hidden_dim": int, "algorithm_type": str}


env_config_default = {
    "action_space": list,
    "obs_space": list,
    "num": list,
    "den": list,
    "x_0": list,
    "dt": float,
    "y_0": float,
    "t_0": float,
    "t_end": float,
    "y_ref": float,
    "steady_state_indicator": int,
}

# Default 1-episode output structure
episode_result_dict_default = {
    "episode_reward": float,
    "episode_policy_loss": float,
    "episode_value_loss": float,
    "total_control_signal": float,
    "total_output_signal": float,
    "step_total": int,
    "output_list": List[float],
    "reward_list": List[float],
    "reference_list": List[float],
    "control_sig_list": List[float],
    "sim_time": List[float],
}
