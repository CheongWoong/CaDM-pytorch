import gymnasium as gym
from gymnasium.envs.registration import (
    load_env_plugins,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

import numpy as np

from .wrappers import HistoryWrapper, ContextWrapper


def make_env(env_id, idx, capture_video, run_name, history_length, len_future, env_config, gui=False):
    def thunk():
        if capture_video or gui:
            kwargs = {"render_mode": "rgb_array"}
        else:
            kwargs = {}
        kwargs.update(env_config)
        env = gym.make(env_id, **kwargs)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
        env = HistoryWrapper(env, history_length, len_future)
        env = ContextWrapper(env)
        return env

    return thunk


# Classic
# ----------------------------------------

register(
    id="cripple_ant",
    entry_point="src.envs.mujoco_envs.ant_cripple_env:CrippleAntEnv",
    max_episode_steps=1000,
)

register(
    id="ant",
    entry_point="src.envs.mujoco_envs.ant_env:AntEnv",
    max_episode_steps=1000,
)

register(
    id="cartpole",
    entry_point="src.envs.mujoco_envs.cartpole:CartPoleEnv",
    max_episode_steps=500,
)

register(
    id="cripple_halfcheetah",
    entry_point="src.envs.mujoco_envs.half_cheetah_cripple_env:CrippleHalfCheetahEnv",
    max_episode_steps=1000,
)

register(
    id="halfcheetah",
    entry_point="src.envs.mujoco_envs.half_cheetah_env:HalfCheetahEnv",
    max_episode_steps=1000,
)

register(
    id="hopper",
    entry_point="src.envs.mujoco_envs.hopper_env:HopperEnv",
    max_episode_steps=500,
)

register(
    id="pendulum",
    entry_point="src.envs.mujoco_envs.pendulum:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="slim_humanoid",
    entry_point="src.envs.mujoco_envs.slim_humanoid_env:SlimHumanoidEnv",
    max_episode_steps=1000,
)