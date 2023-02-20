import numpy as np

import gymnasium as gym


class ContextWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        sim_param_dim = self.num_modifiable_parameters
        
        context_observation_space = {
            "sim_params" : gym.spaces.Box(-np.inf, np.inf, (sim_param_dim,))
        }
        self.observation_space.spaces.update(context_observation_space)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # done = terminated or truncated

        sim_params = {
            "sim_params": self.sim_params.copy(),
        }
        observation.update(sim_params)
        
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        self.sim_params = self.get_sim_parameters()

        sim_params = {
            "sim_params": self.sim_params.copy(),
        }
        observation.update(sim_params)

        return observation, info


if __name__ == "__main__":
    pass