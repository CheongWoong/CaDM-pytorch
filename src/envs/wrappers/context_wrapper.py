import numpy as np

import gymnasium as gym


class ContextWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        context_dim = self.num_modifiable_parameters
        
        context_observation_space = {
            "context" : gym.spaces.Box(-np.inf, np.inf, (context_dim,))
        }
        self.observation_space.spaces.update(context_observation_space)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # done = terminated or truncated

        context = {
            "context": self.context.copy(),
        }
        observation.update(context)
        
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        self.context = self.get_sim_parameters()

        context = {
            "context": self.context.copy(),
        }
        observation.update(context)

        return observation, info


if __name__ == "__main__":
    pass