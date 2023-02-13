import numpy as np

import gymnasium as gym


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, len_history, len_future):
        super().__init__(env)

        self.len_history = len_history
        self.len_future = len_future
        self.obs_dim = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)

        self.history_obs = np.zeros((self.len_history, *self.obs_dim))
        self.history_obs_delta = np.zeros((self.len_history, *self.obs_dim))
        self.history_act = np.zeros((self.len_history, *self.action_space.shape))
        self.future_obs = np.zeros((self.len_future, *self.obs_dim))
        self.future_obs_delta = np.zeros((self.len_future, *self.obs_dim))
        self.future_act = np.zeros((self.len_future, *self.action_space.shape))
        self.future_bool = np.zeros((self.len_future,))
        self.true = np.ones(1)
        self.prev_state = None
        
        history_observation_space = gym.spaces.Dict({
            "obs" : env.observation_space,
            "history_obs" : env.observation_space.__class__(-np.inf, np.inf, (len_history, *self.obs_dim)),
            "history_obs_delta" : env.observation_space.__class__(-np.inf, np.inf, (len_history, *self.obs_dim)),
            "history_act" : env.action_space.__class__(-np.inf, np.inf, (len_history, *self.action_space.shape)),
            "future_obs" : env.observation_space.__class__(-np.inf, np.inf, (len_future, *self.obs_dim)),
            "future_obs_delta" : env.observation_space.__class__(-np.inf, np.inf, (len_future, *self.obs_dim)),
            "future_act" : env.action_space.__class__(-np.inf, np.inf, (len_future, *self.action_space.shape)),
            "future_bool" : env.observation_space.__class__(-np.inf, np.inf, (len_future,)),
        })
        self.observation_space = history_observation_space

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # done = terminated or truncated
        
        self.history_obs = np.concatenate([self.history_obs[1:], self.prev_state[None,:]], axis=0)
        self.history_obs_delta = np.concatenate([self.history_obs_delta[1:], (observation - self.prev_state)[None,:]], axis=0)
        self.history_act = np.concatenate([self.history_act[1:], action[None,:]], axis=0)
        self.future_obs = np.concatenate([self.future_obs[1:], self.prev_state[None,:]], axis=0)
        self.future_obs_delta = np.concatenate([self.future_obs_delta[1:], (observation - self.prev_state)[None,:]], axis=0)
        self.future_act = np.concatenate([self.future_act[1:], action[None,:]], axis=0)
        self.future_bool = np.concatenate([self.future_bool[1:], self.true], axis=0)
        self.prev_state = observation

        observations = {"obs": observation}
        history = {
            "history_obs": self.history_obs.copy(),
            "history_obs_delta": self.history_obs_delta.copy(),
            "history_act": self.history_act.copy(),
            "future_obs": self.future_obs.copy(),
            "future_obs_delta": self.future_obs_delta.copy(),
            "future_act": self.future_act.copy(),
            "future_bool": self.future_bool.copy(),
        }
        observations.update(history)
        
        return observations, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        self.history_obs = np.zeros((self.len_history, *self.obs_dim))
        self.history_obs_delta = np.zeros((self.len_history, *self.obs_dim))
        self.history_act = np.zeros((self.len_history, *self.action_space.shape))
        self.future_obs = np.zeros((self.len_future, *self.obs_dim))
        self.future_obs_delta = np.zeros((self.len_future, *self.obs_dim))
        self.future_act = np.zeros((self.len_future, *self.action_space.shape))
        self.future_bool = np.zeros((self.len_future,))
        self.prev_state = observation

        observations = {"obs": observation}
        history = {
            "history_obs": self.history_obs.copy(),
            "history_obs_delta": self.history_obs_delta.copy(),
            "history_act": self.history_act.copy(),
            "future_obs": self.future_obs.copy(),
            "future_obs_delta": self.future_obs_delta.copy(),
            "future_act": self.future_act.copy(),
            "future_bool": self.future_bool.copy(),
        }
        observations.update(history)

        return observations, info


if __name__ == "__main__":
    pass