import numpy as np

import gymnasium as gym


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length, future_length):
        super().__init__(env)

        self.history_length = history_length
        self.future_length = future_length
        self.obs_dim = env.observation_space.shape if hasattr(env.observation_space, "shape") else tuple(env.observation_space.n)

        self.history_obs = np.zeros((self.history_length, *self.obs_dim))
        self.history_obs_delta = np.zeros((self.history_length, *self.obs_dim))
        self.history_obs_back_delta = np.zeros((self.history_length, *self.obs_dim))
        self.history_act = np.zeros((self.history_length, *self.action_space.shape))
        self.history_mask = np.zeros((self.history_length,))
        self.future_obs = np.zeros((self.future_length, *self.obs_dim))
        self.future_obs_delta = np.zeros((self.future_length, *self.obs_dim))
        self.future_obs_back_delta = np.zeros((self.future_length, *self.obs_dim))
        self.future_act = np.zeros((self.future_length, *self.action_space.shape))
        self.future_mask = np.zeros((self.future_length,))
        self.true = np.ones((1, 1))
        self.prev_state = None
        
        history_observation_space = gym.spaces.Dict({
            "obs" : env.observation_space,
            "history_obs" : env.observation_space.__class__(-np.inf, np.inf, (history_length, *self.obs_dim)),
            "history_obs_delta" : env.observation_space.__class__(-np.inf, np.inf, (history_length, *self.obs_dim)),
            "history_obs_back_delta" : env.observation_space.__class__(-np.inf, np.inf, (history_length, *self.obs_dim)),
            "history_act" : env.action_space.__class__(-np.inf, np.inf, (history_length, *self.action_space.shape)),
            "history_mask" : env.observation_space.__class__(-np.inf, np.inf, (history_length, 1)),
            "future_obs" : env.observation_space.__class__(-np.inf, np.inf, (future_length, *self.obs_dim)),
            "future_obs_delta" : env.observation_space.__class__(-np.inf, np.inf, (future_length, *self.obs_dim)),
            "future_obs_back_delta" : env.observation_space.__class__(-np.inf, np.inf, (future_length, *self.obs_dim)),
            "future_act" : env.action_space.__class__(-np.inf, np.inf, (future_length, *self.action_space.shape)),
            "future_mask" : env.observation_space.__class__(-np.inf, np.inf, (future_length, 1)),
        })
        self.observation_space = history_observation_space

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # done = terminated or truncated
        
        self.history_obs = np.concatenate([self.history_obs[1:], self.prev_state[None,:]], axis=0)
        self.history_obs_delta = np.concatenate([self.history_obs_delta[1:], self.targ_proc(self.prev_state, observation)[None,:]], axis=0)
        self.history_obs_back_delta = np.concatenate([self.history_obs_back_delta[1:], self.targ_proc(observation, self.prev_state)[None,:]], axis=0)
        self.history_act = np.concatenate([self.history_act[1:], action[None,:]], axis=0)
        self.history_mask = np.concatenate([self.history_mask[1:], self.true], axis=0)
        self.future_obs = np.concatenate([self.future_obs[1:], self.prev_state[None,:]], axis=0)
        self.future_obs_delta = np.concatenate([self.future_obs_delta[1:], self.targ_proc(self.prev_state, observation)[None,:]], axis=0)
        self.future_obs_back_delta = np.concatenate([self.future_obs_back_delta[1:], self.targ_proc(observation, self.prev_state)[None,:]], axis=0)
        self.future_act = np.concatenate([self.future_act[1:], action[None,:]], axis=0)
        self.future_mask = np.concatenate([self.future_mask[1:], self.true], axis=0)
        self.prev_state = observation

        observations = {"obs": observation}
        history = {
            "history_obs": self.history_obs.copy(),
            "history_obs_delta": self.history_obs_delta.copy(),
            "history_obs_back_delta": self.history_obs_back_delta.copy(),
            "history_act": self.history_act.copy(),
            "history_mask": self.history_mask.copy(),
            "future_obs": self.future_obs.copy(),
            "future_obs_delta": self.future_obs_delta.copy(),
            "future_obs_back_delta": self.future_obs_back_delta.copy(),
            "future_act": self.future_act.copy(),
            "future_mask": self.future_mask.copy(),
        }
        observations.update(history)
        
        return observations, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)

        self.history_obs = np.zeros((self.history_length, *self.obs_dim))
        self.history_obs_delta = np.zeros((self.history_length, *self.obs_dim))
        self.history_obs_back_delta = np.zeros((self.history_length, *self.obs_dim))
        self.history_act = np.zeros((self.history_length, *self.action_space.shape))
        self.history_mask = np.zeros((self.history_length, 1))
        self.future_obs = np.zeros((self.future_length, *self.obs_dim))
        self.future_obs_delta = np.zeros((self.future_length, *self.obs_dim))
        self.future_obs_back_delta = np.zeros((self.future_length, *self.obs_dim))
        self.future_act = np.zeros((self.future_length, *self.action_space.shape))
        self.future_mask = np.zeros((self.future_length, 1))
        self.prev_state = observation

        observations = {"obs": observation}
        history = {
            "history_obs": self.history_obs.copy(),
            "history_obs_delta": self.history_obs_delta.copy(),
            "history_obs_back_delta": self.history_obs_back_delta.copy(),
            "history_act": self.history_act.copy(),
            "history_mask": self.history_mask.copy(),
            "future_obs": self.future_obs.copy(),
            "future_obs_delta": self.future_obs_delta.copy(),
            "future_obs_back_delta": self.future_obs_back_delta.copy(),
            "future_act": self.future_act.copy(),
            "future_mask": self.future_mask.copy(),
        }
        observations.update(history)

        return observations, info


if __name__ == "__main__":
    pass