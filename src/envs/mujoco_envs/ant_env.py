import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv

import numpy as np
import torch


class AntEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, mass_scale_set=[0.85, 0.9, 0.95, 1.0], damping_scale_set=[1.0], **kwargs):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(112,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "ant.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set, **kwargs)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, a):
        self.xposbefore = self.get_body_com("torso")[0].copy()
        self.do_simulation(a, self.frame_skip)
        self.xposafter = self.get_body_com("torso")[0].copy()

        forward_reward = (self.xposafter - self.xposbefore) / self.dt
        ctrl_cost = 0.005 * np.square(a).sum()
        contact_cost = 0.0
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        not_terminated = (
            np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        )
        terminated = not not_terminated
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()
        return (
            ob,
            reward,
            terminated,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.xposafter.flat,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )
        return obs

    def obs_preproc(self, obs):
        return obs[..., 1:]

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        self.xposafter = self.xposbefore = self.get_body_com("torso")[0].copy()

        random_index = self.np_random.integers(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.integers(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, act, next_obs):
        ctrl_cost = 0.005 * np.square(act).sum(axis=-1)
        forward_reward = (next_obs[..., 0] - obs[..., 0]) / self.dt
        contact_cost = 0.0
        survive_reward = 0.05

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        return reward

    def torch_reward(self, obs, act, next_obs):
        ctrl_cost = 0.005 * torch.square(act).sum(dim=-1)
        forward_reward = (next_obs[..., 0] - obs[..., 0]) / self.dt
        contact_cost = 0.0
        survive_reward = 0.05

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        return reward

    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)

        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping
    
    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])
    
    def num_modifiable_parameters(self):
        return 2

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5