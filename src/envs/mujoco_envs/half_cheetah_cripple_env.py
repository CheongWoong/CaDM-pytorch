__credits__ = ["Rushiv Arora"]

import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CrippleHalfCheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        cripple_set=[0, 1, 2, 3],
        extreme_set=[0],
        mass_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        damping_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            cripple_set,
            extreme_set,
            mass_scale_set,
            damping_scale_set,
            **kwargs
        )

        self.obs_dim = 18
        self.proc_obs_dim = self.obs_dim

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.n_possible_cripple = 4
        self.cripple_mask = np.ones(self.n_possible_cripple)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, action):
        self.prev_qpos = np.copy(self.data.qpos.flat)
        if self.cripple_mask is not None:
            action = self.cripple_mask * action
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        reward_ctrl = -0.1  * np.square(action).sum()
        reward_run = observation[0]
        reward = reward_run + reward_ctrl

        terminated = False
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return torch.cat([obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]], dim=-1)

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return torch.cat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], dim=-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)

        self.change_env()

        observation = self._get_obs()
        return observation

    def reward(self, obs, action, next_obs):
        ctrl_cost = 1e-1 * torch.sum(torch.square(action), dim=-1)
        forward_reward = next_obs[..., 0]
        reward = forward_reward - ctrl_cost
        return reward

    def change_env(self):
        self.prev_qpos = np.copy(self.data.qpos.flat)

        # Pick which legs to remove
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(self.cripple_set, 2, replace=False)
        else:
            raise ValueError(self.extreme_set)

        # Pick which actuators to disable
        self.cripple_mask = np.ones(self.action_space.shape)
        self.cripple_mask[self.crippled_joint] = 0        

        # Change mass
        random_index = self.np_random.integers(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]
        mass = np.copy(self.original_mass)
        mass *= self.mass_scale
        self.model.body_mass[:] = mass

        # Change damping
        random_index = self.np_random.integers(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]
        damping = np.copy(self.original_damping)
        damping *= self.damping_scale
        self.model.dof_damping[:] = damping

    def get_sim_parameters(self):
        return np.concatenate([np.array([self.mass_scale, self.damping_scale]), self.crippled_joint.reshape(-1)])

    @property
    def num_modifiable_parameters(self):
        if self.extreme_set == [0]:
            return 3
        else:
            return 4