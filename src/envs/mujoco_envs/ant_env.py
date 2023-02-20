import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class AntEnv(MujocoEnv, utils.EzPickle):
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
        xml_file="ant.xml",
        mass_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        damping_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            mass_scale_set,
            damping_scale_set,
            **kwargs
        )

        self.obs_dim = 28

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, action):
        self.xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = -0.005 * np.square(action).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        terminated = False
        observation = self._get_obs()
        info = {
            "reward_forward": reward_run,
            "reward_ctrl": reward_ctrl,
            "reward_contact": reward_contact,
            "reward_survive": reward_survive,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate([
            ((self.get_body_com("torso")[0] - self.xposbefore) / self.dt).flat,
            self.data.qpos.flat[2:],
            self.data.qvel.flat,
        ])

    def obs_preproc(self, obs):
        return obs[..., 1:]

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return torch.cat([pred[..., :1], obs[..., 1:] + pred[..., 1:]], dim=-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)

        self.change_env()

        observation = self._get_obs()

        return observation

    def reward(self, obs, action, next_obs):
        reward_ctrl = -0.005 * torch.sum(torch.square(action), dim=-1)
        reward_run = next_obs[..., 0]

        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        return reward

    def change_env(self):
        self.xposbefore = self.get_body_com("torso")[0]

        # Change mass
        random_index = self.np_random.integers(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]
        mass = np.copy(self.original_mass)
        mass[2:5] *= self.mass_scale
        mass[5:8] *= self.mass_scale
        mass[8:11] *= 1.0/self.mass_scale
        mass[11:14] *= 1.0/self.mass_scale
        self.model.body_mass[:] = mass

        # Change damping
        random_index = self.np_random.integers(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]
        damping = np.copy(self.original_damping)
        damping[2:5] *= self.damping_scale
        damping[5:8] *= self.damping_scale
        damping[8:11] *= 1.0/self.damping_scale
        damping[11:14] *= 1.0/self.damping_scale
        self.model.dof_damping[:] = damping

    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    @property
    def num_modifiable_parameters(self):
        return 2