import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        mass_scale_set=[0.5, 0.75, 1.0, 1.25, 1.5],
        damping_scale_set=[0.5, 0.75, 1.0, 1.25, 1.5],
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            mass_scale_set,
            damping_scale_set,
            **kwargs
        )

        self.obs_dim = 7

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            "hopper.xml",
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, action):
        posbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter = self.data.qpos[0]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()
        terminated = False
        observation = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos.flat[1:], np.clip(self.data.qvel.flat, -10, 10)]
        )

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self.change_env()

        observation = self._get_obs()
        return observation

    def reward(self, obs, action, next_obs):
        velocity = next_obs[..., 5]
        alive_bonus = 1.0
        reward = velocity
        reward += alive_bonus
        reward -= 1e-3 * torch.square(action).sum(dim=-1)
        return reward

    def change_env(self):
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
        return np.array([self.mass_scale, self.damping_scale])

    @property
    def num_modifiable_parameters(self):
        return 2

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)