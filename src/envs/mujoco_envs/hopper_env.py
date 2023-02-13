import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv

import numpy as np
import torch


class HopperEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25], **kwargs):
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, "hopper.xml", 4, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set, **kwargs)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, a):
        self.posbefore = self.sim.data.qpos[0].copy()
        self.do_simulation(a, self.frame_skip)
        self.posafter, height, ang = self.sim.data.qpos[0:3].copy()

        alive_bonus = 1.0
        reward = (self.posafter - self.posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        terminated = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()
        return ob, reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate(
            [
                self.posafter.flat,
                self.sim.data.qpos.flat[1:],
                np.clip(self.sim.data.qvel.flat, -10, 10)
            ]
        )

    def obs_preproc(self, obs):
        return obs[..., 1:]

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

        self.posafter = self.posbefore = self.sim.data.qpos[0].copy()

        random_index = self.np_random.integers(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.integers(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, act, next_obs):
        alive_bonus = 1.0
        reward = (next_obs[..., 0] - obs[..., 0]) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(act).sum(axis=-1)

        return reward

    def torch_reward(self, obs, act, next_obs):
        alive_bonus = 1.0
        reward = (next_obs[..., 0] - obs[..., 0]) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * torch.square(act).sum(axis=-1)

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
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20