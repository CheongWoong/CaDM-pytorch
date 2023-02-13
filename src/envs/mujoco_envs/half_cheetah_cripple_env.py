import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv

import numpy as np


class CrippleHalfCheetahEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, cripple_set=[0, 1, 2, 3], extreme_set=[0], mass_scale_set=[1.0], **kwargs):
        observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64),
                "context" : gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
            }
        )
        MuJocoPyEnv.__init__(
            self, "half_cheetah.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, cripple_set, extreme_set, mass_scale_set, **kwargs)

        self.cripple_mask = np.ones(self.action_space.shape)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.original_mass = np.copy(self.model.body_mass)
        self.mass_scale_set = mass_scale_set

    def step(self, action):
        self.xposbefore = self.sim.data.qpos[0].copy()
        if self.cripple_mask is None:
            action = action
        else:
            action = self.cripple_mask * action
        self.do_simulation(action, self.frame_skip)
        self.xposafter = self.sim.data.qpos[0].copy()

        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (self.xposafter - self.xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        terminated = False

        if self.render_mode == "human":
            self.render()
        return (
            ob,
            reward,
            terminated,
            False,
            dict(reward_run=reward_run, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.xposafter.flat,
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return torch.cat([obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]], dim=-1)

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        self.xposafter = self.xposbefore = self.sim.data.qpos[0].copy()

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, act, next_obs):
        reward_ctrl = -0.1 * np.square(act).sum(axis=-1)
        reward_run = (next_obs[..., 0] - obs[..., 0]) / self.dt
        reward = reward_ctrl + reward_run
        return reward

    def change_env(self):
        action_dim = self.action_space.shape
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(self.cripple_set, 2, replace=False)
        else:
            raise ValueError(self.extreme_set)
        self.cripple_mask = np.ones(action_dim)
        self.cripple_mask[self.crippled_joint] = 0

        geom_rgba = self._init_geom_rgba.copy()
        for joint in self.crippled_joint:
            geom_idx = self.model.geom_names.index(self.model.joint_names[joint+3])
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba.copy()

        mass = np.copy(self.original_mass)
        mass *= self.mass_scale
        self.model.body_mass[:] = mass
    
    def get_sim_parameters(self):
        return np.concatenate(np.array([self.crippled_joint]).reshape(-1), np.array([self.mass_scale]))
    
    def num_modifiable_parameters(self):
        return 2

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5