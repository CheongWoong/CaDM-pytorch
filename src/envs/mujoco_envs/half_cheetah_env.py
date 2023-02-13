import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv

import numpy as np


class HalfCheetahEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25], **kwargs):
        observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float64),
                "context" : gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
            }
        )
        MuJocoPyEnv.__init__(
            self, "half_cheetah.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set, **kwargs)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, action):
        self.xposbefore = self.sim.data.qpos[0].copy()
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

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, act, next_obs):
        reward_ctrl = -0.1 * np.square(act).sum(axis=-1)
        reward_run = (next_obs[..., 0] - obs[..., 0]) / self.dt
        reward = reward_ctrl + reward_run

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