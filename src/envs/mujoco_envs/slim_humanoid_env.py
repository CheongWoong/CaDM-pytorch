import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MuJocoPyEnv

import numpy as np


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class SlimHumanoidEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25], **kwargs):
        observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64),
                "context" : gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64)
            }
        )
        MuJocoPyEnv.__init__(
            self, "humanoid.xml", 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set, **kwargs)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                self.pos_after.flat,
                data.qpos.flat[2:],
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def obs_preproc(self, obs):
        return obs[..., 1:]

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def step(self, a):
        self.pos_before = mass_center(self.model, self.sim).copy()
        self.do_simulation(a, self.frame_skip)
        self.pos_after = mass_center(self.model, self.sim).copy()

        alive_bonus = 5.0
        lin_vel_cost = 1.25 * (self.pos_after - self.pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(a).sum()
        quad_impact_cost = 0.0

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        terminated = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        if self.render_mode == "human":
            self.render()
        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-c, high=c, size=self.model.nv,),
        )
        self.pos_after = self.pos_before = mass_center(self.model, self.sim).copy()

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()

    def reward(self, obs, action, next_obs):
        alive_bonus = 5.0
        lin_vel_cost = 1.25 * (next_obs[..., 0] - obs[..., 0]) / self.dt
        quad_ctrl_cost = 0.1 * np.square(action).sum(axis=-1)
        quad_impact_cost = 0.0

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20