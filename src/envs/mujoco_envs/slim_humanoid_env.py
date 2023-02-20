import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class SlimHumanoidEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 67,
    }

    def __init__(self, mass_scale_set=[0.75, 1.0, 1.25], damping_scale_set=[0.75, 1.0, 1.25], **kwargs):
        utils.EzPickle.__init__(
            self,
            mass_scale_set,
            damping_scale_set,
            **kwargs
        )

        self.obs_dim = 7
        self.proc_obs_dim = self.obs_dim

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )

        
        MujocoEnv.__init__(
            self, "humanoid.xml", 5, observation_space=observation_space, **kwargs
        )

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, a):
        old_obs = np.copy(self._get_obs())
        self.do_simulation(a, self.frame_skip)
        data = self.data
        lin_vel_cost = 0.25 / 0.015 * old_obs[..., 22]
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.0
        qpos = self.data.qpos
        terminated = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        alive_bonus = 5.0 * (1 - float(terminated))
        terminated = False
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

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

    def _get_obs(self):
        data = self.data
        return np.concatenate([data.qpos.flat[2:], data.qvel.flat])

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-c, high=c, size=self.model.nv,),
        )
        pos_before = mass_center(self.model, self)
        self.prev_pos = np.copy(pos_before)

        self.change_env()

        return self._get_obs()

    def reward(self, obs, action, next_obs):
        ctrl = action

        lin_vel_cost = 0.25 / 0.015 * next_obs[..., 22]
        quad_ctrl_cost = 0.1 * np.sum(np.square(ctrl), axis=-1)
        quad_impact_cost = 0.0

        done = bool((next_obs[..., 1] < 1.0) or (next_obs[..., 1] > 2.0))
        alive_bonus = 5.0 * (not done)

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20