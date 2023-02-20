import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class CrippleAntEnv(MujocoEnv, utils.EzPickle):
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
        cripple_set=[0, 1, 2],
        extreme_set=[0],
        mass_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        damping_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25],
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            cripple_set,
            extreme_set,
            mass_scale_set,
            damping_scale_set,
            **kwargs
        )

        self.obs_dim = 32
        self.proc_obs_dim = self.obs_dim

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

        self.n_possible_cripple = 4
        self.cripple_mask = np.ones(self.n_possible_cripple)
        self.cripple_set = cripple_set
        self.extreme_set = extreme_set

        self.cripple_dict = {
            0: [2, 3],  # front L
            1: [4, 5],  # front R
            2: [6, 7],  # back L
            3: [0, 1],  # back R
        }

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

    def step(self, action):
        self.xposbefore = self.get_body_com("torso")[0]
        if self.cripple_mask is not None:
            action = self.cripple_mask * action
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = 0.0
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
        return np.concatenate(
            [
                self.data.qpos.flat,
                self.data.qvel.flat,
                self.get_body_com("torso"),
            ]
        )

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)

        self.change_env()

        observation = self._get_obs()

        return observation

    def reward(self, obs, action, next_obs):
        reward_ctrl = 0.0
        vel = (next_obs[..., -3] - obs[..., -3]) / self.dt
        reward_run = vel

        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        return reward

    def set_crippled_joint(self, value):
        self.cripple_mask = np.ones(self.action_space.shape)
        if value == 0:
            self.cripple_mask[2] = 0
            self.cripple_mask[3] = 0
        elif value == 1:
            self.cripple_mask[4] = 0
            self.cripple_mask[5] = 0
        elif value == 2:
            self.cripple_mask[6] = 0
            self.cripple_mask[7] = 0
        elif value == 3:
            self.cripple_mask[0] = 0
            self.cripple_mask[1] = 0
        elif value == -1:
            pass

        self.crippled_leg = value

        # Make the removed leg look red
        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([1, 0, 0])
            geom_rgba[4, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 0])
            geom_rgba[7, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 0, 0])
            geom_rgba[10, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba

        # Make the removed leg not affect anything
        temp_size = self._init_geom_size.copy()
        temp_pos = self._init_geom_pos.copy()

        if self.crippled_leg == 0:
            # Top half
            temp_size[3, 0] = temp_size[3, 0] / 2
            temp_size[3, 1] = temp_size[3, 1] / 2
            # Bottom half
            temp_size[4, 0] = temp_size[4, 0] / 2
            temp_size[4, 1] = temp_size[4, 1] / 2
            temp_pos[4, :] = temp_pos[3, :]

        elif self.crippled_leg == 1:
            # Top half
            temp_size[6, 0] = temp_size[6, 0] / 2
            temp_size[6, 1] = temp_size[6, 1] / 2
            # Bottom half
            temp_size[7, 0] = temp_size[7, 0] / 2
            temp_size[7, 1] = temp_size[7, 1] / 2
            temp_pos[7, :] = temp_pos[6, :]

        elif self.crippled_leg == 2:
            # Top half
            temp_size[9, 0] = temp_size[9, 0] / 2
            temp_size[9, 1] = temp_size[9, 1] / 2
            # Bottom half
            temp_size[10, 0] = temp_size[10, 0] / 2
            temp_size[10, 1] = temp_size[10, 1] / 2
            temp_pos[10, :] = temp_pos[9, :]

        elif self.crippled_leg == 3:
            # Top half
            temp_size[12, 0] = temp_size[12, 0] / 2
            temp_size[12, 1] = temp_size[12, 1] / 2
            # Bottom half
            temp_size[13, 0] = temp_size[13, 0] / 2
            temp_size[13, 1] = temp_size[13, 1] / 2
            temp_pos[13, :] = temp_pos[12, :]

        self.model.geom_size[:] = temp_size
        self.model.geom_pos[:] = temp_pos

    def change_env(self):
        self.xposbefore = self.get_body_com("torso")[0]

        # Pick which legs to remove
        if self.extreme_set == [0]:
            self.crippled_joint = np.array([self.np_random.choice(self.cripple_set)])
        elif self.extreme_set == [1]:
            self.crippled_joint = self.np_random.choice(self.cripple_set, 2, replace=False)
        else:
            raise ValueError(self.extreme_set)

        # Pick which actuators to disable
        self.cripple_mask = np.ones(self.action_space.shape)
        total_crippled_joints = []
        for j in self.crippled_joint:
            total_crippled_joints += self.cripple_dict[j]
        self.cripple_mask[total_crippled_joints] = 0        

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