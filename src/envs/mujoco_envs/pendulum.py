__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np
import torch

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class PendulumEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self,
        render_mode: Optional[str] = None,
        mass_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25], 
        length_set=[0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25]
    ):
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.mass = 1.0
        self.length = 1.0

        self.render_mode = render_mode

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.obs_dim = 3
        self.proc_obs_dim = self.obs_dim

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.mass_set = mass_set
        self.length_set = length_set

    def step(self, u):
        th, thdot = self.state
        g = 10.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        angle_normalize = ((th+np.pi) % (2*np.pi))-np.pi

        costs = angle_normalize**2 + .1*thdot**2 + .001*((u/2.0)**2) # original

        newthdot = thdot + (-3*g/(2*self.length) * np.sin(th + np.pi) + 3./(self.mass*self.length**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        normalized = ((newth+np.pi) % (2*np.pi))-np.pi

        self.state = np.array([newth, newthdot])

        # Extra calculations for is_success()
        # TODO(cpacker): be consistent in increment before or after func body
        self.nsteps += 1
        # Track how long angle has been < pi/3
        if -np.pi/3 <= normalized and normalized <= np.pi/3:
            self.nsteps_vertical += 1
        else:
            self.nsteps_vertical = 0
        # Success if if angle has been kept at vertical for 100 steps
        target = 100
        if self.nsteps_vertical >= target:
            #print("[SUCCESS]: nsteps is {}, nsteps_vertical is {}, reached target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = True
        else:
            #print("[NO SUCCESS]: nsteps is {}, nsteps_vertical is {}, target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = False

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def obs_preproc(self, obs):
        return obs
    
    def obs_postproc(self, obs, pred):
        return obs + pred
    
    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Extra state for is_success()
        self.nsteps = 0
        self.nsteps_vertical = 0

        low = np.array([(7/8)*np.pi, -0.2])
        high = np.array([(9/8)*np.pi, 0.2])

        theta, thetadot = self.np_random.uniform(low=low, high=high)
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([theta, thetadot])

        self.last_u = None

        self.change_env()

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def reward(self, obs, action, next_obs):
        theta = torch.atan2(next_obs[...,1],next_obs[...,0])
        theta_normalize = ((theta + np.pi) % (2 * np.pi)) - np.pi
        thetadot  = next_obs[...,2]
        torque = torch.clamp(action, -self.max_torque, self.max_torque)
        torque = torch.reshape(torque, torque.shape[:-1])
        cost = theta_normalize**2 + 0.1*(thetadot)**2 + 0.001*(torque**2) # original
        return -cost

    def change_env(self):
        random_index = self.np_random.integers(len(self.mass_set))
        self.mass = self.mass_set[random_index]

        random_index = self.np_random.integers(len(self.length_set))
        self.length = self.length_set[random_index]        

    def get_sim_parameters(self):
        return np.array([self.mass, self.length])

    @property
    def num_modifiable_parameters(self):
        return 2

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi