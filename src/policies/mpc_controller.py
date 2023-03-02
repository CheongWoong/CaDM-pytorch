"""
Copyright (c)
https://github.com/iclavera/learning_to_adapt
"""

import numpy as np
import torch

from ..dynamics.core.layers.utils import normalize, denormalize


class MPCController():
    def __init__(
        self,
        args,
        envs,
        dynamics_model,
    ):
        self.args = args
        self.envs = envs
        self.dynamics_model = dynamics_model

        self.gamma = args.gamma
        self.n_candidates = args.n_candidates
        self.n_particles = args.n_particles
        self.ensemble_size = args.ensemble_size
        self.horizon = args.horizon
        self.use_cem = args.use_cem
        self.num_cem_iters = args.num_cem_iters
        self.percent_elites = args.percent_elites
        self.alpha = args.alpha
        self.num_elites = max(int(self.n_candidates * self.percent_elites), 1)

        self.dummy_env = envs.envs[0].unwrapped
        # make sure that env has a reward function
        assert hasattr(self.dummy_env, 'reward'), "env must have a reward function"

        self.device = args.device

        self.action_low = torch.from_numpy(self.dummy_env.action_space.low).to(self.device)
        self.action_high = torch.from_numpy(self.dummy_env.action_space.high).to(self.device)
        self.prev_sol = torch.zeros((args.num_rollouts, self.horizon, args.action_dim), device=self.device)
        self.init_var = 0.25*torch.ones((args.num_rollouts, self.horizon, args.action_dim), device=self.device)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            self.prev_sol *= 0.
        elif isinstance(dones, np.ndarray):
            self.prev_sol = torch.where(torch.as_tensor(dones[:, None, None], device=self.device) > 0, 0., self.prev_sol)
        else:
            self.prev_sol = torch.where(dones[:, None, None] > 0, 0., self.prev_sol)

    def get_action(self, observation):
        observation = self.dynamics_model.apply_normalization(observation)
        observations = {key: torch.Tensor(observation[key]).to(self.device) for key in observation}
        if self.use_cem:
            action = self.get_cem_action(observations)
            self.prev_sol[:, :-1] = action[:, 1:].clone()
            self.prev_sol[:, -1:] *= 0.
            action = action[:, 0].clone()
        else:
            raise NotImplementedError

        return action

    def get_random_action(self, n):
        return torch.distributions.uniform.Uniform(self.action_low, self.action_high).sample((n,))

    def get_cem_action(self, observations):
        ensemble_size = self.ensemble_size
        n = self.n_candidates
        p = self.n_particles
        m = len(observations["obs"])
        h = self.horizon

        lower_bound = -1.0
        upper_bound = 1.0

        mean = self.prev_sol
        var = self.init_var

        expanded_observations = {}
        for key in observations:
            expanded_observations[key] = observations[key].repeat([ensemble_size]+[1]*len(observations[key].shape))
            
        context_output, _ = self.dynamics_model.get_context(expanded_observations)
        context = context_output["context"]
        
        for _ in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - lower_bound, upper_bound - mean
            constrained_var = torch.minimum(
                torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2)), var
            )
            repeated_mean = torch.tile(
                mean[:, None, :, :], [1, n, 1, 1]
            )           
            repeated_var = torch.tile(
                constrained_var[:, None, :, :], [1, n, 1, 1]
            )
            actions = torch.normal(
                repeated_mean, torch.sqrt(repeated_var)
            )
            # truncation
            actions = torch.clamp(actions, repeated_mean - 2*torch.sqrt(repeated_var), repeated_mean + 2*torch.sqrt(repeated_var))

            returns = 0
            observation = torch.tile(
                torch.reshape(observations["obs"], [m, 1, 1, self.args.obs_dim]), [1, n, p, 1]
            )
            if context is not None:
                reshaped_context = torch.permute(context, (1, 0, 2))
                reshaped_context = reshaped_context[
                    :, None, :, None, :
                ]
                reshaped_context = torch.tile(
                    reshaped_context, [1, n, 1, int(p / ensemble_size), 1]
                )
                reshaped_context = torch.permute(
                    reshaped_context, [2, 3, 0, 1, 4]
                )
                reshaped_context = torch.reshape(
                    reshaped_context,
                    [
                        ensemble_size,
                        int(p / ensemble_size) * m * n,
                        self.args.context_out_dim,
                    ]
                )
            else:
                reshaped_context = None

            for t in range(h):
                action = actions[:, :, t]
                normalized_action = normalize(
                    action, self.dynamics_model.torch_normalization["act"][0], self.dynamics_model.torch_normalization["act"][1]
                )
                normalized_action = torch.tile(
                    normalized_action[:, :, None, :], [1, 1, p, 1]
                )
                normalized_action = torch.reshape(
                    torch.permute(normalized_action, [2, 0, 1, 3]),
                    [ensemble_size, int(p / ensemble_size) * m * n, self.args.action_dim]
                )
                
                proc_obs = self.args.obs_preproc(observation)
                normalized_proc_obs = normalize(
                    proc_obs, self.dynamics_model.torch_normalization["obs"][0], self.dynamics_model.torch_normalization["obs"][1]
                )
                normalized_proc_obs = torch.reshape(
                    torch.permute(normalized_proc_obs, [2, 0, 1, 3]),
                    [ensemble_size, int(p / ensemble_size) * m * n, self.args.proc_obs_dim]
                )
                
                x = {"normalized_proc_obs": normalized_proc_obs, "normalized_act": normalized_action}
                
                delta = self.dynamics_model.predict(x, reshaped_context)
                delta = torch.permute(
                    torch.reshape(
                        delta,
                        [
                            ensemble_size,
                            int(p / ensemble_size),
                            m,
                            n,
                            self.args.obs_dim,
                        ],
                    ),
                    [0, 2, 1, 3, 4]
                )
                delta = torch.reshape(
                    delta,
                    [
                        ensemble_size * m,
                        int(p / ensemble_size),
                        n,
                        self.args.obs_dim,
                    ]
                )

                delta = torch.permute(
                    torch.reshape(
                        delta,
                        [ensemble_size, m, int(p / ensemble_size), n, self.args.obs_dim],
                    ),
                    [1, 3, 0, 2, 4]
                )
                delta = torch.reshape(delta, [m, n, p, self.args.obs_dim])

                next_observation = self.args.obs_postproc(
                    observation, delta
                )
                repeated_action = torch.tile(action[:, :, None, :], [1, 1, p, 1])
                reward = self.dummy_env.reward(observation, repeated_action, next_observation)

                returns += self.gamma ** t * reward
                observation = next_observation

            returns = returns.mean(2)
            _, elites_idx = torch.topk(
                returns, k=self.num_elites, sorted=True
            )
            elites_idx += torch.arange(0, m * n, n, device=self.device)[:, None]
            flat_elites_idx = torch.reshape(
                elites_idx, [m * self.num_elites]
            )
            flat_actions = torch.reshape(actions, [m * n, h, self.args.action_dim])
            flat_elites = flat_actions[flat_elites_idx]
            elites = torch.reshape(flat_elites, [m, self.num_elites, h, self.args.action_dim])
            
            new_mean = torch.mean(elites, dim=1)
            new_var = torch.mean(
                torch.square(elites - new_mean[:, None, :, :]), dim=1
            )

            mean = mean * self.alpha + (1 - self.alpha) * new_mean
            var = var * self.alpha + (1 - self.alpha) * new_var
        return mean