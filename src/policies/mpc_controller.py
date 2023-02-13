"""
Copyright (c)
https://github.com/iclavera/learning_to_adapt
"""

from copy import deepcopy
import torch

from .base import Policy


class MPCController(Policy):
    def __init__(
        self,
        args,
        envs,
        dynamics_model,
        reward_model=None,
        use_reward_model=False,
    ):
        self.args = args
        self.envs = envs
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.use_reward_model = use_reward_model

        self.discount = args.mpc.discount
        self.n_candidates = args.mpc.n_candidates
        self.n_particles = args.mpc.n_particles
        self.ensemble_size = args.mpc.ensemble_size
        self.horizon = args.mpc.horizon
        self.use_cem = args.mpc.use_cem
        self.num_cem_iters = args.mpc.num_cem_iters
        self.percent_elites = args.mpc.percent_elites
        self.alpha = args.mpc.alpha
        self.num_elites = max(int(self.n_candidates * self.percent_elites), 1)

        self.dummy_env = envs.envs[0].unwrapped
        # make sure that env has a reward function
        assert hasattr(self.dummy_env, 'reward'), "env must have a reward function"

        self.device = args.device

        self.action_low = torch.from_numpy(self.action_space.low).to(self.device)
        self.action_high = torch.from_numpy(self.action_space.high).to(self.device)
        self.prev_sol = torch.zeros((args.num_envs, self.horizon, args.action_dim), device=self.device)
        self.init_var = torch.ones((args.num_envs, self.horizon, args.action_dim), device=self.device)

        super().__init__(envs=envs)

    @property
    def vectorized(self):
        return True

    def reset(self, dones):
        self.prev_sol = torch.where(torch.as_tensor(dones[:, None, None], device=self.device), 0., self.prev_sol)

    def get_action(self, observation):
        if self.use_cem:
            action = self.get_cem_action(observation)
            self.prev_sol[:, :-1] = action[:, 1:].clone()
            self.prev_sol[:, -1:] = 0.
            action = action[:, 0].clone()
        else:
            action = self.get_rs_action(observation)

        return action

    def get_random_action(self, n):
        return torch.distributions.uniform.Uniform(self.action_low, self.action_high).sample((n,))

    def get_cem_action(self, observations):
        observations = deepcopy(observations)

        ensemble_size = self.ensemble_size
        n = self.n_candidates
        p = self.n_particles
        m = len(observations["obs"])
        h = self.horizon

        mean = self.prev_sol
        var = self.init_var

        context_output, _ = self.dynamics_model.get_context(observations)
        context = context_output["context"]
        
        for _ in range(self.num_cem_iters):
            repeated_mean = torch.tile(mean[:, None, :, :], [1, n, 1, 1])
            repeated_var = torch.tile(var[:, None, :, :], [1, n, 1, 1])
            actions = repeated_mean + torch.randn(repeated_mean.shape, device=self.device)*torch.sqrt(repeated_var)
            actions = torch.clamp(actions, self.action_low, self.action_high)

            returns = 0
            observation = torch.tile(torch.reshape(observations["obs"], [m, 1, 1, -1]), [1, n, p, 1])
            reshaped_context = torch.permute(context, (1, 0, 2))
            reshaped_context = torch.tile(torch.reshape(reshaped_context, [m, 1, ensemble_size, -1]), [1, n, int(p/ensemble_size), 1])
            reshaped_context = torch.reshape(torch.permute(reshaped_context, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])
            for t in range(h):
                action = actions[:, :, t]
                reshaped_action = torch.tile(action[:, :, None, :], [1, 1, p, 1])
                reshaped_action = torch.reshape(torch.permute(reshaped_action, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])

                reshaped_observation = torch.reshape(torch.permute(observation, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])

                x = {"obs": reshaped_observation, "action": reshaped_action}

                delta = self.dynamics_model.predict(x, reshaped_context)
                delta = torch.reshape(delta, [p, m, n, -1])
                delta = torch.permute(delta, [1, 2, 0, 3])

                obs_mean = torch.as_tensor(self.envs.obs_rms["obs"].mean[None, None, None, :], device=self.device)
                obs_var = torch.as_tensor(self.envs.obs_rms["obs"].var[None, None, None, :], device=self.device)
                delta_mean = torch.as_tensor(self.envs.obs_rms["future_obs_delta"].mean[-1][None, None, None, :], device=self.device)
                delta_var = torch.as_tensor(self.envs.obs_rms["future_obs_delta"].var[-1][None, None, None, :], device=self.device)
                denormalized_obs = observation * torch.sqrt(obs_var + self.envs.epsilon) + obs_mean
                denormalized_delta = delta * torch.sqrt(delta_var + self.envs.epsilon) + delta_mean

                denormalized_next_obs = self.dummy_env.obs_postproc(denormalized_obs, denormalized_delta)
                repeated_action = torch.tile(action[:, :, None, :], [1, 1, p, 1])

                if self.use_reward_model:
                    assert self.reward_model is not None
                    reward = self.reward_model.predict(observation, repeated_action, denormalized_next_obs)
                else:
                    reward = self.dummy_env.reward(observation, repeated_action, denormalized_next_obs)
                returns += self.discount ** t * reward
                next_observation = (denormalized_next_obs - obs_mean) / torch.sqrt(obs_var + self.envs.epsilon)
                observation = next_observation
            returns = returns.mean(-1)
            _, elites_idx = torch.topk(returns, k=self.num_elites, dim=-1, sorted=True)
            elites_idx += torch.arange(0, m*n, n, device=self.device)[:, None]
            flat_elites_idx = torch.reshape(elites_idx, [m*self.num_elites])
            flat_actions = torch.reshape(actions, [m*n, h, -1])
            flat_elites = flat_actions[flat_elites_idx]
            elites = torch.reshape(flat_elites, [m, self.num_elites, h, -1])
            
            new_mean = torch.mean(elites, 1)
            new_var = torch.var(elites, 1, unbiased=False)

            mean = mean * self.alpha + (1 - self.alpha) * new_mean
            var = var * self.alpha + (1 - self.alpha) * new_var
        return mean

    def get_rs_action(self, observations):
        observations = deepcopy(observations)

        ensemble_size = self.ensemble_size
        n = self.n_candidates
        p = self.n_particles
        m = len(observations["obs"])
        h = self.horizon

        context_output, _ = self.dynamics_model.get_context(observations)
        context = context_output["context"]

        action = torch.reshape(self.get_random_action(m*n*h), [m, n, h, -1])
        returns = 0

        for t in range(h):
            if t == 0:
                cand_action = action[:, :, 0]
                observation = torch.tile(torch.reshape(observations["obs"], [m, 1, 1, -1]), [1, n, p, 1])
                context = torch.permute(context, (1, 0, 2))
                context = torch.tile(torch.reshape(context, [m, 1, ensemble_size, -1]), [1, n, int(p/ensemble_size), 1])
                reshaped_context = torch.reshape(torch.permute(context, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])
            
            reshaped_observation = torch.reshape(torch.permute(observation, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])

            reshaped_action = action[:, :, t]
            reshaped_action = torch.tile(reshaped_action[:, :, None, :], [1, 1, p, 1])
            reshaped_action = torch.reshape(torch.permute(reshaped_action, [2, 0, 1, 3]), [ensemble_size, int(p/ensemble_size)*m*n, -1])

            x = {"obs": reshaped_observation, "action": reshaped_action}

            delta = self.dynamics_model.predict(x, reshaped_context)
            delta = torch.reshape(delta, [p, m, n, -1])
            delta = torch.permute(delta, [1, 2, 0, 3])

            obs_mean = torch.as_tensor(self.envs.obs_rms["obs"].mean[None, None, None, :], device=self.device)
            obs_var = torch.as_tensor(self.envs.obs_rms["obs"].var[None, None, None, :], device=self.device)
            delta_mean = torch.as_tensor(self.envs.obs_rms["future_obs_delta"].mean[-1][None, None, None, :], device=self.device)
            delta_var = torch.as_tensor(self.envs.obs_rms["future_obs_delta"].var[-1][None, None, None, :], device=self.device)
            denormalized_obs = observation * torch.sqrt(obs_var + self.envs.epsilon) + obs_mean
            denormalized_delta = delta * torch.sqrt(delta_var + self.envs.epsilon) + delta_mean

            denormalized_next_obs = self.dummy_env.obs_postproc(denormalized_obs, denormalized_delta)
            repeated_action = torch.tile(action[:, :, t][:, :, None, :], [1, 1, p, 1])

            if self.use_reward_model:
                assert self.reward_model is not None
                reward = self.reward_model.predict(observation, repeated_action, denormalized_next_obs)
            else:
                reward = self.dummy_env.reward(observation, repeated_action, denormalized_next_obs)
            returns += self.discount ** t * reward
            next_observation = (denormalized_next_obs - obs_mean) / torch.sqrt(obs_var + self.envs.epsilon)
            observation = next_observation
        returns = returns.mean(-1)
        max_return_idxs = torch.argmax(returns, dim=1)
        max_return_idxs = max_return_idxs + torch.arange(0, m*n, n, device=self.device)
        flat_cand_action = torch.reshape(cand_action, [m*n, -1])
        optimal_action = flat_cand_action[max_return_idxs]
        return optimal_action