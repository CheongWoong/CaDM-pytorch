from collections import defaultdict
import numpy as np

import torch


class Sampler():
    def __init__(self, args, envs, policy, writer):
        self.args = args
        self.device = args.device

        self.envs = envs
        self.dummy_env = envs.envs[0].unwrapped
        self.policy = policy
        self.writer = writer

        self.buffer = {
            "rewards": torch.zeros((args.num_envs, args.num_steps), dtype=torch.float32).to(self.device),
            "history_obs": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "history_obs_delta": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "history_act": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.action_dim), dtype=torch.float32).to(self.device),
            "future_obs": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "future_obs_delta": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "future_act": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.action_dim), dtype=torch.float32).to(self.device),
            "future_mask": torch.zeros((args.num_envs, args.num_steps, args.future_length, 1), dtype=torch.float32).to(self.device),
            "context": torch.zeros((args.num_envs, args.num_steps, args.context_dim), dtype=torch.float32).to(self.device),
        }
        self.start_indices = np.zeros((args.num_envs), dtype=np.int32)
        
        self.next_obs, _ = self.envs.reset(seed=args.seed)
        for key in self.next_obs:
            self.next_obs[key] = torch.Tensor(self.next_obs[key]).to(self.device)
        self.next_done = torch.zeros(args.num_envs, dtype=torch.float).to(self.device)

    def sample(self):
        self.reset()
        args = self.args

        paths = []

        for step in range(0, args.num_steps):
            args.global_step += 1 * args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = self.policy.get_action(self.next_obs)

            for key in self.next_obs:
                if key in self.buffer:
                    self.buffer[key][:,step] = self.next_obs[key]

            done_indices = self.next_done.nonzero()
            for done_idx in enumerate(done_indices):
                idx = done_idx[0]
                paths.append({key: self.buffer[key][idx,self.start_indices[idx]:step] for key in self.buffer})
                self.start_indices[idx] = step

            # TRY NOT TO MODIFY: execute the game and log data.
            self.next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            reward = torch.tensor(reward).to(self.device).view(-1)
            self.buffer["rewards"][:,step] = reward
            for key in self.next_obs:
                self.next_obs[key] = torch.Tensor(self.next_obs[key]).to(self.device)
            self.next_done = torch.Tensor(done).to(self.device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            self.policy.reset(done)

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                print(f"global_step={args.global_step}, episodic_return={info['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], args.global_step)
                self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], args.global_step)

        for idx in range(args.num_envs):
            if self.start_indices[idx] != step:
                paths.append({key: self.buffer[key][idx,self.start_indices[idx]:step] for key in self.buffer})
            self.start_indices[idx] = step

        paths = self.preprocess(paths)
        return paths

    def preprocess(self, paths):
        dataset = defaultdict(list)
        for path in paths:
            if self.args.shuffle_future is False:
                if len(path["rewards"]) <= self.args.history_length:
                    continue
                for key in path:
                    if "history" in key:
                        dataset[key].append(path[key][:-self.args.history_length])
                    else:
                        dataset[key].append(path[key][self.args.history_length:])
            else:
                random_idx = np.random.permutation(len(path["rewards"]))
                for key in path:
                    if "history" in key:
                        dataset[key].append(path[key])
                    else:
                        dataset[key].append(path[key][random_idx])
        for key in dataset:
            dataset[key] = torch.cat(dataset[key], dim=0)
        return dataset

    def reset(self):
        args = self.args

        self.buffer = {
            "rewards": torch.zeros((args.num_envs, args.num_steps), dtype=torch.float32).to(self.device),
            "history_obs": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "history_obs_delta": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "history_act": torch.zeros((args.num_envs, args.num_steps, args.history_length, args.action_dim), dtype=torch.float32).to(self.device),
            "future_obs": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "future_obs_delta": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.obs_dim), dtype=torch.float32).to(self.device),
            "future_act": torch.zeros((args.num_envs, args.num_steps, args.future_length, args.action_dim), dtype=torch.float32).to(self.device),
            "future_mask": torch.zeros((args.num_envs, args.num_steps, args.future_length, 1), dtype=torch.float32).to(self.device),
            "context": torch.zeros((args.num_envs, args.num_steps, args.context_dim), dtype=torch.float32).to(self.device),
        }
        self.start_indices = np.zeros((args.num_envs), dtype=np.int32)