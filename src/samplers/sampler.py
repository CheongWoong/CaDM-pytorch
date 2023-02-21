from collections import defaultdict
import time

import numpy as np
import torch


class Sampler():
    def __init__(self, args, envs, policy):
        self.args = args
        self.device = args.device

        self.envs = envs
        self.policy = policy

        self.total_samples = args.num_rollouts * args.max_path_length

    def obtain_samples(self, random=False):
        args = self.args

        paths = []
        episodic_returns = []
        episodic_lengths = []

        n_samples = 0
        self.reset_running_paths()

        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset()

        obses, _ = self.envs.reset(seed=args.seed)
        dones = np.zeros(args.num_rollouts, dtype=np.float32)

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            if random:
                actions = self.envs.action_space.sample()
            else:
                with torch.no_grad():
                    actions = self.policy.get_action(obses).cpu().numpy()
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            env_time += time.time() - t

            new_samples = 0

            dones = np.logical_or(terminateds, truncateds)

            # append new samples to running paths
            for key in obses:
                if key in self.running_paths:
                    self.running_paths[key][:, self.running_path_idx] = obses[key]
            self.running_paths["rewards"][:, self.running_path_idx] = rewards
            self.running_path_idx += 1

            # Only print when at least 1 env is done
            if "final_info" in infos:
                # if running path is done, add it to paths and empty the running path
                done_indices = dones.nonzero()[0]
                for done_idx in done_indices:
                    # for key in self.running_paths:
                    #     if key not in ["rewards"]:
                    #         self.running_paths[key][done_idx, self.running_path_idx[done_idx]:] = self.running_paths[key][done_idx, self.running_path_idx[done_idx] - 1:self.running_path_idx[done_idx]]
                    # paths.append({key: self.running_paths[key][done_idx] for key in self.running_paths})
                    paths.append({key: self.running_paths[key][done_idx, :self.running_path_idx[done_idx]].copy() for key in self.running_paths})
                    
                    for key in self.running_paths:
                        self.running_paths[key][done_idx] *= 0
                    new_samples += self.running_path_idx[done_idx]
                    self.running_path_idx[done_idx] *= 0
                
                self.policy.reset(dones)
                n_samples += new_samples

                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if info is None:
                        continue
                    episodic_returns.append(info["episode"]["r"])
                    episodic_lengths.append(info["episode"]["l"])

            obses = next_obses

        print("policy time:", policy_time)
        print("env time", env_time)

        logger_dict = {}
        logger_dict["charts/episodic_return"] = np.mean(episodic_returns)
        logger_dict["charts/episodic_length"] = np.mean(episodic_lengths)

        return paths, logger_dict

    def process_samples(self, paths):
        t = time.time()
        samples_data = defaultdict(list)
        for path in paths:
            if self.args.shuffle_future is False:
                if len(path["rewards"]) <= self.args.history_length:
                    continue
                for key in path:
                    if "history" in key:
                        samples_data[key].append(path[key][:-self.args.history_length])
                    else:
                        samples_data[key].append(path[key][self.args.history_length:])
            else:
                random_idx = np.random.permutation(len(path["rewards"]))
                for key in path:
                    if "history" in key:
                        samples_data[key].append(path[key])
                    else:
                        samples_data[key].append(path[key][random_idx])
        for key in samples_data:
            samples_data[key] = np.concatenate(samples_data[key], axis=0)
        print("sample processing time:", time.time() - t)
        return samples_data

    def reset_running_paths(self):
        args = self.args

        self.running_paths = {
            "rewards": np.zeros((args.num_rollouts, args.max_path_length), dtype=np.float32),
            "history_cp_obs": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, args.obs_dim), dtype=np.float32),
            "history_obs": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, args.obs_dim), dtype=np.float32),
            "history_obs_delta": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, args.obs_dim), dtype=np.float32),
            "history_obs_back_delta": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, args.obs_dim), dtype=np.float32),
            "history_act": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, args.action_dim), dtype=np.float32),
            "history_mask": np.zeros((args.num_rollouts, args.max_path_length, args.history_length, 1), dtype=np.float32),
            "future_obs": np.zeros((args.num_rollouts, args.max_path_length, args.future_length, args.obs_dim), dtype=np.float32),
            "future_obs_delta": np.zeros((args.num_rollouts, args.max_path_length, args.future_length, args.obs_dim), dtype=np.float32),
            "future_obs_back_delta": np.zeros((args.num_rollouts, args.max_path_length, args.future_length, args.obs_dim), dtype=np.float32),
            "future_act": np.zeros((args.num_rollouts, args.max_path_length, args.future_length, args.action_dim), dtype=np.float32),
            "future_mask": np.zeros((args.num_rollouts, args.max_path_length, args.future_length, 1), dtype=np.float32),
            "sim_params": np.zeros((args.num_rollouts, args.max_path_length, args.sim_param_dim), dtype=np.float32),
        }
        self.running_path_idx = np.zeros((args.num_rollouts), dtype=np.int32)