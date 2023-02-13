# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
from .utils.envs import make_env
from .utils.arguments import test_parse_args
from .models.policies.ppo import Agent

from copy import deepcopy
from omegaconf import OmegaConf
import json

from tqdm import tqdm
import time
import random
import os

import numpy as np
import torch


def evaluate(args, checkpoint):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    args.device = device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env setup
    system_params = []
    context_samples = []
    if "Contextual" in args.env_id:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        randomization_config = OmegaConf.load(os.path.join(dir_path, "../configs/randomization", args.env_id+".yaml"))
        randomization_config = getattr(randomization_config, args.test_randomization_type)
        context_samples = eval(randomization_config.context_samples)
        system_params = randomization_config.system_params if context_samples else dict(randomization_config.system_params)
    envs = make_env(args.env_id, args.test_seed, args.num_envs, args.capture_video, args.record_video_step_frequency, args.run_name, args.len_history, device, system_params, context_samples, gui=args.test_gui)

    args.obs_dim = np.array(envs.single_observation_space["obs"].shape).prod()
    args.action_dim = np.prod(envs.single_action_space.shape)
    if "Contextual" in args.env_id:
        args.context_dim = np.array(envs.single_observation_space["context"].shape).prod()

    agent = Agent(args).to(device)
    agent.load(checkpoint)
    agent.eval()

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)

    episodic_returns = []
    episodic_lengths = []
    consecutive_successes = []

    pbar = tqdm()
    while True:
        # ALGO LOGIC: action logic
        with torch.no_grad():
            context, context_info = agent.get_context({"obs": next_obs, "training": False})
            action, logprob, _, value = agent.get_action_and_value(next_obs, context=context)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, info = envs.step(action)
        envs.render()
        
        for idx, d in enumerate(next_done):
            if d and (len(episodic_returns) < args.test_num_episodes):
                episodic_return = info["r"][idx].item()
                episodic_length = info["l"][idx].item()
                episodic_returns.append(episodic_return)
                episodic_lengths.append(episodic_length)
                
                if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                    consecutive_success = info["consecutive_successes"].item()
                    consecutive_successes.append(consecutive_success)

        pbar.update(1)

        if len(episodic_returns) == args.test_num_episodes:
            break

    # envs.close()
    pbar.close()

    result = {}
    result['num_episodes'] = len(episodic_returns)
    result['mean_episodic_returns'] = np.mean(episodic_returns)
    result['std_episodic_returns'] = np.std(episodic_returns)
    result['mean_episodic_lengths'] = np.mean(episodic_lengths)
    result['std_episodic_lengths'] = np.std(episodic_lengths)
    if consecutive_successes:
        result['mean_consecutive_successes'] = np.mean(consecutive_successes)
        result['std_consecutive_successes'] = np.std(consecutive_successes)

    return result


if __name__ == "__main__":
    args = test_parse_args()

    if os.path.isdir(args.checkpoint_path):
        args.run_name = args.checkpoint_path
        checkpoints = sorted([os.path.join(args.run_name, filename) for filename in os.listdir(args.run_name) if filename.endswith(".pt")])
        raise NotImplementedError("Issues with multiple runs of Isaac Gym")
    else:
        args.run_name = os.path.dirname(args.checkpoint_path)
        checkpoints = [args.checkpoint_path]
    with open(os.path.join(args.run_name, 'training_args.json'), "r") as fin:
        training_args = json.load(fin)
    training_config = OmegaConf.load(os.path.join(args.run_name, 'training_config.yaml'))
    if args.test_config_path:
        override_config = OmegaConf.load(args.config_path)
        training_config = OmegaConf.merge(training_config, override_config)
    vars(args).update(training_args)
    vars(args).update(training_config)
    if args.test_env_id:
        args.env_id = args.test_env_id

    results = {}
    for checkpoint in checkpoints:
        result = evaluate(args, checkpoint)
        results[checkpoint] = result

    print("\nEvaluation done")
    print("=============================================")
    for checkpoint_name, result in results.items():
        print(checkpoint_name)
        for key, value in result.items():
            print("\t"+key+":", value)
        print()