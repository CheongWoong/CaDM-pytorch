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


from copy import deepcopy
from omegaconf import OmegaConf
import json
import datetime
import time
import random
import os

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .envs import make_env
from .utils.arguments import test_parse_args
from .dynamics.dynamics_model import DynamicsModel
from .policies.mpc_controller import MPCController
from .samplers.sampler import Sampler


def evaluate(args, checkpoint, checkpoint_idx, writer):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    args.device = device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env setup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    env_config = OmegaConf.load(os.path.join(dir_path, "../configs/env_configs", args.dataset+".yaml"))
    env_config = getattr(env_config, args.randomization_id)
    env_config = OmegaConf.to_object(env_config)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.dataset, i, args.capture_video, args.run_name + "_test", args.history_length, args.future_length, args.state_diff, env_config, args.gui) for i in range(args.num_rollouts)]
    )

    args.obs_dim = np.prod(envs.single_observation_space["obs"].shape)
    args.action_dim = np.prod(envs.single_action_space.shape)
    args.sim_param_dim = envs.envs[0].num_modifiable_parameters
    args.proc_obs_dim = envs.envs[0].proc_obs_dim
    args.obs_preproc = envs.envs[0].obs_preproc
    args.obs_postproc = envs.envs[0].obs_postproc
    args.targ_proc = envs.envs[0].targ_proc

    dynamics_model = DynamicsModel(args).to(device)
    dynamics_model.load(checkpoint)
    dynamics_model.eval()

    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)

    policy = MPCController(args, envs, dynamics_model)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    for key in next_obs:
        next_obs[key] = torch.Tensor(next_obs[key]).to(device)

    episodic_returns = []
    episodic_lengths = []

    while True:
        global_step += 1 * args.num_rollouts

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action = policy.get_action(next_obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        for key in next_obs:
            next_obs[key] = torch.Tensor(next_obs[key]).to(device)
        done = np.logical_or(terminated, truncated)
        if args.gui:
            envs.envs[0].render()
        
        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        policy.reset(done)

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            print(f"checkpoint={checkpoint_idx}, global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")

            if len(episodic_returns) < args.num_test:
                episodic_returns.append(info['episode']['r'])
                episodic_lengths.append(info['episode']['l'])

        if len(episodic_returns) >= args.num_test:
            break

    writer.add_scalar("test("+args.randomization_id+")/mean_episodic_returns", np.mean(episodic_returns), checkpoint_idx)
    writer.add_scalar("test("+args.randomization_id+")/std_episodic_returns", np.std(episodic_returns), checkpoint_idx)
    writer.add_scalar("test("+args.randomization_id+")/mean_episodic_lengths", np.mean(episodic_lengths), checkpoint_idx)
    writer.add_scalar("test("+args.randomization_id+")/std_episodic_lengths", np.std(episodic_lengths), checkpoint_idx)

    envs.close()


if __name__ == "__main__":
    test_args = test_parse_args()
    test_args_for_save = deepcopy(test_args)

    if os.path.isdir(test_args.checkpoint_path):
        checkpoint_dir = test_args.checkpoint_path
        checkpoints = sorted([os.path.join(checkpoint_dir, filename) for filename in os.listdir(checkpoint_dir) if filename.endswith(".pt")])
    else:
        checkpoint_dir = os.path.dirname(test_args.checkpoint_path)
        checkpoints = [test_args.checkpoint_path]

    with open(os.path.join(checkpoint_dir, 'training_args.json'), "r") as fin:
        training_args = json.load(fin)
    test_config = OmegaConf.load(os.path.join(checkpoint_dir, 'training_config.yaml'))
    if test_args.config_path:
        override_config = OmegaConf.load(test_args.config_path)
        test_config = OmegaConf.merge(test_config, override_config)
    vars(test_args).update(training_args)
    vars(test_args).update(test_config)
    vars(test_args).update(vars(test_args_for_save))

    checkpoint_dir = checkpoint_dir.split("/")
    start_idx = checkpoint_dir.index("runs") + 1
    checkpoint_dir = "/".join(checkpoint_dir[start_idx:])
    test_args.run_name = checkpoint_dir

    writer = SummaryWriter(f"runs/{test_args.run_name}")
    for checkpoint in enumerate(checkpoints):
        checkpoint_idx = int(checkpoint[:-3].split("_")[-1])
        print('[INFO] evaluation starts: checkpoint', checkpoint_idx)
        evaluate(test_args, checkpoint, checkpoint_idx, writer)
    writer.close()

    #### Save test args
    with open(os.path.join('runs', test_args.run_name, 'test_args.json'), 'w') as fout:
        json.dump(vars(test_args_for_save), fout, indent=2)
    with open(os.path.join('runs', test_args.run_name, 'test_config.yaml'), 'w') as fout:
        OmegaConf.save(config=test_config, f=fout)
    print('[INFO] evaluation is done')