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
from .envs.wrappers import NormalizeObservation
from .utils.arguments import parse_args
from .dynamics.dynamics_model import DynamicsModel
from .policies.mpc_controller import MPCController
from .samplers.sampler import Sampler


if __name__ == "__main__":
    args = parse_args()
    args_for_save = deepcopy(args)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = OmegaConf.load(os.path.join(dir_path, "../configs/base_config.yaml"))
    if args.config_path:
        override_config = OmegaConf.load(args.config_path)
        config = OmegaConf.merge(config, override_config)
    vars(args).update(config)

    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
    if args.exp_name:
        run_name = f"{args.env_id}/{args.dynamics_model_type}/{args.exp_name}"
    else:
        run_name = f"{args.env_id}/{args.dynamics_model_type}/{timestamp}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    args.device = device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    # env setup
    env_config = OmegaConf.load(os.path.join(dir_path, "../configs/env_configs", args.env_id+".yaml"))
    env_config = getattr(env_config, args.randomization_id)
    env_config = OmegaConf.to_object(env_config)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.history_length, args.future_length, env_config) for i in range(args.num_envs)]
    )
    envs = NormalizeObservation(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    
    args.num_steps = envs.envs[0].spec.max_episode_steps
    args.total_timesteps = args.n_itr * args.num_envs * args.num_steps
    args.obs_dim = np.prod(envs.single_observation_space["obs"].shape)
    args.action_dim = np.prod(envs.single_action_space.shape)
    args.context_dim = envs.envs[0].num_modifiable_parameters()

    dynamics_model = DynamicsModel(args).to(device)
    policy = MPCController(args, envs, dynamics_model)
    sampler = Sampler(args, envs, policy, writer)

    # TRY NOT TO MODIFY: start the game
    args.global_step = 0
    start_time = time.time()
    
    args.batch_size = int(args.num_envs * args.num_steps)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            dynamics_model.optimizer.param_groups[0]["lr"] = lrnow

        samples = sampler.sample()
        writer_dict = dynamics_model.learn(samples)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for key, value in writer_dict.items():
            writer.add_scalar(key, value, args.global_step)
        print("SPS:", int(args.global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(args.global_step / (time.time() - start_time)), args.global_step)

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

        #### Save periodically
        if update % (num_updates // args.save_freq) == 0 or update == num_updates:
            dynamics_model.obs_rms = envs.obs_rms
            dynamics_model.save(os.path.join('runs', run_name, f'checkpoint_{args.global_step}.pt'))
            with open(os.path.join('runs', run_name, 'training_args.json'), 'w') as fout:
                json.dump(vars(args_for_save), fout, indent=2)
            with open(os.path.join('runs', run_name, 'training_config.yaml'), 'w') as fout:
                OmegaConf.save(config=config, f=fout)
            print('[INFO] Checkpoint is saved')

    envs.close()
    writer.close()