from distutils.util import strtobool

import argparse
import os


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="cartpole",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--minibatch-size", type=int, default=256,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5,
        help="the K epochs to update the model")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")

    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--dynamics-model-type", type=str)
    parser.add_argument("--randomization-type", type=str, default="")

    args = parser.parse_args()
    return args


def test_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--test-record-video-step-frequency", type=int, default=1464,
        help="the frequency at which to record the videos")

    parser.add_argument("--checkpoint-path", type=str)

    parser.add_argument("--test-seed", type=int, default=12345,
        help="seed of the experiment")
    parser.add_argument("--test-config-path", type=str, default="")
    parser.add_argument("--test-num-envs", type=int, default=4096)
    parser.add_argument("--test-num-episodes", type=int, default=4096)
    parser.add_argument("--test-gui", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--test-randomization-type", type=str, default="")
    parser.add_argument("--test-env-id", type=str, default="",
        help="the id of the environment")

    args = parser.parse_args()
    return args