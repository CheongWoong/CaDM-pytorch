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

    # Arguments
    parser.add_argument("--env-id", type=str, default="cartpole",
        help="the id of the environment")
    # DM (Dynamics Model) configs
    parser.add_argument("--history-length", type=int, default=10,
        help="the length of history")
    parser.add_argument("--use-obs-delta", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to use obs difference or not")
    parser.add_argument("--future-length", type=int, default=11,
        help="the length of future")
    parser.add_argument("--shuffle-future", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to sample random future transitions instead of consecutive ones")
    parser.add_argument("--back-coeff", type=float, default=0.5,
        help="the coefficient for backward dynamics loss")
    parser.add_argument("--hidden-dim", type=int, default=200,
        help="the size of hidden dimension")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to use deterministic neural networks or not")
    parser.add_argument("--ensemble-size", type=int, default=5,
        help="the number of ensembles (PE-TS)")
    parser.add_argument("--context-hidden-dim", type=int, default=64,
        help="the size of context hidden dimension")
    # Training configs
    parser.add_argument("--n-itr", type=int, default=20,
        help="the number of iterations")
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--minibatch-size", type=int, default=256,
        help="the number of mini-batches")
    parser.add_argument("--learning-rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--update-epochs", type=int, default=5,
        help="the K epochs to update the model")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    # MPC (Model Predictive Control) configs
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--n-candidates", type=int, default=200,
        help="the number of candidates (MPC)")
    parser.add_argument("--horizon", type=int, default=30,
        help="the horizon size (MPC)")
    parser.add_argument("--use-cem", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to use CEM or not")
    parser.add_argument("--num-cem-iters", type=int, default=5,
        help="the number of CEM iterations")
    parser.add_argument("--percent-elites", type=float, default=0.25,
        help="percent of elites (CEM)")
    parser.add_argument("--alpha", type=float, default=0.1,
        help="the alpha value for CEM")
    parser.add_argument("--n-particles", type=int, default=20,
        help="the number of particles (PE-TS)")
    # other configs
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--dynamics-model-type", type=str)
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--randomization-id", type=str, default="train_0")

    args = parser.parse_args()
    return args


def test_parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--randomization-id", type=str, default="test_0")

    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--gui", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    args = parser.parse_args()
    return args