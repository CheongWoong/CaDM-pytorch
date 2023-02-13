import torch
import torch.nn as nn

from ..layers import create_fc_layers


class MultiHeadDecoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.args, self.config = args, config

    def forward(self, x):
        return x