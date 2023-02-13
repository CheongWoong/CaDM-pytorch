import torch 
import torch.nn as nn

from ..layers import create_fc_layers


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()