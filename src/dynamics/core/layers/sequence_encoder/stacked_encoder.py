import torch
import torch.nn as nn

from ..utils import create_fc_layers


class StackedEncoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.args, self.config = args, config

        fc_hidden_sizes = ((args.obs_dim + args.action_dim)*args.history_length,) + eval(config.fc_hidden_sizes)
        self.fc = create_fc_layers(
            args.ensemble_size,
            fc_hidden_sizes,
            config.activation,
        )

    def forward(self, x):
        h_o = x["normalized_history_cp_obs"]
        h_a = x["normalized_history_act"]
        x = torch.cat([h_o, h_a], dim=-1)

        ensemble_size, batch_size = x.shape[:2]
        x = torch.reshape(x, (ensemble_size, batch_size, -1))

        x = self.fc(x)

        return x