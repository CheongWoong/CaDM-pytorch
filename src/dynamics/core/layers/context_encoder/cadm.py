import torch
import torch.nn as nn

from ..utils import create_fc_layers


class CaDM(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.args, self.config = args, config

        fc_hidden_sizes = (config.context_in_dim, args.context_out_dim)
        self.fc = create_fc_layers(
            args.ensemble_size,
            fc_hidden_sizes,
            config.activation,
        )

    def forward(self, x, context):
        context = self.fc(context)

        context_output = {"context": context}
        context_loss = 0.0

        return context_output, context_loss