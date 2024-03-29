import torch
import torch.nn as nn

from ..utils import create_fc_layers, EnsembleLSTMLayer


class RecurrentEncoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        
        self.args, self.config = args, config

        self.lstm = EnsembleLSTMLayer(args.ensemble_size, args.obs_dim + args.action_dim, config.rnn_hidden_dim)

        fc_hidden_sizes = (config.rnn_hidden_dim,) + eval(config.fc_hidden_sizes)
        self.fc = create_fc_layers(
            args.ensemble_size,
            fc_hidden_sizes,
            config.activation,
        )

    def forward(self, x):
        h_o = x["normalized_history_cp_obs"]
        h_a = x["normalized_history_act"]
        x = torch.cat([h_o, h_a], dim=-1)

        batch_size = x.shape[1]
        hidden_dim = self.config.rnn_hidden_dim
        h0 = torch.zeros(self.args.ensemble_size, batch_size, hidden_dim, device=x.device)
        c0 = torch.zeros(self.args.ensemble_size, batch_size, hidden_dim, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        x = self.fc(lstm_out[:,:,-1,:])

        return x