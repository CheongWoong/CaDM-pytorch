import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .core.sequence_encoder import RecurrentEncoder, StackedEncoder
from .core.context_encoder import Dummy
from .core.decoder import SingleHeadDecoder, MultiHeadDecoder


class DynamicsModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.config = config = getattr(args.dynamics_model, args.dynamics_model_type)

        self.sequence_encoder_config = None
        if hasattr(args.sequence_encoder, config.sequence_encoder_type):
            self.sequence_encoder_config = getattr(args.sequence_encoder, config.sequence_encoder_type)
        self.sequence_encoder = self.get_sequence_encoder(config.sequence_encoder_type)

        self.context_encoder_config = None
        if hasattr(args.context_encoder, config.context_encoder_type):
            self.context_encoder_config = getattr(args.context_encoder, config.context_encoder_type)
        self.context_encoder = self.get_context_encoder(config.context_encoder_type)

        self.decoder_config = getattr(args.decoder, config.decoder_type)
        self.decoder = self.get_decoder(config.decoder_type)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, eps=1e-5)

    def forward(self, x):
        context = None
        if self.sequence_encoder is not None:
            context = self.sequence_encoder(x)
        if self.context_encoder is not None:
            context_output, context_loss = self.context_encoder(context)
        else:
            context_output = {"context": context}
            context_loss = 0.0

        decoder_output, decoder_loss = self.decoder(x, context_output["context"])

        output = dict(context_output, **decoder_output)
        loss = context_loss + decoder_loss
        
        return output, loss

    def get_context(self, x):
        self.eval()
        context = None
        if self.sequence_encoder is not None:
            context = self.sequence_encoder(x)
        if self.context_encoder is not None:
            context_output, context_loss = self.context_encoder(context)
        else:
            context_output = {"context": context}
            context_loss = 0.0
        return context_output, context_loss

    def predict(self, x, context):
        self.eval()
        output, _ = self.decoder(x, context)
        return output["forward_prediction"]

    def learn(self, samples):
        self.train()
        args = self.args

        batch_size = len(samples["rewards"])
        b_inds = np.arange(batch_size)
        # Optimizing the policy and value network
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                x = {key: samples[key][mb_inds] for key in samples}
                
                output, loss = self.forward(x)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                self.optimizer.step()

        writer_dict = {}
        writer_dict["charts/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        for key in output:
            if "loss" in key:
                writer_dict["losses/" + key] = output[key]
        
        return writer_dict

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_sequence_encoder(self, type):
        sequence_encoder_map = {
            "none": lambda *x: None,
            "stacked": StackedEncoder,
            "rnn": RecurrentEncoder,
        }
        return sequence_encoder_map[type](self.args, self.sequence_encoder_config)

    def get_context_encoder(self, type):
        context_encoder_map = {
            "none": lambda *x: None,
            "dummy": Dummy,
        }

        return context_encoder_map[type](self.args, self.context_encoder_config)

    def get_decoder(self, type):
        decoder_map = {
            "single_head": SingleHeadDecoder,
            "multi_head": MultiHeadDecoder,
        }

        return decoder_map[type](self.args, self.decoder_config)