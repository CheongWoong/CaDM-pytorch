import time

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

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, eps=1e-5)

        self._dataset = None

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
        output, _ = self.decoder(x, context) ##### decoder side에서 denormalization 추가
        return output["forward_prediction"]

    def fit(self, samples):
        t = time.time()
        self.train()
        args = self.args

        if self._dataset is None:
            self._dataset = {}
            for key in samples:
                self._dataset[key] = samples[key]
        else:
            for key in samples:
                self._dataset[key] = torch.cat([self._dataset[key], samples[key]], dim=0)

        self.compute_normalization()
        self._dataset = self.normalize(self._dataset)

        batch_size = len(self._dataset["rewards"])
        b_inds = np.arange(batch_size)
        # Optimizing the policy and value network
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                x = {key: self._dataset[key][mb_inds] for key in self._dataset}
                
                output, loss = self.forward(x)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), args.max_grad_norm)
                self.optimizer.step()

        logger_dict = {}
        for key in output:
            if "loss" in key:
                logger_dict["losses/" + key] = output[key]
        print("training time:", time.time() - t)
        return logger_dict

    def compute_normalization(self):
        proc_obs = self.args.obs_preproc(self._dataset["future_obs"])

        self.normalization = {}
        self.normalization["obs"] = (
            torch.mean(proc_obs, dim=(0, 1)),
            torch.std(proc_obs, dim=(0, 1)),
        )
        self.normalization["delta"] = (
            torch.mean(self._dataset["future_obs_delta"], dim=(0, 1)),
            torch.std(self._dataset["future_obs_delta"], dim=(0, 1)),
        )
        self.normalization["act"] = (
            torch.mean(self._dataset["future_act"], dim=(0, 1)),
            torch.std(self._dataset["future_act"], dim=(0, 1)),
        )
        self.normalization["cp_obs"] = (
            torch.mean(self._dataset["history_cp_obs"], dim=(0, 1)),
            torch.std(self._dataset["history_cp_obs"], dim=(0, 1)),
        )
        self.normalization["cp_act"] = (
            torch.mean(self._dataset["history_act"], dim=(0, 1)),
            torch.std(self._dataset["history_act"], dim=(0, 1)),
        )
        self.normalization["back_delta"] = (
            torch.mean(self._dataset["future_obs_back_delta"], dim=(0, 1)),
            torch.std(self._dataset["future_obs_back_delta"], dim=(0, 1)),
        )
        self.normalization["sim_params"] = (
            torch.mean(self._dataset["sim_params"], dim=(0)),
            torch.std(self._dataset["sim_params"], dim=(0)),
        )

    def normalize(self, data):
        keys = list(data.keys())
        for key in keys:
            if "back_delta" in key:
                data["normalized_" + key] = (data[key] - self.normalization["back_delta"][0]) / (self.normalization["back_delta"][1] + 1e-10)
            elif "delta" in key:
                data["normalized_" + key] = (data[key] - self.normalization["delta"][0]) / (self.normalization["delta"][1] + 1e-10)
            elif "cp_obs" in key:
                data["normalized_" + key] = (data[key] - self.normalization["cp_obs"][0]) / (self.normalization["cp_obs"][1] + 1e-10)
            elif "obs" in key:
                proc_key = "proc_" + key
                data["normalized_" + proc_key] = (self.args.obs_preproc(data[key]) - self.normalization["obs"][0]) / (self.normalization["obs"][1] + 1e-10)
            elif "act" in key:
                data["normalized_" + key] = (data[key] - self.normalization["act"][0]) / (self.normalization["act"][1] + 1e-10)
            elif "sim_params" in key:
                data["normalized_" + key] = (data[key] - self.normalization["sim_params"][0]) / (self.normalization["sim_params"][1] + 1e-10)
            else:
                pass

    def save(self, path):
        checkpoint = {
            "model": self.state_dict(),
            "normalization": self.normalization
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        self.normalization = checkpoint["normalization"]

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