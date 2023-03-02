import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .core.layers.utils import normalize, denormalize
from .core.layers.sequence_encoder import StackedEncoder, RecurrentEncoder
from .core.layers.context_encoder import CaDM
from .core.layers.decoder import Decoder


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
            self.context_encoder_config.context_in_dim = eval(self.sequence_encoder_config.fc_hidden_sizes)[-1]
        self.context_encoder = self.get_context_encoder(config.context_encoder_type)

        self.decoder_config = getattr(args.decoder, config.decoder_type)
        self.decoder = Decoder(self.args, self.decoder_config)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, eps=1e-5)

        self._dataset = None

    def forward(self, x):
        context = None
        if self.sequence_encoder is not None:
            context = self.sequence_encoder(x)
        if self.context_encoder is not None:
            context_output, context_loss = self.context_encoder(x, context)
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
            context_output, context_loss = self.context_encoder(x, context)
        else:
            context_output = {"context": context}
            context_loss = 0.0
        return context_output, context_loss

    def predict(self, x, context):
        self.eval()
        output, _ = self.decoder(x, context)
        mu, logvar = output
        denormalized_mu = denormalize(mu, self.torch_normalization["delta"][0], self.torch_normalization["delta"][1])
        if self.args.deterministic:
            output = denormalized_mu
        else:
            denormalized_logvar = logvar + 2*torch.log(torch.as_tensor(self.normalization["delta"][1], device=self.args.device))
            denormalized_std = torch.exp(denormalized_logvar / 2.0)
            output = denormalized_mu + torch.randn_like(denormalized_mu)*denormalized_std
        return output

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
                self._dataset[key] = np.concatenate([self._dataset[key], samples[key]], axis=0)

        self.compute_normalization()
        normalized_dataset = self.apply_normalization(self._dataset)

        batch_size = len(normalized_dataset["rewards"])
        b_inds = np.arange(batch_size)
        # Optimizing the policy and value network
        for epoch in range(args.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                x = {key: torch.as_tensor(normalized_dataset[key][mb_inds], device=args.device).repeat([args.ensemble_size]+[1]*len(normalized_dataset[key][mb_inds].shape)) for key in normalized_dataset}
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
            np.mean(proc_obs, axis=(0, 1)),
            np.std(proc_obs, axis=(0, 1)),
        )
        self.normalization["delta"] = (
            np.mean(self._dataset["future_obs_delta"], axis=(0, 1)),
            np.std(self._dataset["future_obs_delta"], axis=(0, 1)),
        )
        self.normalization["act"] = (
            np.mean(self._dataset["future_act"], axis=(0, 1)),
            np.std(self._dataset["future_act"], axis=(0, 1)),
        )
        self.normalization["cp_obs"] = (
            np.mean(self._dataset["history_cp_obs"], axis=(0, 1)),
            np.std(self._dataset["history_cp_obs"], axis=(0, 1)),
        )
        self.normalization["cp_act"] = (
            np.mean(self._dataset["history_act"], axis=(0, 1)),
            np.std(self._dataset["history_act"], axis=(0, 1)),
        )
        self.normalization["back_delta"] = (
            np.mean(self._dataset["future_obs_back_delta"], axis=(0, 1)),
            np.std(self._dataset["future_obs_back_delta"], axis=(0, 1)),
        )
        self.normalization["sim_params"] = (
            np.mean(self._dataset["sim_params"], axis=(0)),
            np.std(self._dataset["sim_params"], axis=(0)),
        )
        self.torch_normalization = {}
        for key in self.normalization:
            self.torch_normalization[key] = torch.as_tensor(np.array(self.normalization[key]), device=self.args.device)        

    def apply_normalization(self, data):
        normalized_dataset = {}
        for key, value in data.items():
            if "back_delta" in key:
                normalized_dataset["normalized_" + key] = normalize(value, self.normalization["back_delta"][0], self.normalization["back_delta"][1])
            elif "delta" in key:
                normalized_dataset["normalized_" + key] = normalize(value, self.normalization["delta"][0], self.normalization["delta"][1])
            elif "cp_obs" in key:
                normalized_dataset["normalized_" + key] = normalize(value, self.normalization["cp_obs"][0], self.normalization["cp_obs"][1])
            elif "obs" in key:
                normalized_dataset["normalized_proc_" + key] = normalize(self.args.obs_preproc(value), self.normalization["obs"][0], self.normalization["obs"][1])
            elif "act" in key:
                normalized_dataset["normalized_" + key] = normalize(value, self.normalization["act"][0], self.normalization["act"][1])
            elif "sim_params" in key:
                normalized_dataset["normalized_" + key] = normalize(value, self.normalization["sim_params"][0], self.normalization["sim_params"][1])
            else:
                pass
        normalized_dataset.update(data)
        return normalized_dataset

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
            "cadm": CaDM,
        }

        return context_encoder_map[type](self.args, self.context_encoder_config)