import torch
import torch.nn as nn

from ..layers import create_fc_layers

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SingleHeadDecoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()

        self.args = args

        fc_hidden_sizes = (args.obs_dim + args.action_dim + args.context_hidden_dim,) + (args.hidden_size,) * config.num_layers
        self.fc_forward = create_fc_layers(
            args.ensemble_size,
            fc_hidden_sizes,
            config.activation,
        )
        self.fc_backward = create_fc_layers(
            args.ensemble_size,
            fc_hidden_sizes,
            config.activation,
        )

        self.fc_forward_mu_logstd = create_fc_layers(
            args.ensemble_size,
            (args.hidden_size, args.obs_dim*2),
            "none"
        )
        self.fc_backward_mu_logstd = create_fc_layers(
            args.ensemble_size,
            (args.hidden_size, args.obs_dim*2),
            "none"
        )

        self.nll = nn.GaussianNLLLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.l1_loss = nn.L1Loss()

    def forward(self, x, context):
        if self.training:
            forward_obs = x["normalized_proc_future_obs"][:,:-1]
            backward_obs = x["normalized_proc_future_obs"][:,1:]
            action = x["normalized_future_act"][:,:-1]

            forward_target = x["normalized_future_obs_delta"][:,:-1]
            backward_target = x["normalized_future_obs_back_delta"][:,:-1]
            target_mask = x["future_mask"][:,:-1]

            forward_obs = torch.tile(forward_obs[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            backward_obs = torch.tile(backward_obs[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            action = torch.tile(action[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            forward_target = torch.tile(forward_target[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            backward_target = torch.tile(backward_target[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            target_mask = torch.tile(target_mask[None,:,:,:], [self.args.ensemble_size, 1, 1, 1])
            if context is not None:
                context = torch.tile(context[:,:,None,:], [1, 1, self.args.future_length - 1, 1])
                forward_input = torch.cat([forward_obs, action, context], dim=-1)
                backward_input = torch.cat([backward_obs, action, context], dim=-1)
            else:
                forward_input = torch.cat([forward_obs, action], dim=-1)
                backward_input = torch.cat([backward_obs, action], dim=-1)

            forward_input = forward_input.flatten(-3, -2)
            backward_input = backward_input.flatten(-3, -2)
            forward_target = forward_target.flatten(-3, -2)
            backward_target = backward_target.flatten(-3, -2)
            target_mask = target_mask.flatten(-3, -2)

            forward_output = self.fc_forward(forward_input)
            forward_mu_logstd = self.fc_forward_mu_logstd(forward_output)
            forward_mu = forward_mu_logstd[...,:self.args.obs_dim]
            if self.args.deterministic:
                # RMSE
                forward_loss = self.mse(forward_mu, forward_target)
                forward_loss = torch.where(target_mask > 0, forward_loss, 0.).mean()
                forward_loss = torch.sqrt(forward_loss)
            else:
                # Gaussian NLL
                forward_logstd = torch.tanh(forward_mu_logstd[...,self.args.obs_dim:])
                forward_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (forward_logstd + 1)
                forward_std = torch.exp(forward_logstd)
                forward_loss = self.nll(forward_mu, forward_target, torch.square(forward_std))
                forward_loss = torch.where(target_mask > 0, forward_loss, 0.).mean()

            backward_output = self.fc_backward(backward_input)
            backward_mu_logstd = self.fc_backward_mu_logstd(backward_output)
            backward_mu = backward_mu_logstd[...,:self.args.obs_dim]
            if self.args.deterministic:
                # RMSE
                backward_loss = self.mse(backward_mu, backward_target)
                backward_loss = torch.where(target_mask > 0, backward_loss, 0.).mean()
                backward_loss = torch.sqrt(backward_loss)
            else:
                # Gaussian NLL
                backward_logstd = torch.tanh(backward_mu_logstd[...,self.args.obs_dim:])
                backward_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (backward_logstd + 1)
                backward_std = torch.exp(backward_logstd)
                backward_loss = self.nll(backward_mu, backward_target, torch.square(backward_std))
                backward_loss = torch.where(target_mask > 0, backward_loss, 0.).mean()

            loss = forward_loss + self.args.back_coeff*backward_loss
            prediction_error = self.l1_loss(forward_mu, forward_target)

            output = {
                "forward_prediction": forward_mu,
                "backward_prediction": backward_mu,
                "forward_loss": forward_loss.item(),
                "backward_loss": backward_loss.item(),
                "prediction_loss": prediction_error.item(),
            }
        else:
            obs, action = x["normalized_proc_obs"], x["normalized_action"]

            if context is not None:
                forward_input = torch.cat([obs, action, context], dim=-1)
            else:
                forward_input = torch.cat([obs, action], dim=-1)

            forward_output = self.fc_forward(forward_input)
            forward_mu_logstd = self.fc_forward_mu_logstd(forward_output)
            forward_mu = forward_mu_logstd[...,:self.args.obs_dim]
            if self.args.deterministic:
                forward_prediction = forward_mu
            else:
                forward_logstd = torch.tanh(forward_mu_logstd[...,self.args.obs_dim:])
                forward_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (forward_logstd + 1)
                forward_std = torch.exp(forward_logstd)
                forward_prediction = torch.distributions.normal.Normal(forward_mu, forward_std).sample()
            output = {"forward_prediction": forward_prediction}
            loss = 0.0

        return output, loss