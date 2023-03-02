import math
import numpy as np 

import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.trunc_normal_(layer.weight, std=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def create_fc_layers(ensemble_size, hidden_sizes, activation):
    if activation == "relu":
        ACT = nn.ReLU
    elif activation == "tanh":
        ACT = nn.Tanh
    elif activation == "sigmoid":
        ACT = nn.Sigmoid
    elif activation == "softmax":
        ACT = nn.Softmax
    elif activation == "swish":
        ACT = nn.SiLU
    elif activation == "none":
        ACT = nn.Identity
    else:
        raise NotImplementedError()

    net = []
    for i in range(len(hidden_sizes) - 1):
        in_dim = hidden_sizes[i]
        out_dim = hidden_sizes[i + 1]
        net.append(layer_init(EnsembleLinear(ensemble_size, in_dim, out_dim), std=1 / (2 * np.sqrt(in_dim))))
        net.append(ACT())

    return nn.Sequential(*net)

class EnsembleLinear(nn.Module):
    def __init__(self, ensemble_size, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EnsembleLinear, self).__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((ensemble_size, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # if len(input.shape) == 2:
        #     input = input.repeat(self.ensemble_size, 1, 1)
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class EnsembleLSTMCell(nn.Module):
    def __init__(self, ensemble_size, input_size, hidden_size, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EnsembleLSTMCell, self).__init__()
        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.empty((ensemble_size, input_size, hidden_size*4), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((ensemble_size, hidden_size, hidden_size*4), **factory_kwargs))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty((ensemble_size, 1, hidden_size*4), **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty((ensemble_size, 1, hidden_size*4), **factory_kwargs))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -std, std)
    
    def forward(self, input, state):
        # if len(input.shape) == 2:
        #     input = input.repeat(self.ensemble_size, 1, 1)
        hx, cx = state
        gates = torch.baddbmm(self.bias_ih, input, self.weight_ih) + torch.baddbmm(self.bias_hh, hx, self.weight_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class EnsembleLSTMLayer(nn.Module):
    def __init__(self, ensemble_size, input_size, hidden_size, bias=True, device=None, dtype=None):
        super(EnsembleLSTMLayer, self).__init__()
        self.cell = EnsembleLSTMCell(ensemble_size, input_size, hidden_size, bias, device, dtype)

    def forward(self, input, state):
        inputs = input.unbind(-2)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs.append(out)
        return torch.permute(torch.stack(outputs), [1, 2, 0, 3]), state
    
def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)

def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean