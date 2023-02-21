import numpy as np
import torch

def normalize(data_array, mean, std):
    if isinstance(data_array, np.ndarray):
        return (data_array - mean) / (std + 1e-10)
    else:
        return (data_array - torch.as_tensor(mean, device=data_array.device)) / (torch.as_tensor(std, device=data_array.device) + 1e-10)

def denormalize(data_array, mean, std):
    if isinstance(data_array, np.ndarray):
        return data_array * (std + 1e-10) + mean
    else:
        return data_array * (torch.as_tensor(std, device=data_array.device) + 1e-10) + torch.as_tensor(mean, device=data_array.device)