import torch.nn as nn
import torch



def weighted_loss(output, target, kind): 
    crt_mse = nn.MSELoss()
    crt_mae = nn.L1Loss()
    output, target = output.to(dtype=torch.float64).ravel(), target.to(dtype=torch.float64).ravel()
    if kind == 'mse':
        return crt_mse(output, target)
    elif kind == 'mae':
        return crt_mae(output, target)