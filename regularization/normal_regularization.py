import torch
import torch.nn as nn


def add_normal_regularization(losses: list, last_layers: nn.Linear, optimizer: torch.optim.Optimizer):
    losses[-1].backward(retain_graph=True)
    var_f = torch.std(last_layers.weight.grad.detach())
    optimizer.zero_grad()
    for ind, los in enumerate(losses[:-1]):
        losses[ind].backward(retain_graph=True)
        var = torch.std(last_layers.weight.grad.detach())
        optimizer.zero_grad()
        losses[ind] = losses[ind] / var * var_f
    return losses
