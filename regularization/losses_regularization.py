import torch
import torch.nn as nn


def add_losses_regularization(losses: list, last_layers: nn.Linear, optimizer: torch.optim.Optimizer):
    alpha = 0.1
    lambda_regularization = torch.ones((len(losses) - 1))

    losses[-1].backward(retain_graph=True)
    max_f = torch.max(torch.abs(last_layers.weight.grad.detach()))
    optimizer.zero_grad()

    for ind, los in enumerate(losses[:-1]):
        losses[ind].backward(retain_graph=True)
        regularization = max_f / torch.abs(
            last_layers.weight.grad.detach() * lambda_regularization[ind]
        ).mean()
        optimizer.zero_grad()
        losses[ind] = losses[ind] * ((1 - alpha) * lambda_regularization[ind] + alpha * regularization)
    return losses
