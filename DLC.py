from typing import List

import torch

from Condition import Resolver


class DLC:
    def __init__(self):
        self.model = None
        self.device = None
        self.resolver = None
        self.count_of_epoch = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None

    def apply(self,
              model: torch.nn.Module,
              device: torch.device,
              resolver: Resolver,
              count_of_epoch: int,
              loss_function=torch.nn.MSELoss(),
              optimizer=torch.optim.Adam,
              scheduler=torch.optim.lr_scheduler.ExponentialLR
              ):
        self.model = model
        self.device = device
        self.resolver = resolver
        self.count_of_epoch = count_of_epoch
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler


class AfterLossCalculatedDLC(DLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor]):
        pass


class NormalLossesRegularizationDLC(AfterLossCalculatedDLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor]):
        last_layers = list(list(self.model.children())[-1].children())[-1]
        losses[-1].backward(retain_graph=True)
        var_f = torch.std(last_layers.weight.grad.detach())
        self.optimizer.zero_grad()
        for ind, los in enumerate(losses[:-1]):
            losses[ind].backward(retain_graph=True)
            var = torch.std(last_layers.weight.grad.detach())
            self.optimizer.zero_grad()
            losses[ind] = losses[ind] / var * var_f
        return losses


class LambdaLossesRegularizationDLC(AfterLossCalculatedDLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor]):
        last_layers = list(list(self.model.children())[-1].children())[-1]
        alpha = 0.1
        lambda_regularization = torch.ones((len(losses) - 1))

        losses[-1].backward(retain_graph=True)
        max_f = torch.max(torch.abs(last_layers.weight.grad.detach()))
        self.optimizer.zero_grad()

        for ind, los in enumerate(losses[:-1]):
            losses[ind].backward(retain_graph=True)
            regularization = max_f / torch.abs(
                last_layers.weight.grad.detach() * lambda_regularization[ind]
            ).mean()
            self.optimizer.zero_grad()
            losses[ind] = losses[ind] * ((1 - alpha) * lambda_regularization[ind] + alpha * regularization)
        return losses
