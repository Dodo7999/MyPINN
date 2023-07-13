from typing import Callable

import torch

from Condition import Resolver


class PINN:
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            resolver: Resolver,
            loss_function=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,

    ):
        self.model = model
        self.device = device
        self.resolver = resolver
        self.loss_function = loss_function
        self.optimizer = optimizer(model.parameters(), 0.01)
        self.scheduler = scheduler(self.optimizer, gamma=0.9999)
        self.resolver.initialize_conditions()

    def train(self):
        conditions = self.resolver.conditions
        for i in range(5000):
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for condition in conditions:
                data = condition.get_data()
                u = self.model(data)
                loss = condition.get_loss(u)
                losses.append(loss)
            loss_t = torch.stack(losses).sum()
            print(loss_t)
            loss_t.backward()
            self.optimizer.step()
            self.scheduler.step()
