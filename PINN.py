from typing import Callable

import torch
import tqdm

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
        self.model.train()
        iterations = tqdm.trange(5000 + 1, desc="Here we will write the loss")
        conditions = self.resolver.conditions
        for it in iterations:
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for condition in conditions:
                data = condition.get_data()
                u = self.model(data)
                loss = condition.get_loss(u)
                losses.append(loss)
            loss_t = torch.stack(losses).sum()
            loss_t.backward()
            self.optimizer.step()
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]["lr"]
            iterations.set_description(
                f"At epoch #{it}: loss = {loss_t:.3e}, lr = {curr_lr:.3e}, rel_er=--"
            )
            iterations.refresh()
