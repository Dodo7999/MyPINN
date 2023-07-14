from typing import Callable, List

import torch
import tqdm

from Condition import Resolver
from DLC import DLC, AfterLossCalculatedDLC


class PINN:
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            resolver: Resolver,
            count_of_epoch: int,
            loss_function=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam,
            scheduler=torch.optim.lr_scheduler.ExponentialLR,
            dlcs: List[DLC] = []
    ):
        self.model = model
        self.count_of_epoch = count_of_epoch
        self.device = device
        self.resolver = resolver
        self.loss_function = loss_function
        self.optimizer = optimizer(model.parameters())
        self.scheduler = scheduler(self.optimizer, gamma=0.9999)
        self.resolver.initialize_conditions()
        for dlc in dlcs:
            dlc.apply(
                model=model,
                device=device,
                resolver=resolver,
                count_of_epoch=count_of_epoch,
                loss_function=loss_function,
                optimizer=self.optimizer,
                scheduler=self.scheduler
            )
        self.dlcs = dlcs

    def train(self):
        self.model.train()
        iterations = tqdm.trange(self.count_of_epoch + 1, desc="Here we will write the loss")
        conditions = self.resolver.conditions
        for it in iterations:
            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for condition in conditions:
                data = condition.get_data()
                u = self.model(data)
                loss = condition.get_loss(u)
                losses.append(loss)
            for dlc in self.dlcs:
                if isinstance(dlc, AfterLossCalculatedDLC):
                    losses = dlc.do(losses)
            loss_t = torch.stack(losses).sum()
            loss_t.backward()
            self.optimizer.step()
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]["lr"]
            iterations.set_description(
                f"At epoch #{it}: loss = {loss_t:.3e}, lr = {curr_lr:.3e}, rel_er=--"
            )
            iterations.refresh()
