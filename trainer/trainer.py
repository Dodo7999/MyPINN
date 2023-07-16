from typing import List

import torch.optim
import tqdm

from condition.resolver import Resolver
from dlc.after_epoch_dlc.after_epoch_dlc import AfterEpochDLC
from dlc.after_loss_calculated_dlc.after_loss_calculated_dlc import AfterLossCalculatedDLC
from dlc.dlc import DLC


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            resolver: Resolver,
            count_of_epoch: int,
            loss_function: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            dlcs: List[DLC] = []
    ):
        self.model = model
        self.count_of_epoch = count_of_epoch
        self.device = device
        self.resolver = resolver
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
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
            for dlc in self.dlcs:
                if isinstance(dlc, AfterEpochDLC):
                    dlc.do(losses, loss_t, it)
            iterations.refresh()
