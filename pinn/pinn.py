from typing import List

import torch

from condition.resolver import Resolver
from dlc.dlc import DLC
from trainer.trainer import Trainer


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
        self.trainer = Trainer(
            model=model,
            device=device,
            resolver=resolver,
            count_of_epoch=count_of_epoch,
            loss_function=loss_function,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dlcs=self.dlcs
        )

    def train(self):
        self.trainer.train()
