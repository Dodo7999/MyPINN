import torch

from condition.resolver import Resolver


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
