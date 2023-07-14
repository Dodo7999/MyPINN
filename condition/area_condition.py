from typing import Callable

import torch

from condition.condition import Condition


class AreaCondition(Condition):
    def __init__(self, condition: Callable):
        super().__init__(condition)
        self.data = None
        self.data_cat = None

    def initialize(self, data: torch.Tensor):
        values = []
        for value in data:
            values.append(value)
        self.data = values

    def get_data(self):
        values = self.data
        for ind, value in enumerate(values):
            self.data[ind] = value.detach().requires_grad_()
        return torch.cat(self.data, dim=1)

    def get_loss(self, u):
        loss_area = self.condition(*self.data, u)
        return torch.mean(loss_area) ** 2
