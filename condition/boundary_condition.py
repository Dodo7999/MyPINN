from typing import Callable

import torch

from condition.condition import Condition


class BoundaryCondition(Condition):
    def __init__(self, condition: Callable, value_index: int, value_const: int):
        super().__init__(condition)
        self.value = None
        self.data = None
        self.value_index = value_index
        self.value_const = value_const

    def initialize(self, data: torch.Tensor):
        self.data = data
        point = data[:, torch.arange(data.shape[1]) != self.value_index]
        self.value = self.condition(*point.T).reshape(-1,1).detach()

    def get_data(self):
        return self.data.detach().requires_grad_()

    def get_loss(self, u):
        return torch.mean((u - self.value) ** 2)
