from typing import Callable

import torch

from condition.condition import Condition


class EqualsBoundaryCondition(Condition):
    def __init__(
            self,
            value_index_first: float,
            value_const_first: float,
            value_index_second: float,
            value_const_second: float,
            condition: Callable = None
    ):
        super().__init__(condition)
        self.data_first = None
        self.data_second = None
        self.value_index_first = value_index_first
        self.value_const_first = value_const_first
        self.value_index_second = value_index_second
        self.value_const_second = value_const_second
        self.flag = 0
        self.predict_first_value = None

    def initialize(self, data: torch.Tensor):
        if self.flag == 0:
            self.data_first = data
            self.flag = 1
        else:
            self.data_second = data
            self.flag = 0

    def get_data(self):
        if self.flag == 0:
            return self.data_first.detach().requires_grad_()
        else:
            return self.data_second.detach().requires_grad_()

    def get_loss(self, u):
        if self.flag == 0:
            self.predict_first_value = u
            self.flag = 1
            return torch.tensor(0)
        else:
            self.flag = 0
            return torch.mean((u - self.predict_first_value) ** 2)
