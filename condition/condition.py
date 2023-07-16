from typing import Callable

import torch


class Condition:
    def __init__(self, condition: Callable):
        self.condition = condition

    def initialize(self, data: torch.Tensor):
        pass

    def get_data(self):
        pass

    def get_loss(self, u):
        pass
