from typing import Callable, List

import torch

from Generator import Generator


class Condition:
    def __init__(self, condition: Callable):
        self.condition = condition

    def initialize(self, data: torch.Tensor):
        pass

    def get_data(self):
        pass

    def get_loss(self, u):
        pass


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


class BoundaryCondition(Condition):
    def __init__(self, condition: Callable, value_index: int, value_const: int):
        super().__init__(condition)
        self.value = None
        self.data = None
        self.value_index = value_index
        self.value_const = value_const

    def initialize(self, data: torch.Tensor):
        self.data = data
        point = data[:, ~self.value_index]
        if data.shape[1] == 2:
            point = point.reshape(-1, 1)
            self.value = self.condition(point).detach()
        else:
            self.value = self.condition(*point).detach()

    def get_data(self):
        return self.data.detach().requires_grad_()

    def get_loss(self, u):
        return torch.mean((u - self.value) ** 2)


class Resolver:
    def __init__(self, coordinates: list, conditions: List[Condition], device: torch.device):
        self.coordinates = coordinates
        self.conditions = conditions
        self.generator = Generator(device)

    def initialize_conditions(self):
        filtered_not_area = filter(
            lambda condition_this: isinstance(condition_this, BoundaryCondition),
            self.conditions
        )
        filtered_area = filter(
            lambda condition_this: isinstance(condition_this, AreaCondition),
            self.conditions
        )
        list_boundary: List[BoundaryCondition] = list(filtered_not_area)
        list_area: List[AreaCondition] = list(filtered_area)

        for condition in list_boundary:
            data = self.generator.get_data(self.coordinates, condition.value_index, condition.value_const)
            condition.initialize(data)

        for ind, condition in enumerate(list_area):
            if len(self.coordinates) == 1:
                data = self.generator.get_1D_data_area(self.coordinates)
                condition.initialize(data)
            elif len(self.coordinates) == 2:
                data = self.generator.get_2D_data_area(self.coordinates)
                condition.initialize(data)
            elif len(self.coordinates) == 3:
                data = self.generator.get_3D_data_area(self.coordinates)
                condition.initialize(data)
            elif len(self.coordinates) == 4:
                data = self.generator.get_4D_data_area(self.coordinates)
                condition.initialize(data)
