from typing import Callable, List

import torch

from Generator import Generator


class Condition:
    def __init__(self, condition: Callable):
        self.condition = condition


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
        return torch.cat(self.data, dim=1)

    def get_loss(self, u):
        loss_area = None
        if len(self.data) == 1:
            loss_area = self.condition(self.data[0], u)
        elif len(self.data) == 2:
            loss_area = self.condition(self.data[0], self.data[1], u)
        elif len(self.data) == 3:
            loss_area = self.condition(self.data[0], self.data[1], self.data[2], u)
        elif len(self.data) == 4:
            loss_area = self.condition(self.data[0], self.data[1], self.data[2], self.data[3], u)
        return torch.mean(loss_area)**2


class BoundaryCondition(Condition):
    def __init__(self, condition: Callable, value_index: int, value_const: int):
        super().__init__(condition)
        self.value = None
        self.data = None
        self.value_index = value_index
        self.value_const = value_const

    def initialize(self, data: torch.Tensor):
        self.data = data
        for point_all in data:
            point = point_all[~self.value_index].reshape(-1)
            if point.shape[0] == 1:
                self.value = self.condition(point[0])
            elif point.shape[0] == 2:
                self.value = self.condition(point[0], point[1])
            elif point.shape[0] == 3:
                self.value = self.condition(point[0], point[1], point[2])
            elif point.shape[0] == 4:
                self.value = self.condition(point[0], point[1], point[2], point[3])
            elif point.shape[0] == 5:
                self.value = self.condition(point[0], point[1], point[2], point[3], point[4])

    def get_data(self):
        return self.data

    def get_loss(self, u):
        return torch.mean((u-self.value)**2)


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
            if len(self.coordinates) == 2:
                data = self.generator.get_2D_data(self.coordinates, condition.value_index, condition.value_const)
                condition.initialize(data)
            elif len(self.coordinates) == 3:
                data = self.generator.get_3D_data(self.coordinates[~condition.value_index])
                condition.initialize(data)
            elif len(self.coordinates) == 4:
                data = self.generator.get_4D_data(self.coordinates[~condition.value_index])
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
