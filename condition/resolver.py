from typing import List

import torch

from condition.area_condition import AreaCondition
from condition.boundary_condition import BoundaryCondition
from condition.condition import Condition
from generator.uniform_generator import UniformGenerator


class Resolver:
    def __init__(self, coordinates: list, conditions: List[Condition], device: torch.device):
        self.coordinates = coordinates
        self.conditions = conditions
        self.generator = UniformGenerator(device)

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
            data = self.generator.get_data_area(self.coordinates)
            condition.initialize(data)
