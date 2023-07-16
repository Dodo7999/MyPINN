from typing import List

from condition.area_condition import AreaCondition
from condition.boundary_condition import BoundaryCondition
from condition.condition import Condition
from condition.equals_boundary_condition import EqualsBoundaryCondition
from generator.generator import Generator


class Resolver:
    def __init__(self, coordinates: list, conditions: List[Condition]):
        self.coordinates = coordinates
        self.conditions = conditions
        self.generator = None

    def set_generator(self, generator: Generator):
        self.generator = generator

    def initialize_conditions(self):
        filtered_boundary = filter(
            lambda condition_this: isinstance(condition_this, BoundaryCondition),
            self.conditions
        )
        filtered_equals_boundary = filter(
            lambda condition_this: isinstance(condition_this, EqualsBoundaryCondition),
            self.conditions
        )
        filtered_area = filter(
            lambda condition_this: isinstance(condition_this, AreaCondition),
            self.conditions
        )
        list_boundary: List[BoundaryCondition] = list(filtered_boundary)
        list_equals_boundary: List[EqualsBoundaryCondition] = list(filtered_equals_boundary)
        list_area: List[AreaCondition] = list(filtered_area)

        for condition in list_boundary:
            data = self.generator.get_data(self.coordinates, condition.value_index, condition.value_const)
            condition.initialize(data)

        for ind, condition in enumerate(list_equals_boundary):
            self.conditions.append(condition)
            data = self.generator.get_data(self.coordinates, condition.value_index_first, condition.value_const_first)
            condition.initialize(data)
            data = self.generator.get_data(self.coordinates, condition.value_index_second, condition.value_const_second)
            condition.initialize(data)

        for ind, condition in enumerate(list_area):
            data = self.generator.get_data_area(self.coordinates)
            condition.initialize(data)

    def update_condition_data(self):
        pass
