from typing import List

import torch


class Generator:
    def __init__(self, device: torch.device):
        self.device = device

    def get_data(self, coordinate: List[List[int]], index: int, value: int):
        pass

    def get_data_area(self, coordinate: List[List[int]]):
        pass