from typing import List

import torch


class Generator:
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def get_data(self, coordinate: List[List[int]], index: int, value: int):
        bound = []
        for ind_y, y in enumerate(coordinate):
            if ind_y != index:
                bound.append(torch.linspace(y[0], y[1], y[2]))
            else:
                bound.append(torch.ones(y[2]) * value)
        return torch.stack(bound, dim=1).float().to(self.device).requires_grad_()

    def get_1D_data_area(self, coordinate: List[int]):
        return torch.linspace(coordinate[0], coordinate[1], coordinate[2])

    def get_2D_data_area(self, coordinate: List[List[int]]):
        x = torch.linspace(coordinate[0][0], coordinate[0][1], coordinate[0][2])
        y = torch.linspace(coordinate[1][0], coordinate[1][1], coordinate[1][2])
        x, y = torch.meshgrid(x, y, indexing="ij")
        x = x.float().to(self.device).requires_grad_()
        y = y.float().to(self.device).requires_grad_()
        return torch.stack([x.reshape(-1, 1), y.reshape(-1, 1)])

    def get_3D_data_area(self, coordinate: List[List[int]]):
        return torch.linspace(coordinate[0], coordinate[1], coordinate[2])

    def get_4D_data_area(self, coordinate: List[List[int]]):
        return torch.linspace(coordinate[0], coordinate[1], coordinate[2])
