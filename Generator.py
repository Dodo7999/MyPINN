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

    def get_data_area(self, coordinate: List[List[int]]):
        x = []
        for coord in coordinate:
            x.append(torch.linspace(coord[0], coord[1], coord[2]))
        y = torch.stack(list(torch.meshgrid(x)), dim=0)
        x = []
        for z in y:
            x.append(z.reshape(-1, 1).float().to(self.device).requires_grad_())
        return torch.stack(x)
