from typing import List

import torch

from dlc.dlc import DLC


class AfterEpochDLC(DLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor], loss: torch.Tensor, number_epoch: int):
        pass
