from typing import List

import torch

from dlc.dlc import DLC


class AfterLossCalculatedDLC(DLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor]):
        pass
