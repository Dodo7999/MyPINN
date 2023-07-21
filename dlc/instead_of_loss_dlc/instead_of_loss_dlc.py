from typing import List

import torch

from dlc.dlc import DLC


class InsteadOfLossDLC(DLC):
    def __init__(self):
        super().__init__()

    def get_losses(self):
        pass
