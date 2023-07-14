from typing import List

import torch

from dlc.after_loss_calculated_dlc.after_loss_calculated_dlc import AfterLossCalculatedDLC


class NormalLossesRegularizationDLC(AfterLossCalculatedDLC):
    def __init__(self):
        super().__init__()

    def do(self, losses: List[torch.Tensor]):
        last_layers = list(list(self.model.children())[-1].children())[-1]
        losses[-1].backward(retain_graph=True)
        var_f = torch.std(last_layers.weight.grad.detach())
        self.optimizer.zero_grad()
        for ind, los in enumerate(losses[:-1]):
            losses[ind].backward(retain_graph=True)
            var = torch.std(last_layers.weight.grad.detach())
            self.optimizer.zero_grad()
            losses[ind] = losses[ind] / var * var_f
        return losses
