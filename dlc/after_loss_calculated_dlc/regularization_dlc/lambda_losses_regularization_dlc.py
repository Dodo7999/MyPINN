from typing import List

import torch

from dlc.after_loss_calculated_dlc.after_loss_calculated_dlc import AfterLossCalculatedDLC


class LambdaLossesRegularizationDLC(AfterLossCalculatedDLC):
    def __init__(self):
        super().__init__()
        self.lambda_regularization = None

    def do(self, losses: List[torch.Tensor]):
        last_layers = list(list(self.model.children())[-1].children())[-1]
        alpha = 0.5
        if self.lambda_regularization is None:
            self.lambda_regularization = torch.ones((len(losses) - 1))

        losses[-1].backward(retain_graph=True)
        max_f = torch.max(torch.abs(last_layers.weight.grad.detach()))
        self.optimizer.zero_grad()

        for ind, los in enumerate(losses[:-1]):
            losses[ind].backward(retain_graph=True)
            regularization = max_f / torch.abs(
                last_layers.weight.grad.detach() * self.lambda_regularization[ind]
            ).mean()
            self.optimizer.zero_grad()
            losses[ind] = losses[ind] * ((1 - alpha) * self.lambda_regularization[ind] + alpha * regularization)
            self.lambda_regularization[ind] = regularization
        return losses
