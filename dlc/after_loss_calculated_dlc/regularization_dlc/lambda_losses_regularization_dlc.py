from typing import List

import torch

from dlc.after_loss_calculated_dlc.after_loss_calculated_dlc import AfterLossCalculatedDLC


# GitHub с кодом авторов https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/tree/master

class LambdaLossesRegularizationDLC(AfterLossCalculatedDLC):
    # В статье UNDERSTANDING AND MITIGATING GRADIENT FLOW PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS
    # рекомендовалось именно значение alpha = 0.9
    def __init__(self, alpha=0.9):
        super().__init__()
        self.lambda_regularization = None
        self.alpha = alpha

    def do(self, losses: List[torch.Tensor]):
        # Замечание общее, градиенты берутся по всем весам нейронной сети
        parameters = list(self.model.parameters())[:-1]

        if self.lambda_regularization is None:
            self.lambda_regularization = torch.ones((len(losses) - 1))
        # Считаю что первый loss относится к условию внутри области
        losses[0].backward(retain_graph=True)
        grad_of_params = []
        for parameter in parameters:
            grad_of_params.append(torch.max(torch.abs(parameter.grad)))
        max_f = max(grad_of_params)
        self.optimizer.zero_grad()

        for ind, los in enumerate(losses[1:]):
            los.backward(retain_graph=True)
            grad_of_params = []
            for parameter in parameters:
                grad_of_params.append(torch.abs(parameter.grad).mean())
            regularization = max_f / torch.tensor(grad_of_params).mean()
            self.optimizer.zero_grad()
            regularization = (1 - self.alpha) * self.lambda_regularization[ind] + self.alpha * regularization
            losses[ind + 1] = los * regularization
            self.lambda_regularization[ind] = regularization
        return losses
