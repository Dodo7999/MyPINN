import torch

from dlc.instead_of_loss_dlc.instead_of_loss_dlc import InsteadOfLossDLC


# GitHub с кодом авторов https://github.com/shamsbasir/investigating_mitigating_failure_modes_in_pinns/blob/main/Helmholtz/Helmholtz.ipynb

class BigLambdaLossesRegularizationDLC(InsteadOfLossDLC):
    # В статье Investigating and Mitigating Failure Modes in Physicsinformed Neural Networks (PINNs)
    # рекомендовалось именно значение beta = 0.9 и eta = 0.01
    def __init__(self):
        super().__init__()
        self.v_bc = []
        self.lambda_bc = []
        self.beta = 0.9
        self.eta = 0.01
        self.conditions = None

    def get_losses(self):
        if self.conditions is None:
            self.conditions = self.resolver.conditions
        else:
            # После первого шага оптимизации нужно обновлять коэффициенты регуляризации, ниже код обновляет коэффициенты
            for ind, condition in enumerate(self.conditions[1:]):
                data = condition.get_data()
                u = self.model(data)
                boundary_res = (u - condition.value) ** 2

                with torch.no_grad():
                    self.v_bc[ind] = (1.0 - self.beta) * boundary_res.pow(2) + self.beta * self.v_bc[ind]
                    self.lambda_bc[ind] += self.eta / (self.v_bc[ind] + 1e-8).sqrt() * boundary_res

        # Считаю что самый первый condition является условием области, его не нужно регуляризовывать, поэтому он обрабатывается отдельно
        losses = []
        condition = self.conditions[0]
        data = condition.get_data()
        u = self.model(data)
        loss = condition.get_loss(u)
        losses.append(loss)
        # добавление регуляризации на граничные условия
        for ind, condition in enumerate(self.conditions[1:]):
            data = condition.get_data()
            u = self.model(data)
            boundary_res = (u - condition.value) ** 2
            if len(self.lambda_bc) == ind:
                self.lambda_bc.append(torch.ones_like(boundary_res))

            if len(self.v_bc) == ind:
                self.v_bc.append(torch.ones_like(boundary_res))

            boundary_loss = (self.lambda_bc[ind] * boundary_res).sum()

            losses.append(boundary_loss)

        return losses
