import torch

from Condition import Resolver, AreaCondition, BoundaryCondition
from DLC import NormalLossesRegularizationDLC, LambdaLossesRegularizationDLC
from PDE import PDE
from PINN import PINN
from neural_network.feedforward_neural_network import FNN


def PDE_function(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
    return PDE.diff_x(t, u) - PDE.diff_xx(x, u) - 2 + 4 * torch.exp(2 * x)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolver = Resolver(
        conditions=[
            AreaCondition(
                condition=PDE_function
            ),
            BoundaryCondition(
                condition=lambda x: torch.exp(2 * x),
                value_index=0,
                value_const=0,
            ),
            BoundaryCondition(
                condition=lambda t: 1 + 2 * t,
                value_index=1,
                value_const=0,
            ),
            BoundaryCondition(
                condition=lambda t: torch.exp(torch.tensor(4)) + 2 * t,
                value_index=1,
                value_const=2,
            )
        ],
        coordinates=[
            [0, 1, 50],
            [0, 2, 50]
        ],
        device=device
    )
    layers = [2, 128, 128, 128, 1]

    model = FNN(layers_all=layers).to(device)

    pinn = PINN(
        model=model,
        device=device,
        resolver=resolver,
        count_of_epoch=1_000,
        dlcs=[
            LambdaLossesRegularizationDLC()
        ]
    )

    pinn.train()
