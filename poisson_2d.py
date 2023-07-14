import torch

from condition.area_condition import AreaCondition
from condition.boundary_condition import BoundaryCondition
from condition.resolver import Resolver
from dlc.after_loss_calculated_dlc.normal_losses_regularization_dlc import NormalLossesRegularizationDLC
from generator.uniform_generator import UniformGenerator
from neural_network.feedforward_neural_network import FNN
from pinn.pinn import PINN


def PDE_function(x: torch.Tensor, y: torch.Tensor, u: torch.Tensor):
    u_x, u_y = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return -u_yy - u_xx - torch.pi ** 2 * torch.sin(torch.pi * x) - torch.pi ** 2 * torch.sin(torch.pi * y)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolver = Resolver(
        conditions=[
            AreaCondition(
                condition=PDE_function
            ),
            BoundaryCondition(
                condition=lambda x: torch.sin(torch.pi * x) + torch.sin(-torch.tensor(torch.pi)),
                value_index=0,
                value_const=-1,
            ),
            BoundaryCondition(
                condition=lambda x: torch.sin(torch.pi * x) + torch.sin(torch.tensor(torch.pi)),
                value_index=0,
                value_const=1,
            ),
            BoundaryCondition(
                condition=lambda y: torch.sin(torch.pi * y) + torch.sin(-torch.tensor(torch.pi)),
                value_index=0,
                value_const=-1,
            ),
            BoundaryCondition(
                condition=lambda y: torch.sin(torch.pi * y) + torch.sin(torch.tensor(torch.pi)),
                value_index=1,
                value_const=1,
            ),
        ],
        coordinates=[
            [-1, 1, 50],
            [-1, 1, 50]
        ],
    )
    layers = [2, 128, 128, 128, 1]

    model = FNN(layers_all=layers).to(device)

    pinn = PINN(
        model=model,
        device=device,
        resolver=resolver,
        generator=UniformGenerator(device),
        count_of_epoch=5_000,
        dlcs=[
            # NormalLossesRegularizationDLC()
        ]
    )

    pinn.train()
