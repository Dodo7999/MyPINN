import torch

from condition.area_condition import AreaCondition
from condition.boundary_condition import BoundaryCondition
from condition.resolver import Resolver
from dlc.after_epoch_dlc.visualize_dlc.visualize_prediction_dlc import VisualizePredictionDLC
from dlc.after_loss_calculated_dlc.regularization_dlc.normal_losses_regularization_dlc import NormalLossesRegularizationDLC
from generator.uniform_generator import UniformGenerator
from neural_network.feedforward_neural_network import FNN
from pinn.pinn import PINN


def PDE_function(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
    u_x, u_t = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return u_t - u_xx - 2 + 4 * torch.exp(2 * x)


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
            VisualizePredictionDLC()
            # NormalLossesRegularizationDLC()
        ]
    )

    pinn.train()
