import torch

from condition.area_condition import AreaCondition
from condition.boundary_condition import BoundaryCondition
from condition.equals_boundary_condition import EqualsBoundaryCondition
from condition.resolver import Resolver
from dlc.after_epoch_dlc.visualize_dlc.visualize_prediction_dlc import VisualizePredictionDLC
from generator.uniform_generator import UniformGenerator
from neural_network.feedforward_neural_network import FNN
from pinn.pinn import PINN
from settings.settings import VisualizeSettings

betta = 30


def PDE_function(x: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
    u_x, u_t = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    return u_t + betta * u_x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolver = Resolver(
        conditions=[
            AreaCondition(
                condition=PDE_function
            ),
            BoundaryCondition(
                condition=lambda x: torch.sin(x),
                value_index=1,
                value_const=0,
            ),
            EqualsBoundaryCondition(
                value_index_first=0,
                value_index_second=0,
                value_const_first=0,
                value_const_second=2 * torch.pi
            ),
        ],
        coordinates=[
            [0, 2 * torch.pi, 50],
            [0, 1, 50],
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
            *VisualizeSettings
        ]
    )

    pinn.train()
