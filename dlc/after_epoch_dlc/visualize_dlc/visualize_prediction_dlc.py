from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dlc.after_epoch_dlc.after_epoch_dlc import AfterEpochDLC


class VisualizePredictionDLC(AfterEpochDLC):
    def __init__(self):
        super().__init__()
        self.lambda_regularization = None

    def do(self, losses: List[torch.Tensor], loss: torch.Tensor, number_epoch: int):
        if number_epoch % 100 == 0:
            with torch.no_grad():
                condition = self.resolver.conditions[0]
                data = condition.get_data()
                u = self.model(data).detach().cpu().numpy().reshape(50, 50)

            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(111)

            h = ax.imshow(u, interpolation='nearest', cmap='rainbow',
                          extent=[-1, 1, -1, 1],
                          origin='lower', aspect='auto', vmin=np.min(u), vmax=np.max(u))

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=15)

            ax.set_xlabel('y', fontweight='bold', size=30)
            ax.set_ylabel('x', fontweight='bold', size=30)

            ax.legend(
                labels='Visualize u_predicted.',
                loc='upper center',
                bbox_to_anchor=(0.9, -0.05),
                ncol=5,
                frameon=False,
                prop={'size': 15}
            )

            ax.tick_params(labelsize=15)

            plt.savefig('foo.png')
