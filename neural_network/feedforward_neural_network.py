import torch
import torch.nn as nn
from siren import Sine


class FNN(nn.Module):
    """
    Класс Feedforward Neural Network (FNN).

    Основная нейросетевая модель для решение дифференциальных уравнений.
    """

    def __init__(
            self,
            layers_all: list,
            activation_function_array: list = None
    ):
        """
        Args:
            layers_all (list): Список, содержащий количества нейроннов для каждого линейного слоя. Первый элемент
            обозначает рамерность входных параметров, последний элемент обозначает размерность выходных параметров.

            activation_function_array (list): Лист, содержащий функции активации, которые используются при обучении.
        """
        super(FNN, self).__init__()

        if activation_function_array is None:
            activation_function_array = [Sine(w0=1), nn.GELU()]

        layers = layers_all[1:-1]

        linears = [nn.Sequential(
            nn.Linear(layers_all[0], layers[0]),
            activation_function_array[0]
        )]
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            linears.append(
                nn.Sequential(
                    nn.Linear(layer, next_layer),
                    activation_function_array[-1]
                )
            )

        linears.append(
            nn.Linear(layers[-1], layers_all[-1])
        )

        self.linears = nn.ModuleList(linears)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход

        Args:
            x (torch.Tensor):  Входной тензор (размер батча, размерность x).
            t (torch.Tensor):  Входной тензор (размер батча, размерность t).

        returns:
            torch.Tensor: Выходной тензон (размер батча, 1).
        """

        for layer in self.linears:
            x = layer(x)
        return x
