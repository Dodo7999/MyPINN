import torch
import torch.nn as nn
from siren import Sine

from src.neural_network.light_residual_block import LightResidualBlock


class ResNet(nn.Module):
    """
    Класс Residual Neural Network (ResNet).
    """

    def __init__(
            self,
            layers_all: list,
            blocks: list = None,
            res_block: nn.Module = LightResidualBlock,
            activation_function_array: list = None
    ):
        super(ResNet, self).__init__()
        """
        Args:
            layers_all (list): Список, содержащий количества нейроннов для каждого блока слоев resnet. 
            Включая первый элемент - размерность входных данных и  послеждний элемент размерность выходных данных.

            blocks (list): Список, содержащий количество ResidualBlock для каждого количества нейроннов, 
            можно не указывать, тогда для каждого элемента layers будет один ResidualBlock.

            res_block (nn.Module): Модуль ResidualBlock, который будет использоваться в модели.

            activation_function_array (list): Список, содержащий функции активации, которые используются при обучении.
        """

        if activation_function_array is None:
            activation_function_array = [Sine(w0=1), nn.GELU()]

        if blocks is None:
            blocks = []

        layers = layers_all[1:-1]

        layers_list = [
            nn.Sequential(
                nn.Linear(in_features=layers_all[0], out_features=layers[0]),
                activation_function_array[0]
            )
        ]

        if len(blocks) == 0:
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layers_list.append(
                    self.__make_layers(
                        res_block,
                        1,
                        layer,
                        next_layer,
                        activation_function_array[-1]
                    )
                )
            layers_list.append(
                self.__make_layers(
                    res_block,
                    1,
                    layers[-1],
                    layers[-1],
                    activation_function_array[-1],
                    False
                )
            )
        else:
            for layer, block, next_layer in zip(layers[:-1], blocks[:-1], layers[1:]):
                layers_list.append(
                    self.__make_layers(
                        res_block,
                        block,
                        layer,
                        next_layer,
                        activation_function_array[-1]
                    )
                )
            layers_list.append(
                self.__make_layers(
                    res_block,
                    blocks[-1],
                    layers[-1],
                    layers[-1],
                    activation_function_array[-1],
                    False
                )
            )

        layers_list.append(
            nn.Linear(in_features=layers[-1], out_features=layers_all[-1])
        )

        self.layers = nn.ModuleList(layers_list)

    def __make_layers(
            self,
            res_block: nn.Module,
            count_blocks: int,
            in_features: int,
            out_features: int,
            activation: nn.Module,
            is_not_last: bool = True
    ):
        """
        Function:
            Функция создающая блоки ResidualBlock одного размера и последующий слой смены размерности.
        Args:
            count_blocks (int): Количество ResidualBlock для конкретного элемента спика layers.

            in_features (int): Размерность входных данных.

            out_features (int): Размерность выходных данных.

            activation (nn.Module): Функция активации. Для кастомных функция активации достаточно реализовать класс
            обертку наследующий nn.Module и реализующий функцию forward.

            res_block (nn.Module): Модуль ResidualBlock, который будет использоваться в модели.

            is_not_last (bool): Флаг обозначающий, что следующего блока ResidualBlock нет.
        """

        layers = []

        for i in range(count_blocks - 1):
            layers.append(res_block(activation, in_features))

        layers.append(res_block(activation, in_features))

        if is_not_last:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    activation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Прямой проход

        Args:
            x (torch.Tensor):  Входной тензор (размер батча, размерность x).

            t (torch.Tensor):  Входной тензор (размер батча, размерность t).

        returns:
            torch.Tensor: Выходной тензон (размер батча, 1).
        """
        for layer in self.layers:
            x = layer(x)
        return x
