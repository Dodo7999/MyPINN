import torch
import torch.nn as nn


class LightResidualBlock(nn.Module):
    """
    Класс Residual Block.
    """

    def __init__(
            self,
            activation,
            features: int
    ):
        """
        Args:
            features (int): Размерность входных и выходных данных данных.

            activation (nn.Module): Функция активации. Для кастомных функция активации достаточно реализовать класс
            обертку наследующий nn.Module и реализующий функцию forward.
        """
        super(LightResidualBlock, self).__init__()
        self.linear_first = nn.Sequential(
            nn.Linear(features, features),
            activation
        )
        self.linear_second = nn.Linear(features, features)
        self.activation = activation

    def forward(self, x: torch.Tensor):
        """
        Прямой проход

        Args:
            x (torch.Tensor):  Входной тензор (размер батча, размерность in_features).

        returns:
            torch.Tensor: Выходной тензон (размер батча, out_features).
        """
        residual = x
        out = self.linear_first(x)
        out = self.linear_second(out)
        out += residual
        out = self.activation(out)
        return out
