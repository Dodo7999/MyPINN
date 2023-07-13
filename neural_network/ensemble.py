import torch
import torch.nn as nn

class Ensemble(nn.Module):
    """
    Ансамбль моделей для решение дифференциальных уравнений
    """

    def __init__(self, model_array, input):
        """
        Args:
        model_array (list) : лист моделей, которые входят в ансамбль
        """
        super(Ensemble, self).__init__()
        self.model_array = model_array
        self.fc = nn.Linear(input, 1)
        self.predict_models_array = []

    def forward(self, x: torch.Tensor):
        predict_models_array = [self.predict_models_array.append(models(x)) for models in self.model_array]
        total_prediction = sum(predict_models_array)
        output = self.fc(total_prediction)
        return output
