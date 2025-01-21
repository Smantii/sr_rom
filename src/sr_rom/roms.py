from abc import ABC, abstractmethod
import numpy as np
from sklearn import linear_model
from typing import Dict
import numpy.typing as npt
from pyoperon.sklearn import SymbolicRegressor
import torch
from torch import Tensor, nn
from skorch import NeuralNetRegressor
from skorch.utils import to_device
import torch.jit as jit
from typing import List


class NeuralNetwork(nn.Module):
    """Standard Multi-Layer Perceptron.

    Args:
        r: input dimension.
        hidden_units: list containing the number of nodes per hidden layer.
        dropout_rate: dropout rate per layer.
    """

    def __init__(self, r: int, hidden_units: List, dropout_rate: float = 0.5):
        super().__init__()
        self.flatten = nn.Flatten()
        # define hidden_layers
        hidden_layers = [nn.Linear(r, hidden_units[0]),
                         nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for i in range(1, len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.LeakyReLU())
            hidden_layers.append(nn.Dropout(dropout_rate))
        # append last layer
        hidden_layers.append(nn.Linear(hidden_units[-1], 1))

        # nn stack
        self.linear_relu_stack = nn.Sequential(*hidden_layers)

    def forward(self, x):
        """Compute the output of the net given an input x.

        Args:
            x: the input vector
        Returns:
            the output of the net.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class rom(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray | Tensor, y: npt.NDArray | Tensor):
        """
        Train the model on the given data.
        :param X: Feature matrix (2D array-like)
        :param y: Target vector (1D array-like)
        """
        pass

    @abstractmethod
    def predict(self, X: npt.NDArray | Tensor) -> npt.NDArray | Tensor:
        """
        Predict the target values for the given data.
        :param X: Feature matrix (2D array-like)
        :return: Predicted values (1D array-like)
        """
        pass

    @abstractmethod
    def inference_mode(self) -> None:
        pass


class lr_rom(rom):
    def __init__(self, r: int, alpha: float):
        self.r = r
        self.alpha = alpha
        # define linear regressor
        self.reg = [linear_model.Ridge(alpha=alpha) for _ in range(r)]

    def fit(self, X: npt.NDArray, y: npt.NDArray):
        for i in range(self.r):
            self.reg[i].fit(X, y[:, i])

    def predict(self, X: npt.NDArray):
        predictions = [self.reg[i].predict(X) for i in range(self.r)]
        return predictions

    def inference_mode(self):
        pass


class sr_rom(rom):
    def __init__(self, r: int, model_parameters: Dict):
        self.r = r
        fixed_params = {"population_size": 1000,
                        "optimizer_iterations": 10,
                        "n_threads": 32,
                        "tournament_size": 5,
                        'objectives':  ['r2'],
                        'max_depth': 5}
        # merge dictionaries
        self.model_parameters = model_parameters | fixed_params
        # define symbolic regressor
        self.reg = [SymbolicRegressor(**self.model_parameters) for _ in range(self.r)]

    def fit(self, X: npt.NDArray, y: npt.NDArray):
        for i in range(self.r):
            self.reg[i].fit(X, y[:, i])

    def predict(self, X: npt.NDArray):
        predictions = [self.reg[i].predict(X) for i in range(self.r)]
        return predictions

    def inference_mode(self):
        pass


class nn_rom(rom):
    def __init__(self, r: int, model_parameters: Dict, device: str):
        self.r = r
        self.device = device
        fixed_params = {"module": NeuralNetwork,
                        "batch_size": 512,
                        "verbose": 0,
                        "optimizer": torch.optim.Adam,
                        "max_epochs": 100,
                        "train_split": None,
                        "device": self.device,
                        "iterator_train__shuffle": True}
        self.model_parameters = fixed_params | model_parameters
        # define neural net regressor
        self.reg = [NeuralNetRegressor(**self.model_parameters) for _ in range(self.r)]

    def fit(self, X: Tensor, y: Tensor):
        X_nn = torch.from_numpy(X).to(torch.float32)
        y_nn = torch.from_numpy(y).to(torch.float32)
        for i in range(self.r):
            self.reg[i].fit(X_nn, y_nn[:, i].reshape(-1, 1))

    def predict(self, X: Tensor):
        with torch.no_grad():
            X_reshaped = torch.from_numpy(X).to(torch.float32)
            predictions = [self.reg[i].module_.forward(
                X_reshaped) for i in range(self.r)]
            return predictions

    def inference_mode(self):
        for i in range(self.r):
            self.reg[i].set_params(device='cpu')

            # Moves current model instance to cpu
            to_device(self.reg[i].module_, 'cpu')
            self.reg[i].module_.eval()
            self.reg[i].module_ = jit.script(self.reg[i].module_)
