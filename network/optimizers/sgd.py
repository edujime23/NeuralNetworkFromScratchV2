from ..types import Tensor
from .base import Optimizer
from numba import njit
import numpy as np


class SGD(Optimizer):
    """
    Complex-valued Stochastic Gradient Descent (SGD) optimizer.
    Applies updates based on the gradient direction with a configurable learning rate.
    Automatically conjugates gradients in apply_gradients().
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def update_step(self, grad: Tensor, var: Tensor) -> None:
        var_update = self._update_step_math(grad.numpy, self.learning_rate)
        var[...] -= Tensor(var_update)

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(grad: np.ndarray, learning_rate: float) -> np.ndarray:
        return learning_rate * grad

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({"learning_rate": self.learning_rate})
        return base_config

    @classmethod
    def get_slot_names(cls) -> list[str]:
        return []
