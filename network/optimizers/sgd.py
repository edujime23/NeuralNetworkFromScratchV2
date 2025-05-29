import numpy as np
from typing import List
from .base import Optimizer
from numba import njit


class SGD(Optimizer):
    """
    Complex-valued Stochastic Gradient Descent (SGD) optimizer.
    Applies updates based on the gradient direction with a configurable learning rate.
    Automatically conjugates gradients in apply_gradients().
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
        var_update = self._update_step_math(grad, self.learning_rate)
        var[...] -= var_update

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(grad: np.ndarray, learning_rate: float) -> np.ndarray:
        return learning_rate * grad

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({"learning_rate": self.learning_rate})
        return base_config

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return []
