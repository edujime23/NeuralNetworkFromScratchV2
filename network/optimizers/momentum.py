import numpy as np
from typing import List, Tuple
from .base import Optimizer
from numba import njit


class Momentum(Optimizer):
    """
    Complex-valued SGD with Momentum.
    Accumulates an exponentially decaying moving average of past gradients and uses that to update variables.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def build(self, var_list: List[np.ndarray]) -> None:
        for var in var_list:
            self.add_slot(var, 'velocity')

    def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
        velocity = self.get_slot(var, 'velocity')

        velocity_new, var_update = self._update_step_math(
            velocity, grad, self.learning_rate, self.momentum
        )

        velocity[...] = velocity_new
        var[...] -= var_update

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        velocity: np.ndarray,
        grad: np.ndarray,
        learning_rate: float,
        momentum: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        velocity_new = momentum * velocity + learning_rate * grad
        return velocity_new, velocity_new

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({
            "learning_rate": self.learning_rate,
            "momentum": self.momentum
        })
        return base_config

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return ['velocity']
