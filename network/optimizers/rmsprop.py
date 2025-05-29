import numpy as np
from typing import List, Tuple
from .base import Optimizer
from numba import njit


class RMSProp(Optimizer):
    """
    Complex-valued RMSProp optimizer.
    Maintains a moving average of the squared gradients to normalize the gradient updates.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-7
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def build(self, var_list: List[np.ndarray]) -> None:
        for var in var_list:
            self.add_slot(var, 'avg_sq_grad')

    def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
        avg_sq_grad = self.get_slot(var, 'avg_sq_grad')

        avg_sq_grad_new, var_update = self._update_step_math(
            avg_sq_grad, grad, self.rho, self.epsilon, self.learning_rate
        )

        avg_sq_grad[...] = avg_sq_grad_new
        var[...] -= var_update

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        avg_sq_grad: np.ndarray,
        grad: np.ndarray,
        rho: float,
        epsilon: float,
        learning_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad_sq = np.real(grad * np.conj(grad))
        avg_sq_grad_new = rho * avg_sq_grad + (1.0 - rho) * grad_sq
        var_update = learning_rate * grad / (np.sqrt(avg_sq_grad_new) + epsilon)
        return avg_sq_grad_new, var_update

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({
            "learning_rate": self.learning_rate,
            "rho": self.rho,
            "epsilon": self.epsilon
        })
        return base_config

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return ['avg_sq_grad']
