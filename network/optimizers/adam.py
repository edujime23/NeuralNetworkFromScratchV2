import numpy as np
from typing import List
from .base import Optimizer
from numba import njit


class AdamOptimizer(Optimizer):
    """
    Adam optimizer with bias-corrected first and second moment estimates.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list: List[np.ndarray]) -> None:
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def update_step(self, grad: np.ndarray, var: np.ndarray) -> None:
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        t = self.iterations + 1

        bias_correction_1 = 1 - self.beta_1 ** t
        bias_correction_2 = 1 - self.beta_2 ** t

        m_new, v_new, var_update = self._update_step_math(
            m, v, grad, self.beta_1, self.beta_2, self.epsilon,
            self.learning_rate, bias_correction_1, bias_correction_2
        )

        m[...] = m_new
        v[...] = v_new
        var[...] -= var_update

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        m, v, grad,
        beta_1, beta_2, epsilon,
        learning_rate, bias_correction_1, bias_correction_2
    ):
        one_minus_beta1 = 1.0 - beta_1
        one_minus_beta2 = 1.0 - beta_2

        m_new = beta_1 * m + one_minus_beta1 * grad
        grad_sq = np.real(grad * np.conj(grad))
        v_new = beta_2 * v + one_minus_beta2 * grad_sq

        m_hat = m_new / bias_correction_1
        v_hat = v_new / bias_correction_2

        var_update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        return m_new, v_new, var_update

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return ['m', 'v']