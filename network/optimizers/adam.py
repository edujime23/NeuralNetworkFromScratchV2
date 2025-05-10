import numpy as np
from typing import List
from .base import Optimizer
from numba import njit, prange


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

    def build(self, var_list: List[np.typing.ArrayLike]) -> None:
        # Create 'm' and 'v' slots for each variable
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def update_step(self, grad: np.typing.ArrayLike, var: np.typing.ArrayLike) -> None:
        # Retrieve slots (momentum and second moment estimates)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Compute time step t (number of iterations)
        t = self.iterations + 1

        # Precompute bias correction factors to avoid repetitive calculations
        bias_correction_1 = 1 - self.beta_1 ** t
        bias_correction_2 = 1 - self.beta_2 ** t

        # Update step calculation in a separate, optimized function
        res = self._update_step_math(m=m, v=v, grad=grad, beta_1=self.beta_1,
                                      beta_2=self.beta_2, epsilon=self.epsilon,
                                      learning_rate=self.learning_rate,
                                      bias_correction_1=bias_correction_1,
                                      bias_correction_2=bias_correction_2)

        # Update first and second moment estimates (m and v)
        m[...] = res['m']
        v[...] = res['v']

        # Apply the parameter update (w)
        var[...] -= res['var']

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True, nogil=True, inline='always')
    def _update_step_math(
        m: np.typing.ArrayLike, v: np.typing.ArrayLike, grad: np.typing.ArrayLike,
        beta_1: float, beta_2: float, epsilon: float, learning_rate: float,
        bias_correction_1: float, bias_correction_2: float
    ):
        # Ensure arrays are contiguous for performance
        m = np.ascontiguousarray(m)
        v = np.ascontiguousarray(v)
        grad = np.ascontiguousarray(grad)

        # Compute complementary factors for beta values
        one_minus_beta1 = 1.0 - beta_1
        one_minus_beta2 = 1.0 - beta_2

        # Update biased first moment estimate (m)
        m_new = beta_1 * m + one_minus_beta1 * grad

        # Compute squared gradient (element-wise)
        grad_sq = np.real(grad * np.conj(grad))

        # Update biased second moment estimate (v)
        v_new = beta_2 * v + one_minus_beta2 * grad_sq

        # Compute bias-corrected first and second moment estimates
        m_hat = m_new / bias_correction_1
        v_hat = v_new / bias_correction_2

        # Compute parameter update value
        var_new = learning_rate * m_hat / ((v_hat)**1/2 + epsilon)

        # Return updated values for m, v, and parameter adjustment
        return {'m': m_new, 'v': v_new, 'var': var_new}

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return ['m', 'v']
