import numpy as np
from typing import List, Tuple, Optional
from .base import Optimizer
from numba import njit


class AdamOptimizer(Optimizer):
    """
    Adam optimizer with bias-corrected first and second moment estimates.
    This implementation floors denominators with epsilon and guards against NaNs/Infs.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: Optional[float] = None
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # If epsilon is None, use float32 machine epsilon
        self.epsilon = epsilon if epsilon is not None else float(np.finfo(np.float32).eps)

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
            m.numpy, v.numpy, grad, self.beta_1, self.beta_2,
            self.epsilon, self.learning_rate,
            bias_correction_1, bias_correction_2
        )

        m[...] = m_new
        v[...] = v_new
        # Prevent NaN/Inf in update
        if np.any(np.isnan(var_update)) or np.any(np.isinf(var_update)):
            var_update = np.zeros_like(var_update)
        var[...] -= var_update

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        m: np.ndarray,
        v: np.ndarray,
        grad: np.ndarray,
        beta_1: float,
        beta_2: float,
        epsilon: float,
        learning_rate: float,
        bias_correction_1: float,
        bias_correction_2: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core Adam update step. Floors denominator to epsilon to avoid division by zero.
        """
        one_minus_beta1 = 1.0 - beta_1
        one_minus_beta2 = 1.0 - beta_2

        m_new = beta_1 * m + one_minus_beta1 * grad
        grad_sq = np.real(grad * np.conj(grad))
        v_new = beta_2 * v + one_minus_beta2 * grad_sq

        m_hat = m_new / bias_correction_1
        v_hat = v_new / bias_correction_2

        # Compute denominator and floor it using magnitude comparison
        denom = np.sqrt(v_hat) + epsilon
        for idx in np.ndindex(denom.shape):
            val = denom[idx]
            if not np.isfinite(val) or np.abs(val) < epsilon:
                denom[idx] = epsilon

        var_update = learning_rate * m_hat / denom
        # Guard against NaN/Inf in var_update
        for idx in np.ndindex(var_update.shape):
            if not np.isfinite(var_update[idx]):
                var_update[idx] = 0.0 + 0.0j

        return m_new, v_new, var_update

    @classmethod
    def get_slot_names(cls) -> List[str]:
        return ['m', 'v']
