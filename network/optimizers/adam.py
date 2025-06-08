import numpy as np
from numba import njit

from ..types import Tensor
from .base import Optimizer


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
        epsilon: float | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # If epsilon is None, use float32 machine epsilon
        self.epsilon = (
            epsilon if epsilon is not None else float(np.finfo(np.float32).eps)
        )

    def build(self, var_list: list[Tensor]) -> None:
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")

    def update_step(self, grad: Tensor, var: Tensor) -> None:
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        t = self.iterations + 1
        bias_correction_1 = 1 - self.beta_1**t
        bias_correction_2 = 1 - self.beta_2**t
        m_new, v_new, var_update = self._update_step_math(
            m.numpy,
            v.numpy,
            grad.data,
            self.beta_1,
            self.beta_2,
            self.epsilon,
            self.learning_rate,
            bias_correction_1,
            bias_correction_2,
        )

        m[...] = Tensor(m_new)
        v[...] = Tensor(v_new)
        # Prevent NaN/Inf in update
        if np.any(np.isnan(var_update)) or np.any(np.isinf(var_update)) or np.any(np.isneginf(var_update)):
            var_update = np.zeros_like(var_update)
        var[...] -= Tensor(var_update)

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
        bias_correction_2: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        mask = ~np.isfinite(denom) | (np.abs(denom) < epsilon)
        denom = np.where(mask, epsilon, denom)

        var_update = learning_rate * m_hat / denom
        # Guard against NaN/Inf in var_update
        var_update = learning_rate * m_hat / denom
        mask = ~np.isfinite(var_update)
        var_update = np.where(mask, 0.0, var_update)

        return m_new, v_new, var_update

    @classmethod
    def get_slot_names(cls) -> list[str]:
        return ["m", "v"]
