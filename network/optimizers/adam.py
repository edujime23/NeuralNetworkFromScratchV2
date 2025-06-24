from itertools import product

import numpy as np
from numba import njit, prange

from network.types.tensor import Tensor
from network.types.variable import Variable

from .base import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer with bias-corrected first and second moment estimates.
    This implementation floors denominators with epsilon and guards against NaNs/Infs.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float | None = None,
    ) -> None:
        super().__init__(lr=lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon if epsilon is not None else np.finfo(np.float32).eps

    def build(
        self, var_list: list[Variable], dtypes: list[np.dtype] | None = None
    ) -> None:
        for var, dtype in product(var_list, dtypes):
            self.add_slot(var, "m", dtype)
            self.add_slot(var, "v", dtype)

    def update_step(
        self, gradient: Tensor, variable: Variable, slots: dict[str, Variable]
    ) -> Tensor:
        m = slots["m"]
        v = slots["v"]

        t = self.iterations + 1
        bc1 = 1 - self.beta_1**t
        bc2 = 1 - self.beta_2**t

        # Call the generic-rank helper (no explicit signature)
        m_new, v_new, var_update = self._update_step_math(
            m.numpy,
            v.numpy,
            gradient.numpy,
            self.beta_1,
            self.beta_2,
            self.epsilon,
            self.lr,
            bc1,
            bc2,
        )

        # Write results back into the slot Variables
        m.assign(m_new)
        v.assign(v_new)
        return Tensor(var_update)

    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        m: np.ndarray,
        v: np.ndarray,
        grad: np.ndarray,
        beta_1: float,
        beta_2: float,
        epsilon: float,
        lr: float,
        bias_correction_1: float,
        bias_correction_2: float,
    ):
        """
        Parallelized, Numba-compiled Adam step.
        - m, v, grad are ND arrays of identical shape.
        - Returns (m_new, v_new, var_delta) arrays, each same shape as inputs.

        Uses prange to parallelize across all elements.
        """
        # Number of total elements
        n = m.size

        # Allocate output arrays
        m_new = np.empty_like(m)
        v_new = np.empty_like(v)
        var_delta = np.empty_like(grad, dtype=m.dtype)

        # View as 1D for parallel loop
        m_flat = m.reshape(n)
        v_flat = v.reshape(n)
        g_flat = grad.reshape(n)
        m_new_flat = m_new.reshape(n)
        v_new_flat = v_new.reshape(n)
        d_flat = var_delta.reshape(n)

        # Precompute constants
        one_minus_beta1 = 1.0 - beta_1
        one_minus_beta2 = 1.0 - beta_2

        for i in prange(n):
            # 1) Update biased first moment
            m_i = beta_1 * m_flat[i] + one_minus_beta1 * g_flat[i]

            gi = g_flat[i]
            grad_sq = np.real(gi * np.conj(gi))
            v_i = beta_2 * v_flat[i] + one_minus_beta2 * grad_sq

            # 3) Bias-corrected estimates
            m_hat = m_i / bias_correction_1
            v_hat = v_i / bias_correction_2

            # 4) Denominator with epsilon floor
            denom = np.sqrt(v_hat) + epsilon
            # If denom is NaN/Inf or below epsilon, floor to epsilon
            if not np.isfinite(denom) or np.abs(denom) < epsilon:
                denom = epsilon

            # 5) Compute parameter delta
            delta = lr * m_hat / denom

            # 6) Guard against NaN/Inf in delta
            if not np.isfinite(delta):
                delta = 0.0

            # Write back into flat outputs
            m_new_flat[i] = m_i
            v_new_flat[i] = v_i
            d_flat[i] = delta

        return m_new, v_new, var_delta

    @classmethod
    def get_slot_names(cls) -> list[str]:
        return ["m", "v"]
