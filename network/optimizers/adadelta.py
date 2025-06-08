import numpy as np
from ..types import Tensor
from .base import Optimizer
from numba import njit


class AdaDelta(Optimizer):
    """
    Complex-valued AdaDelta optimizer.
    An adaptive learning rate method that maintains per-dimension learning rates using only first-order information.
    This version floors denominators with ε and replaces any NaN/Inf entries, ensuring no division-by-zero.
    If ε is None, it defaults to np.finfo(np.float32).eps.
    """

    def __init__(self, rho: float = 0.95, epsilon: float | None = None) -> None:
        super().__init__()
        self.rho = rho
        # Use float32 machine-ε if none provided
        self.epsilon = (
            epsilon
            if epsilon is not None
            else float(np.finfo(np.float32).eps)
        )

    def build(self, var_list: list[Tensor]) -> None:
        for var in var_list:
            self.add_slot(var, 'accumulated_grad')
            self.add_slot(var, 'accumulated_update')

    def update_step(self, grad: Tensor, var: Tensor) -> None:
        accumulated_grad = self.get_slot(var, 'accumulated_grad')
        accumulated_update = self.get_slot(var, 'accumulated_update')

        # Run the core update math (Numba-compiled)
        accumulated_grad_new, accumulated_update_new, var_update = self._update_step_math(
            accumulated_grad,
            accumulated_update,
            grad,
            self.rho,
            self.epsilon
        )

        # Write slots back in place
        accumulated_grad[...] = Tensor(accumulated_grad_new)
        accumulated_update[...] = Tensor(accumulated_update_new)

        # If var_update has NaN or Inf anywhere, zero out those entries
        if np.any(np.isnan(var_update)) or np.any(np.isinf(var_update)):
            var_update = np.where(
                np.isfinite(var_update),
                var_update,
                np.zeros_like(var_update)
            )

        var[...] -= Tensor(var_update)

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        accumulated_grad: np.ndarray,
        accumulated_update: np.ndarray,
        grad: np.ndarray,
        rho: float,
        epsilon: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core AdaDelta math (Numba-compiled). Floors denominators to ε to avoid division by zero.
        """
        # 1. squared magnitude of the gradient (real, ≥0)
        grad_sq = np.real(grad * np.conj(grad))

        # 2. update running average of squared gradients
        accumulated_grad_new = rho * accumulated_grad + (1.0 - rho) * grad_sq

        # 3. compute update term:
        #    sqrt(accumulated_update + ε) / ( sqrt(accumulated_grad_new + ε) + ε ) * grad
        sqrt_accum_update = np.sqrt(accumulated_update + epsilon)
        sqrt_accum_grad_new = np.sqrt(accumulated_grad_new + epsilon) + epsilon

        # Floor denominator magnitude to ε
        for idx in np.ndindex(sqrt_accum_grad_new.shape):
            val = sqrt_accum_grad_new[idx]
            if not np.isfinite(val) or np.abs(val) < epsilon:
                sqrt_accum_grad_new[idx] = epsilon

        update = (sqrt_accum_update / sqrt_accum_grad_new) * grad

        # 4. update running average of squared updates
        update_sq = np.real(update * np.conj(update))
        accumulated_update_new = rho * accumulated_update + (1.0 - rho) * update_sq

        # 5. guard against any NaN or Inf
        for idx in np.ndindex(accumulated_grad_new.shape):
            if not np.isfinite(accumulated_grad_new[idx]):
                accumulated_grad_new[idx] = 0.0
        for idx in np.ndindex(accumulated_update_new.shape):
            if not np.isfinite(accumulated_update_new[idx]):
                accumulated_update_new[idx] = 0.0
        for idx in np.ndindex(update.shape):
            if not np.isfinite(update[idx]):
                update[idx] = 0.0 + 0.0j

        return accumulated_grad_new, accumulated_update_new, update

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({
            "rho": self.rho,
            "epsilon": self.epsilon
        })
        return base_config

    @classmethod
    def get_slot_names(cls) -> list[str]:
        return ['accumulated_grad', 'accumulated_update']
