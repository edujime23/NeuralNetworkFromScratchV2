import numpy as np
from .base import Optimizer
from ..types import Tensor
from numba import njit


class AdaGrad(Optimizer):
    """
    Complex-valued AdaGrad optimizer.
    Scales learning rates inversely proportional to the square root of all historical squared gradients.
    This implementation guards against NaNs and division‐by‐zero by flooring the denominator with epsilon.
    If epsilon is None, it defaults to np.finfo(np.float32).eps.
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float | None = None) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        # If no epsilon provided, use a safe float32 machine epsilon
        self.epsilon = (
            epsilon
            if epsilon is not None
            else float(np.finfo(np.float32).eps)
        )

    def build(self, var_list: list[Tensor]) -> None:
        for var in var_list:
            self.add_slot(var, 'accumulated_grad')

    def update_step(self, grad: Tensor, var: Tensor) -> None:
        accumulated_grad = self.get_slot(var, 'accumulated_grad')

        accumulated_grad_new, var_update = self._update_step_math(
            accumulated_grad.numpy,
            grad.numpy,
            self.learning_rate,
            self.epsilon
        )

        # Write updated accumulator back in place
        accumulated_grad[...] = Tensor(accumulated_grad_new)

        # Prevent NaN/Inf from propagating into var
        if np.any(np.isnan(var_update)) or np.any(np.isinf(var_update)):
            var_update = np.zeros_like(var_update)

        var[...] -= Tensor(var_update)

    @staticmethod
    @njit(fastmath=True, cache=True, nogil=True)
    def _update_step_math(
        accumulated_grad: np.ndarray,
        grad: np.ndarray,
        learning_rate: float,
        epsilon: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Core AdaGrad math (Numba‐compiled). Denominator = sqrt(accumulated_grad_new + ε) + ε
        Floors the denominator with epsilon to avoid division by zero or underflow.
        """

        # Compute squared magnitude of gradient (non‐negative real array)
        grad_sq = np.real(grad * np.conj(grad))

        # Update running sum of squared gradients
        accumulated_grad_new = accumulated_grad + grad_sq

        # Compute denominator: sqrt(accumulated_grad_new + ε) + ε
        sqrt_accum = np.sqrt(accumulated_grad_new + epsilon)
        denom = sqrt_accum + epsilon

        # Prevent any zero or NaN in denom
        # (We compare absolute value because denom may be complex if something went wrong,
        #  but in principle √ of a nonnegative real should be real.)
        for idx in np.ndindex(denom.shape):
            val = denom[idx]
            # If magnitude is less than ε or not finite, set to ε (real)
            if not np.isfinite(val) or np.abs(val) < epsilon:
                denom[idx] = epsilon

        # Compute the update
        var_update = learning_rate * grad / denom

        # As a safety measure, replace any NaN/Inf in var_update with zero
        for idx in np.ndindex(var_update.shape):
            if not np.isfinite(var_update[idx]):
                var_update[idx] = 0.0 + 0.0j

        return accumulated_grad_new, var_update

    def get_config(self) -> dict:
        base_config = super().get_config()
        base_config.update({
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon
        })
        return base_config

    @classmethod
    def get_slot_names(cls) -> list[str]:
        return ['accumulated_grad']
