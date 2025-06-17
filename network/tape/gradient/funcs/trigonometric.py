import numpy as np

from ....types import Tensor
from .util import epsilon


class TrigonometricGradients:
    @staticmethod
    def sin(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_h = grad_output_h * np.cos(z)
        return [grad_h]

    @staticmethod
    def cos(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        dz = grad_output_h * -np.sin(z)
        return [dz]

    @staticmethod
    def tan(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        # Add global epsilon to avoid division by zero
        cos_z = np.cos(z)
        denom = cos_z * cos_z + epsilon

        dz = grad_output_h * (1.0 / denom)
        return [dz]

    @staticmethod
    def arcsin(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        inside = 1.0 - z * z
        safe_inside = inside + epsilon

        dz = grad_output_h / np.sqrt(safe_inside)
        return [dz]

    @staticmethod
    def arccos(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        inside = 1.0 - (z**2)
        safe_inside = inside + epsilon

        dz = -grad_output_h / np.sqrt(safe_inside)
        return [dz]

    @staticmethod
    def arctan(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        dz = grad_output_h / (1.0 + z * z)
        return [dz]

    @staticmethod
    def arctan2(
        grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]
    ):
        y, x = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        w = x + 1j * y
        safe_w = w + (epsilon + 0j)

        inv_w = 1.0 / safe_w
        grad_y_h = grad_output_h * np.real(inv_w)
        grad_x_h = grad_output_h * np.imag(inv_w)
        return [grad_y_h, grad_x_h]

    @staticmethod
    def sinc(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (x,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        pi_x = np.pi * x

        numerator = np.cos(pi_x) * (np.pi * x) - np.sin(pi_x)
        denom = np.pi * x * x
        grad_val = np.where(x != 0, numerator / (denom + epsilon), 0.0)

        grad_x_h = grad_output_h * grad_val
        return [grad_x_h]

    @staticmethod
    def hypot(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        z1, z2 = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1]
            if isinstance(grad_output, tuple)
            else np.zeros_like(grad_output_h)
        )

        norm_sq = np.abs(z1) * np.abs(z1) + np.abs(z2) * np.abs(z2)
        safe_norm = norm_sq + epsilon
        r = np.sqrt(safe_norm)

        grad_z1_h = grad_output_h * (np.conj(z1) / (2.0 * r))
        grad_z1_ah = grad_output_ah * (z1 / (2.0 * r))
        grad_z2_h = grad_output_h * (np.conj(z2) / (2.0 * r))
        grad_z2_ah = grad_output_ah * (z2 / (2.0 * r))

        return [
            (grad_z1_h, grad_z1_ah),
            (grad_z2_h, grad_z2_ah),
        ]
