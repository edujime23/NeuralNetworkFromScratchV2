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

        # Promote to a float‐type so cos(z) isn’t integer‐division:
        work_dtype = np.result_type(z.dtype, np.float32)
        zf = z.astype(work_dtype, copy=False)

        # Add global epsilon to avoid division by zero
        cos_z = np.cos(zf)
        denom = cos_z * cos_z + epsilon

        dz = grad_output_h * (1.0 / denom)
        return [dz]

    @staticmethod
    def arcsin(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        work_dtype = np.result_type(z.dtype, np.float32)
        zf = z.astype(work_dtype, copy=False)

        inside = 1.0 - zf * zf
        safe_inside = inside + epsilon

        dz = grad_output_h / np.sqrt(safe_inside)
        return [dz]

    @staticmethod
    def arccos(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        work_dtype = np.result_type(z.dtype, np.float32)
        zf = z.astype(work_dtype, copy=False)

        inside = 1.0 - zf * zf
        safe_inside = inside + epsilon

        dz = -grad_output_h / np.sqrt(safe_inside)
        return [dz]

    @staticmethod
    def arctan(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        work_dtype = np.result_type(z.dtype, np.float32)
        zf = z.astype(work_dtype, copy=False)

        dz = grad_output_h / (1.0 + zf * zf)
        return [dz]

    @staticmethod
    def arctan2(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        y, x = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )

        work_dtype = np.result_type(x.dtype, y.dtype, np.complex64)
        xf = x.astype(work_dtype, copy=False)
        yf = y.astype(work_dtype, copy=False)

        w = xf + 1j * yf
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

        work_dtype = np.result_type(x.dtype, np.float32)
        xf = x.astype(work_dtype, copy=False)
        pi_x = np.pi * xf

        numerator = np.cos(pi_x) * (np.pi * xf) - np.sin(pi_x)
        denom = np.pi * xf * xf
        grad_val = np.where(xf != 0, numerator / (denom + epsilon), 0.0)

        grad_x_h = grad_output_h * grad_val
        return [grad_x_h]

    @staticmethod
    def hypot(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        z1, z2 = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )

        work_dtype = np.result_type(z1.dtype, z2.dtype, np.float64)
        z1f = z1.astype(work_dtype, copy=False)
        z2f = z2.astype(work_dtype, copy=False)

        norm_sq = np.abs(z1f) * np.abs(z1f) + np.abs(z2f) * np.abs(z2f)
        safe_norm = norm_sq + epsilon
        r = np.sqrt(safe_norm)

        grad_z1_h  = grad_output_h  * (np.conj(z1f) / (2.0 * r))
        grad_z1_ah = grad_output_ah * (     z1f   / (2.0 * r))
        grad_z2_h  = grad_output_h  * (np.conj(z2f) / (2.0 * r))
        grad_z2_ah = grad_output_ah * (     z2f   / (2.0 * r))

        return [
            (grad_z1_h, grad_z1_ah),
            (grad_z2_h, grad_z2_ah),
        ]