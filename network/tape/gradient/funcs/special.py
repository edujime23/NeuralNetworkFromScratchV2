import numpy as np
from .util import epsilon
from ....types import Tensor

class SpecialGradients:
    @staticmethod
    def erf(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (inp,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad = grad_output_h * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [grad]

    @staticmethod
    def erfc(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (inp,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad = -grad_output_h * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [grad]

    @staticmethod
    def cbrt(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_z_h = grad_output_h * (1/3) * np.power(z + epsilon, -2/3)
        grad_z_ah = np.zeros_like(z)
        return [(grad_z_h, grad_z_ah)]

    @staticmethod
    def heaviside(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        z, k = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )
        abs_z = np.abs(z)
        abs_k = np.abs(k)
        delta_mask = np.abs(abs_z - abs_k) < epsilon
        z_unit = z / (abs_z + epsilon)
        z_conj_unit = np.conj(z) / (abs_z + epsilon)
        grad_z_h = 0.5 * grad_output_h * z_conj_unit * delta_mask
        grad_z_ah = 0.5 * grad_output_ah * z_unit * delta_mask
        grad_k_h = np.zeros_like(k)
        grad_k_ah = np.zeros_like(k)
        return [(grad_z_h, grad_z_ah), (grad_k_h, grad_k_ah)]

    @staticmethod
    def clip(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        z, min_val, max_val = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )
        real_z, imag_z = np.real(z), np.imag(z)
        real_min, imag_min = np.real(min_val), np.imag(min_val)
        real_max, imag_max = np.real(max_val), np.imag(max_val)
        mask_real_in = (real_z >= real_min) & (real_z <= real_max)
        mask_real_min = real_z < real_min
        mask_real_max = real_z > real_max
        mask_imag_in = (imag_z >= imag_min) & (imag_z <= imag_max)
        mask_imag_min = imag_z < imag_min
        mask_imag_max = imag_z > imag_max
        grad_real = 0.5 * mask_real_in
        grad_imag = 0.5 * mask_imag_in
        grad_z_h = grad_output_h * (grad_real - 1j * grad_imag)
        grad_z_ah = grad_output_ah * (grad_real + 1j * grad_imag)
        grad_min_h = grad_output_h * (0.5 * mask_real_min - 1j * 0.5 * mask_imag_min)
        grad_min_ah = grad_output_ah * (0.5 * mask_real_min + 1j * 0.5 * mask_imag_min)
        grad_max_h = grad_output_h * (0.5 * mask_real_max - 1j * 0.5 * mask_imag_max)
        grad_max_ah = grad_output_ah * (0.5 * mask_real_max + 1j * 0.5 * mask_imag_max)
        return [
            (grad_z_h, grad_z_ah),
            (-grad_min_h, -grad_min_ah),
            (-grad_max_h, -grad_max_ah)
        ]

    @staticmethod
    def sqrt(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (inp,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        sqrt_inp = np.sqrt(inp + epsilon)
        grad_h = grad_output_h / (sqrt_inp * 2)
        grad_ah = np.zeros_like(grad_output_h)
        return [(grad_h, grad_ah)]

    @staticmethod
    def real(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )
        grad_h = grad_output_h * 0.5
        grad_ah = grad_output_ah * 0.5
        return [(grad_h, grad_ah)]

    @staticmethod
    def imag(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )
        grad_h = grad_output_h * (-1j / 2)
        grad_ah = grad_output_ah * (1j / 2)
        return [(grad_h, grad_ah)]

    @staticmethod
    def sign(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_output_ah = (
            grad_output[1] if isinstance(grad_output, tuple) else np.zeros_like(grad_output_h)
        )
        abs_z = np.abs(z) + epsilon
        grad_h = grad_output_h * (0.5 / abs_z)
        grad_ah = grad_output_ah * (-0.5 * z**2) / (abs_z**3)
        return [(grad_h, grad_ah)]

    @staticmethod
    def conjugate(grad_output: Tensor | tuple[Tensor, Tensor], inputs: tuple[Tensor, ...]):
        (z,) = inputs
        grad_output_h = (
            grad_output[0] if isinstance(grad_output, tuple) else grad_output
        )
        grad_h = np.zeros_like(z) if np.iscomplexobj(z) else grad_output_h
        grad_ah = grad_output_h if np.iscomplexobj(z) else np.zeros_like(z)
        return [(grad_h, grad_ah)]