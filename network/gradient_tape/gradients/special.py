from typing import Tuple, Any, Union
import numpy as np
from .util import ensure_shape
import warnings

class SpecialGradients:
    @staticmethod
    def erf(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Compute the gradient of erf using the derivative formula
        grad = grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def erfc(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Compute the gradient of erfc using the derivative formula
        grad = -grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def cbrt(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output  # Only use holomorphic part
        else:
            grad_output_h = grad_output

        # Compute f'(z) = (1/3) * z^{-2/3}
        eps = np.finfo(z.dtype).eps
        grad_z_h = grad_output_h * (1/3) * np.power(z + eps, -2/3)

        grad_z_ah = np.zeros_like(z)

        return [
            (ensure_shape(grad_z_h, z.shape), ensure_shape(grad_z_ah, z.shape))
        ]
        
    @staticmethod
    def heaviside(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        z, k = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        # Small epsilon to approximate delta function behavior
        eps = np.finfo(z.dtype).eps ** 0.5
        abs_z = np.abs(z)
        abs_k = np.abs(k)

        # Approximate delta spike mask: where |z| is close to |k|
        delta_mask = np.abs(abs_z - abs_k) < eps

        z_unit = z / (abs_z + eps)
        z_conj_unit = np.conj(z) / (abs_z + eps)

        # Apply the formula from Wirtinger derivatives
        grad_z_h = 0.5 * grad_output_h * z_conj_unit * delta_mask
        grad_z_ah = 0.5 * grad_output_ah * z_unit * delta_mask

        # No gradients flow to k in the Heaviside(|z| - |k|) function
        grad_k_h = np.zeros_like(k)
        grad_k_ah = np.zeros_like(k)

        return [
            (ensure_shape(grad_z_h, z.shape), ensure_shape(grad_z_ah, z.shape)),
            (ensure_shape(grad_k_h, k.shape), ensure_shape(grad_k_ah, k.shape))
        ]
        
    @staticmethod
    def clip(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        z, min_val, max_val = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        # Split real and imaginary parts
        real_z, imag_z = np.real(z), np.imag(z)
        real_min, imag_min = np.real(min_val), np.imag(min_val)
        real_max, imag_max = np.real(max_val), np.imag(max_val)

        # Masks for active clipping per component
        mask_real_in = (real_z >= real_min) & (real_z <= real_max)
        mask_real_min = real_z < real_min
        mask_real_max = real_z > real_max

        mask_imag_in = (imag_z >= imag_min) & (imag_z <= imag_max)
        mask_imag_min = imag_z < imag_min
        mask_imag_max = imag_z > imag_max

        # Wirtinger gradient helpers
        grad_real = (1/2) * mask_real_in
        grad_imag = (1/2) * mask_imag_in

        # Gradients for z
        grad_z_h = grad_output_h * (grad_real - 1j * grad_imag)
        grad_z_ah = grad_output_ah * (grad_real + 1j * grad_imag)

        # Gradients for min_val (receive -grad where min is selected)
        grad_min_h = grad_output_h * (
            (1/2) * mask_real_min - 1j * (1/2) * mask_imag_min
        )
        grad_min_ah = grad_output_ah * (
            (1/2) * mask_real_min + 1j * (1/2) * mask_imag_min
        )

        # Gradients for max_val (receive -grad where max is selected)
        grad_max_h = grad_output_h * (
            (1/2) * mask_real_max - 1j * (1/2) * mask_imag_max
        )
        grad_max_ah = grad_output_ah * (
            (1/2) * mask_real_max + 1j * (1/2) * mask_imag_max
        )

        return [
            (ensure_shape(grad_z_h, z.shape), ensure_shape(grad_z_ah, z.shape)),
            (ensure_shape(-grad_min_h, min_val.shape), ensure_shape(-grad_min_ah, min_val.shape)),
            (ensure_shape(-grad_max_h, max_val.shape), ensure_shape(-grad_max_ah, max_val.shape)),
        ]

    @staticmethod
    def sqrt(
        grad_output: np.typing.NDArray[Any],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        inp = inputs[0]
        
        # For complex inputs, we want to ensure the input is treated as complex
        sqrt_inp = np.sqrt(inp + np.finfo(inp.dtype).eps)  # Avoid division by zero with eps

        # Gradients: f_z = 1 / (2 * sqrt(z)), and f_conj_z = 0 (anti-holomorphic part)
        grad_h = grad_output / (2 * sqrt_inp)
        grad_ah = np.zeros_like(grad_output)  # No contribution to the anti-holomorphic part

        return [
            (ensure_shape(grad_h, inp.shape), ensure_shape(grad_ah, inp.shape))
        ]

    @staticmethod
    def real(
        grad_output: np.typing.NDArray[Any],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        if not np.iscomplexobj(inputs[0]):
            return [(ensure_shape(grad_output[0], inputs[0].shape), np.zeros_like(grad_output[1]))]
        
        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        grad_h = grad_output_h * (1 / 2)
        grad_ah = grad_output_ah * (1 / 2)

        return [
            (ensure_shape(grad_h, inputs[0].shape), ensure_shape(grad_ah, inputs[0].shape))
        ]

    @staticmethod
    def imag(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        if not np.iscomplexobj(inputs[0]):
            return [
                (np.zeros_like(grad_output[0], dtype=grad_output[0].dtype), np.zeros_like(grad_output[1], dtype=grad_output[1].dtype))
                ]
        
        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)    
        
        grad_h = grad_output_h * (-1j / 2)
        grad_ah = grad_output_ah * (1j / 2)
        return [
            (ensure_shape(grad_h, inputs[0].shape), ensure_shape(grad_ah, inputs[0].shape))
        ]

    @staticmethod
    def sign(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        eps = np.finfo(z.dtype).eps
        abs_z = np.abs(z) + eps  # to avoid division by zero

        grad_h = grad_output_h * (0.5 / abs_z)
        grad_ah = grad_output_ah * (-0.5 * z**2) / (abs_z**3)

        return [
            (ensure_shape(grad_h, z.shape), ensure_shape(grad_ah, z.shape))
        ]

    @staticmethod
    def conjugate(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        if np.isrealobj(z):
            grad_h = grad_output_ah
            grad_ah = grad_output_h
        else:
            grad_h = np.zeros_like(z)
            grad_ah = grad_output_ah

        return [
            (ensure_shape(grad_h, z.shape), ensure_shape(grad_ah, z.shape))
        ]