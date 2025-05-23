from typing import Tuple, Any, Union
import numpy as np
from .util import ensure_shape
import warnings

class SpecialGradients:
    @staticmethod
    def erf(grad_output: np.typing.NDArray[Any], inputs: Tuple[np.typing.NDArray[Any]]):
        inp = inputs[0]
        grad = grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [ensure_shape(grad, np.shape(inp))]

    @staticmethod
    def erfc(grad_output: np.typing.NDArray[Any], inputs: Tuple[np.typing.NDArray[Any]]):
        inp = inputs[0]
        grad = -grad_output * (2 / np.sqrt(np.pi)) * np.exp(-np.conjugate(inp)**2)
        return [ensure_shape(grad, np.shape(inp))]

    @staticmethod
    def cbrt(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        eps = np.finfo(z.dtype).eps
        grad_z_h = grad_output_h * (1/3) * np.power(z + eps, -2/3)
        grad_z_ah = np.zeros_like(z)
        return [(ensure_shape(grad_z_h, np.shape(z)), ensure_shape(grad_z_ah, np.shape(z)))]

    @staticmethod
    def heaviside(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        z, k = inputs
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(grad_output))
        eps = np.finfo(z.dtype).eps ** 0.5
        abs_z = np.abs(z)
        abs_k = np.abs(k)
        delta_mask = np.abs(abs_z - abs_k) < eps
        z_unit = z / (abs_z + eps)
        z_conj_unit = np.conj(z) / (abs_z + eps)
        grad_z_h = 0.5 * grad_output_h * z_conj_unit * delta_mask
        grad_z_ah = 0.5 * grad_output_ah * z_unit * delta_mask
        grad_k_h = np.zeros_like(k)
        grad_k_ah = np.zeros_like(k)
        return [
            (ensure_shape(grad_z_h, np.shape(z)), ensure_shape(grad_z_ah, np.shape(z))),
            (ensure_shape(grad_k_h, np.shape(k)), ensure_shape(grad_k_ah, np.shape(k)))
        ]

    @staticmethod
    def clip(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        z, min_val, max_val = inputs
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(grad_output))
        real_z, imag_z = np.real(z), np.imag(z)
        real_min, imag_min = np.real(min_val), np.imag(min_val)
        real_max, imag_max = np.real(max_val), np.imag(max_val)
        mask_real_in = (real_z >= real_min) & (real_z <= real_max)
        mask_real_min = real_z < real_min
        mask_real_max = real_z > real_max
        mask_imag_in = (imag_z >= imag_min) & (imag_z <= imag_max)
        mask_imag_min = imag_z < imag_min
        mask_imag_max = imag_z > imag_max
        grad_real = (1/2) * mask_real_in
        grad_imag = (1/2) * mask_imag_in
        grad_z_h = grad_output_h * (grad_real - 1j * grad_imag)
        grad_z_ah = grad_output_ah * (grad_real + 1j * grad_imag)
        grad_min_h = grad_output_h * ((1/2) * mask_real_min - 1j * (1/2) * mask_imag_min)
        grad_min_ah = grad_output_ah * ((1/2) * mask_real_min + 1j * (1/2) * mask_imag_min)
        grad_max_h = grad_output_h * ((1/2) * mask_real_max - 1j * (1/2) * mask_imag_max)
        grad_max_ah = grad_output_ah * ((1/2) * mask_real_max + 1j * (1/2) * mask_imag_max)
        return [
            (ensure_shape(grad_z_h, np.shape(z)), ensure_shape(grad_z_ah, np.shape(z))),
            (ensure_shape(-grad_min_h, np.shape(min_val)), ensure_shape(-grad_min_ah, np.shape(min_val))),
            (ensure_shape(-grad_max_h, np.shape(max_val)), ensure_shape(-grad_max_ah, np.shape(max_val)))
        ]

    @staticmethod
    def sqrt(grad_output: np.typing.NDArray[Any], inputs: Tuple[np.typing.NDArray[Any]]):
        inp = inputs[0]
        sqrt_inp = np.sqrt(inp + np.finfo(inp.dtype).eps)
        grad_h = grad_output / (2 * sqrt_inp)
        grad_ah = np.zeros_like(grad_output)
        return [(ensure_shape(grad_h, np.shape(inp)), ensure_shape(grad_ah, np.shape(inp)))]

    @staticmethod
    def real(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(grad_output))
        grad_h = grad_output_h * 0.5
        grad_ah = grad_output_ah * 0.5
        return [(ensure_shape(grad_h, np.shape(z)), ensure_shape(grad_ah, np.shape(z)))]

    @staticmethod
    def imag(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(grad_output))
        grad_h = grad_output_h * (-1j / 2)
        grad_ah = grad_output_ah * (1j / 2)
        return [(ensure_shape(grad_h, np.shape(z)), ensure_shape(grad_ah, np.shape(z)))]

    @staticmethod
    def sign(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]
        grad_output_h, grad_output_ah = grad_output if isinstance(grad_output, tuple) else (grad_output, np.zeros_like(grad_output))
        eps = np.finfo(z.dtype).eps
        abs_z = np.abs(z) + eps
        grad_h = grad_output_h * (0.5 / abs_z)
        grad_ah = grad_output_ah * (-0.5 * z**2) / (abs_z**3)
        return [(ensure_shape(grad_h, np.shape(z)), ensure_shape(grad_ah, np.shape(z)))]

    @staticmethod
    def conjugate(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        grad_h = np.zeros_like(z) if np.iscomplexobj(z) else grad_output_h
        grad_ah = grad_output_h if np.iscomplexobj(z) else np.zeros_like(z)
        return [(ensure_shape(grad_h, np.shape(z)), ensure_shape(grad_ah, np.shape(z)))]
