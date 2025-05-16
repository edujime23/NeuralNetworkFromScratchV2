import warnings
from .util import ensure_shape
from typing import Any, Tuple, Union
import numpy as np

class TrigonometricGradients:
    @staticmethod
    def sin(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_h = grad_output_h * np.cos(z)

        return [
            (ensure_shape(grad_h, z.shape), np.zeros_like(z))
        ]
    
    @staticmethod
    def cos(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        dz = grad_output_h * (-np.sin(z))

        return [
            (ensure_shape(dz, z.shape), np.zeros_like(z))
        ]
    
    @staticmethod
    def tan(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        dz = grad_output_h * (1 / np.cos(z) ** 2)

        return [
            (ensure_shape(dz, z.shape), np.zeros_like(z))
        ]
    
    @staticmethod
    def arcsin(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        dz = grad_output_h / np.sqrt(1 - z**2)

        return [
            (ensure_shape(dz, z.shape), np.zeros_like(z))
        ]

    @staticmethod
    def arccos(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        # ∂/∂z arccos(z) = -1 / sqrt(1 - z^2)
        dz = -grad_output_h / np.sqrt(1 - z**2)

        return [
            (ensure_shape(dz, z.shape), np.zeros_like(z))
        ]

    @staticmethod
    def arctan(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        z = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        dz = grad_output_h / (1 + z**2)

        return [
            (ensure_shape(dz, z.shape), np.zeros_like(z))
        ]
    
    @staticmethod
    def arctan2(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        y, x = inputs

        # Handle possible tuple grad_output for holo and anti-holo parts
        if isinstance(grad_output, tuple):
            grad_out_h, _ = grad_output
        else:
            grad_out_h = grad_output

        # Combine into complex argument w = x + i y
        w = x + 1j * y
        inv_w = 1.0 / w

        # Holomorphic gradient components
        grad_y_h = grad_out_h * np.real(inv_w)
        grad_x_h = grad_out_h * np.imag(inv_w)

        return [
            (ensure_shape(grad_y_h, y.shape), np.zeros_like(y)),
            (ensure_shape(grad_x_h, x.shape), np.zeros_like(x))
        ]
        
    @staticmethod
    def sinc(grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]], inputs: Tuple[np.typing.NDArray[Any]]):
        x = inputs[0]

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        # Calculate the sinc function gradient using the formula
        pi_x = np.pi * x
        grad_val = np.where(x != 0,
                            (np.cos(pi_x) * np.pi * x - np.sin(pi_x)) / (np.pi * x**2),
                            0)  # Handle x == 0 case

        # Gradients for sinc function (holomorphic and anti-holomorphic parts)
        grad_x_h = grad_output_h * grad_val

        # Ensure the result has the same shape as the input
        return [
            (ensure_shape(grad_x_h, x.shape), np.zeros_like(x))
        ]
        
    @staticmethod
    def hypot(grad_output: np.ndarray,
            inputs: Tuple[np.ndarray, np.ndarray]):
        z1, z2 = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output

        norm_sq = np.abs(z1)**2 + np.abs(z2)**2 + np.finfo(z1.dtype).eps
        r = np.sqrt(norm_sq)

        grad_z1_h  = grad_output_h  * (np.conj(z1) / (2 * r))
        grad_z1_ah = grad_output_ah * (z1 / (2 * r))

        grad_z2_h  = grad_output_h  * (np.conj(z2) / (2 * r))
        grad_z2_ah = grad_output_ah * (z2 / (2 * r))

        return [
            (ensure_shape(grad_z1_h, z1.shape), ensure_shape(grad_z1_ah, z1.shape)),
            (ensure_shape(grad_z2_h, z2.shape), ensure_shape(grad_z2_ah, z2.shape))
        ]