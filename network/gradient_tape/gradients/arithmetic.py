from typing import Tuple, Any, Union
import numpy as np
from .util import ensure_shape, complex_log

class ArithmeticGradients:
    @staticmethod
    def add(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_a_h = grad_output_h
        grad_b_h = grad_output_h

        return [
            ensure_shape(grad_a_h, np.shape(a)),
            ensure_shape(grad_b_h, np.shape(b))
        ]
        
    @staticmethod
    def subtract(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_a_h = grad_output_h
        grad_b_h = -grad_output_h

        return [
            ensure_shape(grad_a_h, np.shape(a)),
            ensure_shape(grad_b_h, np.shape(b))
        ]

    @staticmethod
    def multiply(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        # Holomorphic derivatives
        grad_a_h = grad_output_h * b
        grad_b_h = grad_output_h * a

        return [
            ensure_shape(grad_a_h, np.shape(a)),
            ensure_shape(grad_b_h, np.shape(b))
        ]
        
    @staticmethod
    def divide(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_a_h = grad_output_h / b
        grad_b_h = -grad_output_h * a / (b * b)

        return [
            ensure_shape(grad_a_h, np.shape(a)),
            ensure_shape(grad_b_h, np.shape(b))
        ]
        
    @staticmethod
    def floor_divide(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        grad_a_h = np.zeros_like(a)
        grad_b_h = np.zeros_like(b)

        return [
            ensure_shape(grad_a_h, np.shape(a)), 
            ensure_shape(grad_b_h, np.shape(b))
        ]
        
    @staticmethod
    def remainder(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        eps = np.finfo(a.dtype).eps
        safe_b = b + eps * (np.abs(b) < eps)

        val_floor = np.floor(np.divide(a, safe_b))

        grad_a_h = grad_output_h
        grad_b_h = -grad_output_h * val_floor

        return [
            ensure_shape(grad_a_h, np.shape(a)),
            ensure_shape(grad_b_h, np.shape(b))
        ]
        
    @staticmethod
    def power(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        base, exp = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        log_base = complex_log(base)

        grad_base = grad_output_h * exp * np.power(base, exp - 1)
        grad_exp = grad_output_h * np.power(base, exp) * log_base

        return [
            ensure_shape(grad_base, np.shape(base)),
            ensure_shape(grad_exp, np.shape(exp))
        ]
        
    @staticmethod
    def square(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z, = inputs

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_h = grad_output_h * (2 * z)

        return [
            ensure_shape(grad_h, np.shape(z))
        ]

    @staticmethod
    def float_power(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        x, y = inputs
        x_shape = np.shape(x)
        y_shape = np.shape(y)

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_x = grad_output_h * np.where(x != 0, y * x**(y - 1), 0)
        grad_y = grad_output_h * np.where(x > 0, x**y * np.log(x), 0) 


        return [
            ensure_shape(grad_x, x_shape), 
            ensure_shape(grad_y, y_shape)  
        ]
    
    @staticmethod
    def fmod(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs
        a_shape = np.shape(a)
        b_shape = np.shape(b)

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        with np.errstate(divide='ignore', invalid='ignore'):
            trunc_div_result = np.trunc(np.divide(a, b))
        
        grad_a = grad_output_h * np.ones_like(a, dtype=grad_output_h.dtype)
        grad_b = grad_output_h * (-trunc_div_result.astype(grad_output_h.dtype))

        return [
            ensure_shape(grad_a, a_shape),
            ensure_shape(grad_b, b_shape)
        ]
        
    @staticmethod
    def negative(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]
        z_shape = np.shape(z)

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        grad_h = -grad_output_h

        return [
            ensure_shape(grad_h, z_shape)
        ]

    @staticmethod
    def absolute(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        z = inputs[0]
        z_shape = np.shape(z)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        if np.iscomplexobj(z):
            abs_z = np.abs(z) + np.finfo(z.dtype).eps
            grad_h = grad_output_h * (np.conj(z) / (2 * abs_z))
            grad_ah = grad_output_ah * (z / (2 * abs_z))
        else:
            sgn = np.sign(z)
            grad_h = grad_output_h * sgn
            grad_ah = grad_output_ah * sgn

        return [
            (ensure_shape(grad_h, z_shape), ensure_shape(grad_ah, z_shape))
        ]
