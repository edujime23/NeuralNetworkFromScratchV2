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

        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)

        return [
            (ensure_shape(grad_a_h, np.shape(a)), ensure_shape(grad_a_ah, np.shape(a))),
            (ensure_shape(grad_b_h, np.shape(b)), ensure_shape(grad_b_ah, np.shape(b))),
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

        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)

        return [
            (ensure_shape(grad_a_h, np.shape(a)), ensure_shape(grad_a_ah, np.shape(a))),
            (ensure_shape(grad_b_h, np.shape(b)), ensure_shape(grad_b_ah, np.shape(b))),
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

        # Anti-holomorphic derivatives are zero
        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)

        return [
            (ensure_shape(grad_a_h, np.shape(a)), ensure_shape(grad_a_ah, np.shape(a))),
            (ensure_shape(grad_b_h, np.shape(b)), ensure_shape(grad_b_ah, np.shape(b))),
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
        
        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)


        return [
            (ensure_shape(grad_a_h, np.shape(a)), grad_a_ah),
            (ensure_shape(grad_b_h, np.shape(b)), grad_b_ah)
        ]
        
    @staticmethod
    def floor_divide(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any], np.typing.NDArray[Any]]
    ):
        a, b = inputs

        grad_a_h = np.zeros_like(a)
        grad_b_h = np.zeros_like(b)

        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)

        return [
            (grad_a_h, grad_a_ah),
            (grad_b_h, grad_b_ah),
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

        grad_a_ah = np.zeros_like(a)
        grad_b_ah = np.zeros_like(b)

        return [
            (ensure_shape(grad_a_h, np.shape(a)), grad_a_ah),
            (ensure_shape(grad_b_h, np.shape(b)), grad_b_ah),
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

        base_safe = base

        log_base = complex_log(base_safe)

        grad_base = grad_output_h * exp * np.power(base_safe, exp - 1)
        grad_exp = grad_output_h * np.power(base_safe, exp) * log_base

        grad_base_ah = np.zeros_like(base)
        grad_exp_ah = np.zeros_like(exp)

        return [
            (ensure_shape(grad_base, np.shape(base)), grad_base_ah),
            (ensure_shape(grad_exp, np.shape(exp)), grad_exp_ah),
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


        return [(ensure_shape(grad_h, np.shape(z)), np.zeros_like(z))]

    @staticmethod
    def float_power(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        x, y = inputs
        x_shape = np.shape(x)
        y_shape = np.shape(y)

        if isinstance(grad_output, tuple):
            grad_output_h, grad_output_ah = grad_output
        else:
            grad_output_h = grad_output
            grad_output_ah = np.zeros_like(grad_output)

        grad_x = grad_output_h * np.where(x != 0, y * x**(y - 1), 0)
        
        grad_y = grad_output_h * np.where(x > 0, x**y * np.log(x), 0) 

        return [
            (ensure_shape(grad_x, x_shape), grad_output_ah), 
            (ensure_shape(grad_y, y_shape), grad_output_ah)  
        ]
    
    @staticmethod
    def fmod(
        grad_output: Union[np.typing.NDArray[Any], Tuple[np.typing.NDArray[Any]]],
        inputs: Tuple[np.typing.NDArray[Any]]
    ):
        a, b = inputs
        a_shape = np.shape(a) if hasattr(a, 'shape') else ()
        b_shape = np.shape(b) if hasattr(b, 'shape') else ()

        if isinstance(grad_output, tuple):
            grad_output_h, _ = grad_output
        else:
            grad_output_h = grad_output

        with np.errstate(divide='ignore', invalid='ignore'):
            trunc_div_result = np.trunc(np.divide(a, b))
        
        grad_z = grad_output_h * np.ones_like(a, dtype=grad_output_h.dtype)

        grad_zh = np.zeros_like(a, dtype=grad_output_h.dtype)

        grad_b = grad_output_h * (-trunc_div_result.astype(grad_output_h.dtype))

        return [
            (ensure_shape(grad_z, a_shape), grad_zh),
            (ensure_shape(grad_b, b_shape), grad_zh)
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
        grad_ah = np.zeros_like(z)

        return [
            (ensure_shape(grad_h, z_shape), grad_ah)
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

        abs_z = np.abs(z)
        safe_abs = np.where(abs_z == 0, 1.0, abs_z)

        if np.iscomplexobj(z):
            grad_h = grad_output_h * (np.conj(z) / (2 * safe_abs))
            grad_ah = grad_output_ah * (z / (2 * safe_abs))
        else:
            sgn = np.sign(z)
            grad_h = grad_output_h * (sgn / 2)
            grad_ah = grad_output_ah * (sgn / 2)

        return [(ensure_shape(grad_h, z_shape), ensure_shape(grad_ah, z_shape))]