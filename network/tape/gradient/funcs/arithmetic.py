from typing import Tuple, Any, Union
import numpy as np
from .util import complex_log

class ArithmeticGradients:
    @staticmethod
    def add(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [grad_output_h, grad_output_h]

    @staticmethod
    def subtract(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [grad_output_h, -grad_output_h]

    @staticmethod
    def multiply(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [grad_output_h * b, grad_output_h * a]

    @staticmethod
    def divide(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [grad_output_h / b, -grad_output_h * a / (b * b)]

    @staticmethod
    def floor_divide(grad_output, inputs):
        a, b = inputs
        return [np.zeros_like(a), np.zeros_like(b)]

    @staticmethod
    def remainder(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        eps = np.finfo(a.dtype).eps
        safe_b = b + eps * (np.abs(b) < eps)
        val_floor = np.floor(np.divide(a, safe_b))
        return [grad_output_h, -grad_output_h * val_floor]

    @staticmethod
    def power(grad_output, inputs):
        base, exp = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        log_base = complex_log(base)
        grad_base = grad_output_h * exp * np.power(base, exp - 1)
        grad_exp = grad_output_h * np.power(base, exp) * log_base
        return [grad_base, grad_exp]

    @staticmethod
    def square(grad_output, inputs):
        z, = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [grad_output_h * (2 * z)]

    @staticmethod
    def float_power(grad_output, inputs):
        x, y = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        grad_x = grad_output_h * np.where(x != 0, y * x**(y - 1), 0)
        grad_y = grad_output_h * np.where(x > 0, x**y * np.log(x), 0)
        return [grad_x, grad_y]

    @staticmethod
    def fmod(grad_output, inputs):
        a, b = inputs
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        with np.errstate(divide='ignore', invalid='ignore'):
            trunc_div_result = np.trunc(np.divide(a, b))
        grad_a = grad_output_h * np.ones_like(a, dtype=grad_output_h.dtype)
        grad_b = grad_output_h * (-trunc_div_result.astype(grad_output_h.dtype))
        return [grad_a, grad_b]

    @staticmethod
    def negative(grad_output, inputs):
        z = inputs[0]
        grad_output_h = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        return [-grad_output_h]

    @staticmethod
    def absolute(grad_output, inputs):
        z = inputs[0]
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

        return [(grad_h, grad_ah)]
