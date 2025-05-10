from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class ArithmeticGradients:
    @staticmethod
    def add(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        grad_a = grad_output
        grad_b = grad_output

        return [ensure_shape(grad_a, np.shape(a)),
                ensure_shape(grad_b, np.shape(b))]
        
    @staticmethod
    def subtract(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        grad_a = grad_output
        grad_b = -grad_output

        return [ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def multiply(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs

        # If inputs are complex, apply conjugate, else use them as is for real inputs
        grad_a = grad_output * np.conjugate(b)  # Derivative of f(x, y) = x * y w.r.t. x is just y
        grad_b = grad_output * np.conjugate(a)  # Derivative w.r.t. y is just x

        return [
            ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
            ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())
        ]
        
    @staticmethod
    def divide(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        
        # Gradient with respect to a: grad_a = grad_output / np.conjugate(b)
        grad_a = grad_output / np.conjugate(b)
        
        # Gradient with respect to b: grad_b = -grad_output * np.conjugate(a) / (np.conjugate(b) * np.conjugate(b))
        grad_b = -grad_output * np.conjugate(a) / (np.conjugate(b) * np.conjugate(b))

        # Ensure the gradients have the correct shape based on the input shapes
        return [
            ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
            ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())
        ]
        
    @staticmethod
    def floor_divide(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        # warnings.warn("Gradient of floor_divide is zero almost everywhere and undefined at discontinuities.")
        a, b = inputs
        
        # The gradient of floor_divide is zero almost everywhere
        return [
            np.zeros_like(a, dtype=grad_output.dtype), 
            np.zeros_like(b, dtype=grad_output.dtype)
        ]
        
    @staticmethod
    def remainder(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs

        grad_a = grad_output
        grad_b = -grad_output * np.floor(np.divide(a, b))

        return [
            ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
            ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())
        ]
        
    @staticmethod
    def power(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        base, exp = inputs
        base_val = base if hasattr(base, 'shape') else np.array(base)
        exp_val = exp if hasattr(exp, 'shape') else np.array(exp)

        base_shape = base.shape if hasattr(base, 'shape') else ()
        exp_shape = exp.shape if hasattr(exp, 'shape') else ()

        # Gradient with respect to base: grad_base = exp * base^(exp - 1)
        grad_base = grad_output * exp_val * np.power(base_val, exp_val - 1)

        # Gradient with respect to exp: grad_exp = base^exp * log(base)
        # For complex base, we use the complex logarithm: np.log(base)
        # For real base, this will also work correctly as it defaults to the real logarithm.
        grad_exp = grad_output * np.log(np.abs(base_val) + np.finfo(base_val.dtype).eps) * np.power(base_val, exp_val)

        # If the exponent is a scalar (i.e., exp.shape == ()), we need to reduce the gradient of exp
        if exp_shape == ():
            grad_exp = np.sum(grad_exp)
        
        # If the base is a scalar (i.e., base.shape == ()), we need to reduce the gradient of base
        if base_shape == ():
            grad_base = np.sum(grad_base)

        return [
            ensure_shape(grad_base, base_shape),
            ensure_shape(grad_exp, exp_shape)
        ]
        
    @staticmethod
    def square(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp, = inputs
        
        # Compute the gradient with respect to the input
        grad_inp = grad_output * 2 * inp  # for complex inputs, use conjugate

        # Handle scalar case by summing the gradient
        if hasattr(inp, 'shape') and inp.shape == ():
            grad_inp = np.sum(grad_inp)
        
        # Ensure the gradient has the same shape as the input
        return [ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def float_power(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x, y = inputs
        
        # Gradient with respect to x: y * x^(y - 1)
        grad_x = grad_output * np.where(x != 0, y * x**(y - 1), 0)
        
        # Gradient with respect to y: x^y * log(x)
        grad_y = grad_output * np.where(x != 0, x**y * np.log(x), 0)
        
        # Ensure the gradients have the same shape as the inputs
        return [ensure_shape(grad_x, x.shape), ensure_shape(grad_y, y.shape)]
    
    @staticmethod
    def reciprocal(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Derivative of reciprocal(x) = -1/x^2, using conjugate for complex inputs
        return [
            ensure_shape(
                -grad_output / (np.conjugate(inp) ** 2 + np.finfo(inp.dtype).eps),  # Handling complex conjugate and small eps
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
    
    @staticmethod
    def fmod(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        # warnings.warn("Gradient of fmod is not well defined at discontinuities.")
        a, b = inputs

        with np.errstate(divide='ignore', invalid='ignore'):
            trunc_div_result = np.trunc(np.divide(a, b))

        return [ensure_shape(grad_output * np.ones_like(grad_output, dtype=grad_output.dtype), a.shape if hasattr(a, 'shape') else ()),
                ensure_shape(grad_output * -trunc_div_result.astype(grad_output.dtype), b.shape if hasattr(b, 'shape') else ())]
        
    @staticmethod
    def negative(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        
        # The derivative of -x with respect to x is -1, so we multiply by -1.
        grad_inp = -grad_output
        
        return [ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def absolute(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):

        inp = inputs[0]
        inp_shape = inp.shape if hasattr(inp, 'shape') else ()

        if np.iscomplexobj(inp):
            abs_inp = np.abs(inp)
            denom = np.where(abs_inp == 0, 1.0, abs_inp)
            grad_inp = grad_output * inp / denom
        else:

            grad_inp = grad_output * np.sign(inp)

        return [ensure_shape(grad_inp, inp_shape)]