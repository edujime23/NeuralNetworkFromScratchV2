from typing import Tuple, Any
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
    def cbrt(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        return [
            ensure_shape(
                grad_output / (3 * np.cbrt(np.conjugate(inp)) ** 2),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
        
    @staticmethod
    def heaviside(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        is_a_zero = (a == 0)  # Mask where a is zero
        grad_b = np.zeros_like(b, dtype=grad_output.dtype)  # Initialize grad_b to zero
        grad_b[is_a_zero] = 1.0  # Set grad_b to 1 where a is zero
        
        try:
            broadcasted_grad_output = np.broadcast_to(grad_output, is_a_zero.shape)  # Attempt to broadcast grad_output
            grad_b[is_a_zero] = broadcasted_grad_output[is_a_zero] * 1.0  # Broadcast grad_output for grad_b
        except ValueError:
            warnings.warn("Could not broadcast grad_output to the shape of a mask for b gradient. Setting b gradient to zero where a is zero.")
            grad_b[is_a_zero] = 0.0  # Fallback to zero if broadcasting fails

        return [ensure_shape(np.zeros_like(a, dtype=grad_output.dtype), a.shape if hasattr(a, 'shape') else ()), 
                ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]
        
    @staticmethod
    def clip(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp, min_val, max_val = inputs

        inp_shape = inp.shape if hasattr(inp, 'shape') else ()
        min_shape = np.shape(min_val)
        max_shape = np.shape(max_val)

        # Avoid complex warnings — define all gradients properly
        mask_in = (inp >= min_val) & (inp <= max_val)
        mask_min = inp < min_val
        mask_max = inp > max_val

        grad_inp = grad_output * mask_in
        grad_min = grad_output * mask_min
        grad_max = grad_output * mask_max

        return [
            ensure_shape(grad_inp, inp_shape),
            ensure_shape(grad_min, min_shape),
            ensure_shape(grad_max, max_shape),
        ]

    @staticmethod
    def sqrt(grad_output: np.typing.NDArray[Any],
            inputs: Tuple[np.typing.NDArray[Any], ...]):
        inp, = inputs

        # Avoid division by zero by adding epsilon
        sqrt_inp = np.sqrt(inp + np.finfo(inp.dtype).eps)

        grad_z    = grad_output / (2 * sqrt_inp)
        grad_zbar = np.zeros_like(grad_output)

        return [
            ensure_shape(grad_z,    inp.shape if hasattr(inp, 'shape') else ()),
            ensure_shape(grad_zbar, inp.shape if hasattr(inp, 'shape') else ()),
        ]

    @staticmethod
    def real(grad_output: np.typing.NDArray[Any],
            inputs: Tuple[np.typing.NDArray[Any], ...]):
        inp, = inputs

        grad_z    = grad_output * 0.5
        grad_zbar = grad_output * 0.5

        return [
            ensure_shape(grad_z,    inp.shape if hasattr(inp, 'shape') else ()),
            ensure_shape(grad_zbar, inp.shape if hasattr(inp, 'shape') else ()),
        ]

    @staticmethod
    def imag(grad_output: np.typing.NDArray[Any],
            inputs: Tuple[np.typing.NDArray[Any], ...]):
        inp, = inputs

        grad_z    = grad_output * (-1j / 2)
        grad_zbar = grad_output * ( 1j / 2)

        return [
            ensure_shape(grad_z,    inp.shape if hasattr(inp, 'shape') else ()),
            ensure_shape(grad_zbar, inp.shape if hasattr(inp, 'shape') else ()),
        ]
        
    @staticmethod
    def sign(grad_output: np.typing.NDArray[Any],
            inputs: Tuple[np.typing.NDArray[Any], ...]):
        inp, = inputs

        abs_inp = np.abs(inp)
        denom = 2 * abs_inp**3

        grad_z    = np.zeros_like(grad_output)  # holomorphic part is 0
        grad_zbar = -grad_output * inp**2 / denom  # anti-holomorphic part

        # Avoid division by zero (set gradient to 0 where |z| == 0)
        grad_zbar = np.where(abs_inp == 0, 0.0, grad_zbar)

        return [
            ensure_shape(grad_z, inp.shape),
            ensure_shape(grad_zbar, inp.shape),
        ]

    @staticmethod
    def conjugate(grad_output: np.typing.NDArray[Any],
                inputs: Tuple[np.typing.NDArray[Any], ...]):
        inp, = inputs
        
        grad_z    = np.zeros_like(grad_output)   # holomorphic channel
        grad_zbar = grad_output                  # anti-holomorphic channel

        return [ensure_shape(grad_z,    inp.shape), ensure_shape(grad_zbar, inp.shape)]