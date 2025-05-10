from typing import Tuple, Any
import warnings
import numpy as np
from .util import ensure_shape

class LogarithmicGradients:
    @staticmethod
    def log(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        
        # Ensure we handle both real and complex inputs correctly.
        grad_inp = grad_output / (np.conjugate(inp) + np.finfo(inp.dtype).eps)
        
        return [
            ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())
        ]
        
    @staticmethod
    def log2(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Gradient of log2(x) is 1 / (x * ln(2))
        return [
            ensure_shape(
                grad_output / (np.conjugate(inp) * np.log(2) + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def log10(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Gradient of log10(x) is 1 / (x * ln(10))
        return [
            ensure_shape(
                grad_output / (np.conjugate(inp) * np.log(10) + np.finfo(inp.dtype).eps),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
        
    @staticmethod
    def log1p(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        return [
            ensure_shape(
                grad_output / (np.conjugate(inp) + 1),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
        
    @staticmethod
    def logaddexp(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        # Handle complex input warning
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            warnings.warn("Gradient of logaddexp is not well-defined for complex inputs. Returning zero gradients.")
            return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        # Compute exp(a) and exp(b)
        exp_a, exp_b = np.exp(a), np.exp(b)
        # Compute the denominator (exp(a) + exp(b)) to avoid recalculating it
        denom = exp_a + exp_b + np.finfo(inputs[0].dtype).eps
        
        # Compute gradients with respect to a and b
        grad_a = grad_output * exp_a / denom
        grad_b = grad_output * exp_b / denom

        # Ensure the gradients have the correct shape
        return [ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]

    @staticmethod
    def logaddexp2(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        a, b = inputs
        # Handle complex input warning
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            warnings.warn("Gradient of logaddexp2 is not well-defined for complex inputs. Returning zero gradients.")
            return [np.zeros_like(a, dtype=grad_output.dtype), np.zeros_like(b, dtype=grad_output.dtype)]

        # Compute 2^a and 2^b
        exp2_a, exp2_b = 2**a, 2**b
        # Compute the denominator (2^a + 2^b) to avoid recalculating it
        denom = exp2_a + exp2_b + np.finfo(inputs[0].dtype).eps
        
        # Compute gradients with respect to a and b
        grad_a = grad_output * exp2_a / denom * np.log(2)
        grad_b = grad_output * exp2_b / denom * np.log(2)

        # Ensure the gradients have the correct shape
        return [ensure_shape(grad_a, a.shape if hasattr(a, 'shape') else ()),
                ensure_shape(grad_b, b.shape if hasattr(b, 'shape') else ())]
        