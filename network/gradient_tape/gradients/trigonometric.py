from .util import ensure_shape
from typing import Any, Tuple
import numpy as np

class TrigonometricGradients:
    @staticmethod
    def sin(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        
        # For real or complex inputs, we compute the gradient of sin(x) -> cos(x)
        grad_inp = grad_output * np.cos(np.conjugate(inp))  # Use conjugate for complex inputs
        
        return [ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def cos(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        
        # For real or complex inputs, we compute the gradient of cos(x) -> -sin(x)
        grad_inp = -grad_output * np.sin(np.conjugate(inp))  # Use conjugate for complex inputs
    
        return [ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def tan(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        return [
            ensure_shape(
                grad_output
                / (np.cos(np.conjugate(inp)) ** 2),
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
    
    @staticmethod
    def arcsin(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        conj_inp = np.conjugate(inp)
        denom = np.sqrt(1 - conj_inp ** 2)
        
        return [
            ensure_shape(
                grad_output / denom,
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def arccos(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        conj_inp = np.conjugate(inp)
        denom = np.sqrt(1 - conj_inp ** 2)

        return [
            ensure_shape(
                -grad_output / denom,
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]

    @staticmethod
    def arctan(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        conj_inp = np.conjugate(inp)
        denom = 1 + conj_inp ** 2

        return [
            ensure_shape(
                grad_output / denom,
                inp.shape if hasattr(inp, 'shape') else (),
            )
        ]
    
    @staticmethod
    def arctan2(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        y, x = inputs
        if np.iscomplexobj(y) or np.iscomplexobj(x):
            warnings.warn("Gradient of arctan2 is not well-defined for complex inputs. Returning zero gradients.")
            return [np.zeros_like(y, dtype=grad_output.dtype), np.zeros_like(x, dtype=grad_output.dtype)]

        denom = x ** 2 + y ** 2 + np.finfo(inputs[0].dtype).eps
        grad_y = grad_output * x / denom
        grad_x = -grad_output * y / denom

        return [
            ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ()),
            ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ())
        ]
        
    @staticmethod
    def sinc(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # Calculate the gradient of sinc(x) using the formula
        pi_x = np.pi * x
        grad_val = np.where(x != 0,
                            (np.cos(pi_x) * np.pi * x - np.sin(pi_x)) / (np.pi * x**2),
                            0)  # Handle x == 0 case
        
        # Multiply the gradient value by the grad_output (chain rule)
        grad = grad_output * grad_val
        
        # Ensure the result has the same shape as the input
        return [ensure_shape(grad, x.shape)]
        
    @staticmethod
    def hypot(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x, y = inputs
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            # For complex numbers, we calculate the Euclidean distance in the complex plane
            # The same formula holds for complex numbers, but we use np.abs for magnitude
            denom = np.sqrt(np.conj(x) * x + np.conj(y) * y + np.finfo(inputs[0].dtype).eps)
        else:
            # For real numbers, use the same formula
            denom = np.sqrt(x ** 2 + y ** 2 + np.finfo(inputs[0].dtype).eps)
        grad_x = grad_output * x / denom
        grad_y = grad_output * y / denom
        return [
            ensure_shape(grad_x, x.shape if hasattr(x, 'shape') else ()),
            ensure_shape(grad_y, y.shape if hasattr(y, 'shape') else ())
        ]