from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class HyperbolicGradients:
    @staticmethod
    def sinh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        return [ensure_shape(
            grad_output * np.cosh(np.conjugate(inp)), 
            inp.shape if hasattr(inp, 'shape') else ()
        )]

    @staticmethod
    def cosh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        return [
            ensure_shape(
                grad_output * np.sinh(np.conjugate(inp)), 
                inp.shape if hasattr(inp, 'shape') else ()
            )
        ]
        
    @staticmethod
    def tanh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        tanh_conj_inp = np.tanh(np.conjugate(inp))
        return [ensure_shape(grad_output * (1 - tanh_conj_inp ** 2), inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def arcsinh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # Derivative of arcsinh(x) is 1 / sqrt(x^2 + 1)
        grad = grad_output / np.sqrt(x**2 + 1)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]
    
    @staticmethod
    def arccosh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]

        # Ensure input is in the valid domain for real values (x >= 1)
        if np.any(np.abs(x) < 1):
            raise ValueError("Input to arccosh must be >= 1 for real values.")
        
        # Derivative of arccosh(x) is 1 / sqrt(x^2 - 1)
        grad = grad_output / np.sqrt(x**2 - 1)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]

    @staticmethod
    def arctanh(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # Derivative of arctanh(x) is 1 / (1 - x^2)
        grad = grad_output / (1 - x**2)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]