from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class ExponentialGradients:
    @staticmethod
    def exp(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        
        # The gradient of exp(x) with respect to x is exp(x)
        grad_inp = grad_output * np.exp(inp)
        
        return [ensure_shape(grad_inp, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def exp2(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # Compute the gradient using the chain rule
        grad = grad_output * np.log(2) * 2**np.conjugate(inp)
        return [ensure_shape(grad, inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def expm1(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        inp = inputs[0]
        # The derivative of expm1(x) = e^x - 1 is e^x
        return [
            ensure_shape(
                grad_output * np.exp(np.conjugate(inp)), 
                inp.shape if hasattr(inp, 'shape') else ()
            )
        ]