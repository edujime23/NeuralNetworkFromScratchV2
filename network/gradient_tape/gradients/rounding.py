from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class RoundingGradients:
    @staticmethod
    def floor(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        # warnings.warn("Gradient of floor is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]
    
    @staticmethod
    def ceil(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        # warnings.warn("Gradient of ceil is zero almost everywhere and undefined at integers.")
        inp = inputs[0]
        return [ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def round(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        # warnings.warn("Gradient of round is zero almost everywhere and undefined at .5 boundaries.")
        inp = inputs[0]
        return [ensure_shape(np.zeros_like(grad_output, dtype=grad_output.dtype), inp.shape if hasattr(inp, 'shape') else ())]

    @staticmethod
    def trunc(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # The gradient of truncation is zero everywhere, as it's non-differentiable at integer values.
        grad = np.zeros_like(x)

        # Ensure the gradient has the same shape as the input x
        return [ensure_shape(grad, x.shape)]