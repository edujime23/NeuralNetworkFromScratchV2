from typing import Tuple, Any
import numpy as np
from .util import ensure_shape

class ShapeGradients:
    @staticmethod
    def reshape(grad_output: np.typing.NDArray[Any], inputs: Tuple[(np.typing.NDArray[Any], ...)]):
        x = inputs[0]
        
        # The gradient needs to be reshaped to match the original shape of x
        grad = grad_output.reshape(x.shape)

        return [grad]